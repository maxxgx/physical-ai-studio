# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Routes of a robot's simulation against a stub service: status codes and error codes."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import api.robot_simulation as module
from api.dependencies import get_robot_simulation_service
from api.robot_simulation import resolve_export_cameras
from exceptions import (
    ResourceNotFoundError,
    ResourceType,
    SimulationInvalidRequestError,
    SimulationNotRunningError,
    SimulationNotServingError,
    SimulationNotSupportedError,
)
from main import app
from schemas.project_camera import IPCamera, SimulationCamera
from schemas.simulation import RobotSimulation, SimulationRelayReply

PROJECT_ID = uuid4()
ROBOT_ID = uuid4()
ENVIRONMENT_ID = uuid4()


class StubService:
    error: Exception | None = None

    async def get(self, project_id, robot_id) -> RobotSimulation:
        if self.error is not None:
            raise self.error
        return RobotSimulation(running=True, started_by="external", owner_name="sim", scene="yahtzee")

    async def set_recording(self, project_id, robot_id, request) -> None:
        self.recording = request
        if self.error is not None:
            raise self.error

    async def relay_recording(self, project_id, robot_id, request) -> SimulationRelayReply:
        self.relayed = request
        if self.error is not None:
            raise self.error
        return SimulationRelayReply(active=True, owner=True, commands=[])

    async def add_cameras(self, project_id, environment_id, robot_id, names, camera_names) -> list:
        self.added = (names, camera_names)
        if self.error is not None:
            raise self.error
        return []


@pytest.fixture
def stub() -> Iterator[StubService]:
    stub = StubService()
    app.dependency_overrides[get_robot_simulation_service] = lambda: stub
    yield stub
    app.dependency_overrides.clear()


def test_get_returns_the_robot_simulation(stub: StubService) -> None:
    response = TestClient(app).get(f"/api/projects/{PROJECT_ID}/robots/{ROBOT_ID}/simulation")

    assert response.status_code == 200
    assert response.json()["started_by"] == "external"


def test_add_cameras_returns_201(stub: StubService) -> None:
    response = TestClient(app).post(
        f"/api/projects/{PROJECT_ID}/environments/{ENVIRONMENT_ID}/simulation-cameras",
        json={"robot_id": str(ROBOT_ID), "cameras": ["wrist"], "names": {"wrist": "Wrist"}},
    )

    assert response.status_code == 201
    assert stub.added == (["wrist"], {"wrist": "Wrist"})


def test_add_cameras_names_are_optional(stub: StubService) -> None:
    response = TestClient(app).post(
        f"/api/projects/{PROJECT_ID}/environments/{ENVIRONMENT_ID}/simulation-cameras",
        json={"robot_id": str(ROBOT_ID), "cameras": ["wrist"]},
    )

    assert response.status_code == 201
    assert stub.added == (["wrist"], {})


def test_add_cameras_needs_at_least_one_camera(stub: StubService) -> None:
    response = TestClient(app).post(
        f"/api/projects/{PROJECT_ID}/environments/{ENVIRONMENT_ID}/simulation-cameras",
        json={"robot_id": str(ROBOT_ID), "cameras": []},
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    ("error", "status_code", "error_code"),
    [
        (SimulationNotSupportedError("no simulation"), 400, "simulation_not_supported"),
        (ResourceNotFoundError(ResourceType.ROBOT, ROBOT_ID), 404, "Robot_not_found"),
        (SimulationNotRunningError("not running"), 503, "simulation_not_running"),
        (SimulationInvalidRequestError("no camera named x"), 400, "simulation_invalid_request"),
    ],
)
def test_errors_map_to_their_codes(stub: StubService, error: Exception, status_code: int, error_code: str) -> None:
    stub.error = error

    response = TestClient(app).post(
        f"/api/projects/{PROJECT_ID}/environments/{ENVIRONMENT_ID}/simulation-cameras",
        json={"robot_id": str(ROBOT_ID), "cameras": ["wrist"]},
    )

    assert response.status_code == status_code
    assert response.json()["error_code"] == error_code


PLAIN = IPCamera(
    id=uuid4(), name="Top", fingerprint={"url": "rtsp://x"}, hardware_name="Top", payload={"url": "rtsp://x"}
)
SIMULATED = SimulationCamera(
    id=uuid4(),
    name="Wrist",
    fingerprint={"robot_id": str(ROBOT_ID), "camera": "wrist"},
    hardware_name="wrist",
    payload={"robot_id": ROBOT_ID, "camera": "wrist"},
)


async def test_an_export_without_simulation_cameras_needs_no_simulation_manager() -> None:
    # No app state: looking up the manager would fail.
    assert await resolve_export_cameras(SimpleNamespace(), None, PROJECT_ID, [PLAIN]) == [PLAIN]  # type: ignore[arg-type]


async def test_an_export_resolves_its_simulation_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    class FakeService:
        def __init__(self, session: Any, registry: Any, manager: Any) -> None:
            calls.append(manager)

        async def resolve_cameras(self, project_id: Any, cameras: list[Any]) -> list[Any]:
            return [PLAIN for _ in cameras]

    manager = object()
    monkeypatch.setattr(module, "RobotSimulationService", FakeService)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(simulation_manager=manager)))

    resolved = await resolve_export_cameras(request, None, PROJECT_ID, [SIMULATED])  # type: ignore[arg-type]

    assert resolved == [PLAIN]
    assert calls == [manager]


def test_the_recording_relay_passes_the_exchange_on(stub: StubService) -> None:
    url = f"/api/projects/{PROJECT_ID}/robots/{ROBOT_ID}/simulation/recording-relay"
    session = {"dataset": True, "recording": False, "teleop": True}

    response = TestClient(app).post(url, json={"client": "page-1", "session": session})

    assert response.status_code == 200
    assert response.json() == {"active": True, "owner": True, "commands": []}
    assert stub.relayed.session.teleop is True
    for bad in ({"session": session}, {"client": "page-1", "session": {**session, "teleop": "maybe"}}):
        assert TestClient(app).post(url, json=bad).status_code in {400, 422}
    stub.error = SimulationNotServingError("update physicalai-mujoco-plugin")
    response = TestClient(app).post(url, json={"client": "page-1", "session": session})
    assert response.status_code >= 400
    assert response.json()["error_code"] == "simulation_not_serving"


def test_automatic_recording_is_started_with_validated_settings(stub: StubService) -> None:
    url = f"/api/projects/{PROJECT_ID}/robots/{ROBOT_ID}/simulation/recording"

    response = TestClient(app).post(url, json={"enabled": True, "max_episodes": 5, "keep": "all"})

    assert response.status_code == 204
    assert (stub.recording.max_episodes, stub.recording.keep) == (5, "all")
    for bad in (
        {"enabled": True, "max_attempts": 9},
        {"enabled": True, "recovery_fraction": 0.7},
        {"enabled": True, "task": ""},
    ):
        assert TestClient(app).post(url, json=bad).status_code in {400, 422}, bad
