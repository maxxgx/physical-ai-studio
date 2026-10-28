# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Editing a camera added from a simulation: whether it stays its simulated robot's."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_camera_claim_registry, get_camera_service, get_robot_service
from exceptions import ResourceNotFoundError, ResourceType
from main import app
from schemas.project_camera import IPCamera
from services.simulation_manager import _move_simulation_cameras

PROJECT_ID = uuid4()
A, B = uuid4(), uuid4()
A_URL, B_URL = "http://127.0.0.1:50001", "http://127.0.0.1:50002"


class FakeCameras:
    def __init__(self, camera: IPCamera) -> None:
        self.cameras = {camera.id: camera}

    async def get_camera_by_id(self, project_id: UUID, camera_id: UUID) -> IPCamera:
        return self.cameras[camera_id]

    async def get_camera_list(self, project_id: UUID) -> list[IPCamera]:
        return list(self.cameras.values())

    async def update_camera(self, project_id: UUID, camera: IPCamera) -> IPCamera:
        self.cameras[camera.id] = camera
        return camera


class FakeRobots:
    robots = {
        A: SimpleNamespace(id=A, payload=SimpleNamespace(http_url=A_URL)),
        B: SimpleNamespace(id=B, payload=SimpleNamespace(http_url=B_URL)),
    }

    async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> Any:
        if robot_id not in self.robots:
            raise ResourceNotFoundError(ResourceType.ROBOT, str(robot_id))
        return self.robots[robot_id]


@pytest.fixture
def cameras() -> Iterator[FakeCameras]:
    url = f"{A_URL}/cameras/wrist/mjpeg"
    camera = IPCamera(
        id=uuid4(),
        name="A wrist",
        fingerprint={"url": url},
        hardware_name="A wrist",
        payload={"url": url, "simulation_robot_id": A},
    )
    fake = FakeCameras(camera)
    app.dependency_overrides[get_camera_service] = lambda: fake
    app.dependency_overrides[get_robot_service] = lambda: FakeRobots()
    app.dependency_overrides[get_camera_claim_registry] = lambda: SimpleNamespace(holder_of=lambda _: None)
    yield fake
    app.dependency_overrides.clear()


def edit_url(cameras: FakeCameras, url: str) -> IPCamera:
    """Save the camera form with a new stream URL; the form sends back the rest of the payload as it was."""
    (camera,) = cameras.cameras.values()
    body = camera.model_dump(mode="json")
    body["fingerprint"] = {"url": url}
    body["payload"]["url"] = url
    response = TestClient(app).put(f"/api/projects/{PROJECT_ID}/cameras/{camera.id}", json=body)
    assert response.status_code == 200
    return cameras.cameras[camera.id]


async def test_a_camera_pointed_at_another_simulation_stays_there_when_its_robot_restarts(
    cameras: FakeCameras,
) -> None:
    edited = edit_url(cameras, f"{B_URL}/cameras/wrist/mjpeg")

    await _move_simulation_cameras(cameras, PROJECT_ID, A, "http://127.0.0.1:50003")  # type: ignore[arg-type]

    assert edited.payload.simulation_robot_id is None
    assert cameras.cameras[edited.id].payload.url == f"{B_URL}/cameras/wrist/mjpeg"


async def test_a_camera_pointed_at_another_stream_of_its_robot_still_follows_it(cameras: FakeCameras) -> None:
    edited = edit_url(cameras, f"{A_URL}/cameras/overview/mjpeg")

    await _move_simulation_cameras(cameras, PROJECT_ID, A, "http://127.0.0.1:50003")  # type: ignore[arg-type]

    assert edited.payload.simulation_robot_id == A
    assert cameras.cameras[edited.id].payload.url == "http://127.0.0.1:50003/cameras/overview/mjpeg"


def test_saving_a_camera_does_not_let_the_request_choose_its_robot(cameras: FakeCameras) -> None:
    (camera,) = cameras.cameras.values()
    body = camera.model_dump(mode="json")
    body["payload"]["simulation_robot_id"] = str(B)

    response = TestClient(app).put(f"/api/projects/{PROJECT_ID}/cameras/{camera.id}", json=body)

    assert response.status_code == 200
    assert cameras.cameras[camera.id].payload.simulation_robot_id == A
