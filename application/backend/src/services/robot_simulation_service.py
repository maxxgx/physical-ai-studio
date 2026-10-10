# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""What a simulated robot's simulation serves (viewer, speed, cameras), and adding its cameras."""

import asyncio
from typing import Any
from uuid import UUID, uuid4

import httpx
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import (
    ResourceNotFoundError,
    ResourceType,
    SimulationInvalidRequestError,
    SimulationNotRunningError,
    SimulationNotSupportedError,
)
from repositories.project_environment_repo import ProjectEnvironmentRepository
from robots.catalog.registry import RobotCatalogRegistry
from robots.robot_service import RobotService
from schemas.environment import CameraEnvironmentConfiguration
from schemas.project_camera import IPCamera, IPCameraPayload
from schemas.robot import UnavailableRobot
from schemas.simulation import RobotSimulation, SimulationCamera, SimulationStatus
from services.environment_service import EnvironmentService
from services.project_camera_service import ProjectCameraService
from services.simulation_manager import SimulationManager

PROBE_TIMEOUT_S = 1.0


class RobotSimulationService:
    def __init__(
        self,
        session: AsyncSession,
        registry: RobotCatalogRegistry,
        manager: SimulationManager,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._session = session
        self._registry = registry
        self._manager = manager
        self._transport = transport

    async def get(self, project_id: UUID, robot_id: UUID) -> RobotSimulation:
        """Return the state of a simulated robot's simulation, probing it over HTTP.

        Raises:
            ResourceNotFoundError: The robot does not exist.
            SimulationNotSupportedError: The robot type has no simulation.
        """
        robot = await RobotService(self._session, self._registry).get_robot_by_id(project_id, robot_id)
        owner_name = self._owner_name(robot)

        state = self._manager.state()
        studio = state.robot_id == robot_id and state.status is SimulationStatus.RUNNING
        http_url = state.http_url if studio else getattr(robot.payload, "http_url", None)
        if not isinstance(http_url, str):
            return RobotSimulation(running=False, owner_name=owner_name)

        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S, transport=self._transport) as client:
            root, health, cameras = await asyncio.gather(
                _get_json(client, f"{http_url}/"),
                _get_json(client, f"{http_url}/health"),
                _get_json(client, f"{http_url}/cameras"),
            )
        answers = isinstance(root, dict) and root.get("service") == owner_name
        if not answers and not studio:
            return RobotSimulation(running=False, owner_name=owner_name)

        root = root if isinstance(root, dict) else {}
        health = health if isinstance(health, dict) else {}
        timing = health.get("timing")
        real_time_factor = timing.get("real_time_factor") if isinstance(timing, dict) else None
        viewer_url = state.viewer_url if studio else root.get("viewer_url")
        return RobotSimulation(
            running=True,
            started_by="studio" if studio else "external",
            owner_name=owner_name,
            http_url=http_url,
            viewer_url=viewer_url if isinstance(viewer_url, str) else None,
            scene=_str_or_none(health.get("scene", root.get("scene"))),
            real_time_factor=float(real_time_factor) if isinstance(real_time_factor, int | float) else None,
            cameras=_cameras(http_url, cameras),
        )

    async def add_cameras(
        self, project_id: UUID, environment_id: UUID, robot_id: UUID, names: list[str]
    ) -> list[IPCamera]:
        """Create IP cameras for some of a running simulation's cameras and add them to an environment.

        Raises:
            ResourceNotFoundError: The robot or the environment does not exist.
            SimulationNotSupportedError: The robot type has no simulation.
            SimulationNotRunningError: The robot's simulation does not run.
            SimulationInvalidRequestError: The simulation has no camera with one of the names.
        """
        environment = await ProjectEnvironmentRepository(self._session, project_id, self._registry).get_by_id(
            environment_id
        )
        if environment is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment_id))
        simulation = await self.get(project_id, robot_id)
        if not simulation.running:
            raise SimulationNotRunningError(f"The simulation {simulation.owner_name!r} is not running.")
        by_name = {camera.name: camera for camera in simulation.cameras}
        unknown = [name for name in names if name not in by_name]
        if unknown:
            raise SimulationInvalidRequestError(
                f"The simulation has no camera named {', '.join(unknown)}; it has {', '.join(by_name) or 'none'}."
            )

        robot = await RobotService(self._session, self._registry).get_robot_by_id(project_id, robot_id)
        camera_service = ProjectCameraService(self._session, self._registry)
        created: list[IPCamera] = []
        for name in dict.fromkeys(names):
            source = by_name[name]
            camera_name = f"{robot.name} {name}"
            camera = IPCamera(
                id=uuid4(),
                name=camera_name,
                fingerprint={"url": source.stream_url},
                # Required by the database; the camera form also uses the name when there is no hardware.
                hardware_name=camera_name,
                payload=IPCameraPayload(
                    url=source.stream_url, width=source.width, height=source.height, fps=source.fps
                ),
            )
            await camera_service.create_camera(project_id, camera)
            created.append(camera)

        added = [CameraEnvironmentConfiguration(camera_id=camera.id) for camera in created]
        await EnvironmentService(self._session, self._registry).update_environment(
            project_id,
            environment.model_copy(update={"cameras": [*environment.cameras, *added]}),
        )
        return created

    def _owner_name(self, robot: Any) -> str:
        definition = None if isinstance(robot, UnavailableRobot) else self._registry.get_definition(robot.type)
        launch = getattr(definition, "simulation", None)
        if launch is None:
            raise SimulationNotSupportedError(f"Robot {robot.name!r} ({robot.type}) has no simulation.")
        return launch.owner_name(robot.payload)


async def _get_json(client: httpx.AsyncClient, url: str) -> Any:
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as error:
        logger.debug("Simulation probe {} failed: {}", url, error)
        return None


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _cameras(http_url: str, cameras: object) -> list[SimulationCamera]:
    if not isinstance(cameras, list):
        return []
    result: list[SimulationCamera] = []
    for camera in cameras:
        if not isinstance(camera, dict) or camera.get("failed"):
            continue
        try:
            result.append(
                SimulationCamera(
                    name=camera["name"],
                    width=camera["width"],
                    height=camera["height"],
                    fps=camera["fps"],
                    source=camera.get("source", ""),
                    stream_url=f"{http_url}{camera['stream_url']}",
                    snapshot_url=f"{http_url}{camera['snapshot_url']}",
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            logger.debug("Ignoring a malformed simulation camera {}: {}", camera, error)
    return result
