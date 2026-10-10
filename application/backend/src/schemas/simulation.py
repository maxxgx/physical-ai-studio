# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""State of the simulation Studio starts and supervises for a simulated robot."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class SimulationStatus(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class SimulationPhase(StrEnum):
    FETCH = "fetch"
    CONNECT = "connect"
    LOAD = "load"
    CAMERAS = "cameras"


class SimulationDownload(BaseModel):
    bytes: int = Field(..., description="Bytes of the robot model downloaded so far")
    total: int | None = Field(None, description="Size of the download in bytes, if known")


class SimulationError(BaseModel):
    message: str = Field(..., description="Why the simulation failed")
    phase: SimulationPhase | None = Field(
        None, description="Startup phase the simulation failed in, if it had not started"
    )
    exit_code: int | None = Field(None, description="Exit code of the simulation process, once it has exited")


class Simulation(BaseModel):
    """The simulation started by Studio. There is at most one; ``idle`` means none."""

    status: SimulationStatus = SimulationStatus.IDLE
    project_id: UUID | None = None
    robot_id: UUID | None = None
    robot_type: str | None = None
    owner_name: str | None = Field(None, description="Name the simulation publishes its robot under")
    scene: str | None = None
    seed: int | None = None
    phase: SimulationPhase | None = Field(None, description="Current startup phase while starting")
    download: SimulationDownload | None = Field(None, description="Robot model download progress while fetching")
    pid: int | None = Field(None, description="Process id of the simulation once it is running")
    started_at: datetime | None = None
    http_url: str | None = Field(None, description="HTTP API of the running simulation")
    viewer_url: str | None = Field(None, description="3D viewer of the running simulation")
    cameras: list[str] = Field(default_factory=list, description="Cameras the running simulation streams")
    error: SimulationError | None = None


class StartSimulationRequest(BaseModel):
    project_id: UUID
    robot_id: UUID
    scene: str = Field(..., description="Id of one of the robot type's simulation scenes")
    seed: int | None = Field(None, ge=0, description="Fixed seed for the scene, or null for a random layout")
    replace: bool = Field(False, description="Stop the simulation Studio is running, if any, before starting this one")


class SimulationLogs(BaseModel):
    lines: list[str] = Field(..., description="Latest log lines of the current or last simulation, oldest first")
