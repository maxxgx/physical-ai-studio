# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""State of the simulation Studio starts and supervises for a simulated robot."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
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
    viewer_proxy_url: str | None = Field(
        None, description="The 3D viewer through Studio, for the browser: a path on Studio's own address"
    )
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


SimulationCameraSource = Literal["override", "model", "default", "scene"]
"""Where a simulation camera comes from, as the simulation reports it."""


class SimulationCameraInfo(BaseModel):
    """A camera a running simulation reports."""

    name: str
    width: int
    height: int
    fps: int
    source: SimulationCameraSource = Field(
        ..., description="Where the camera comes from: override, model or default for a robot camera, else scene"
    )
    stream_url: str = Field(..., description="MJPEG stream of the camera")
    snapshot_url: str = Field(..., description="Latest frame of the camera as a JPEG")
    snapshot_proxy_url: str | None = Field(
        None, description="The latest frame through Studio, for the browser: a path on Studio's own address"
    )


class SimulationAutopilot(BaseModel):
    """The simulation's autopilot, a scripted demonstrator, as the simulation's ``/health`` reports it.

    ``mode`` is ``off``, ``drive`` (it moves the arm itself) or ``leader`` (it publishes targets
    that a teleoperating client sends back). The rest is set while it runs, as far as the scene
    reports it: what it works on, and for an episode its attempt and how it went (``running``,
    ``success``, ``failed`` or ``stalled``, with why it failed).
    """

    available: bool
    unavailable_reason: str | None = None
    mode: str = "off"
    phase: str | None = None
    target: str | None = None
    attempt: int | None = None
    max_attempts: int | None = None
    outcome: str | None = None
    failure: str | None = None


class SimulationRecording(BaseModel):
    """The simulation's automatic recording ("Record in Studio" in its viewer), as its ``/health`` reports it.

    While it runs (``phase`` neither ``off``, ``done``, ``stopped`` nor ``error``), Studio's
    recording page relays it: see ``SimulationRelayRequest``.
    """

    phase: str = "off"
    message: str = ""
    saved: int = 0
    discarded: int = 0
    max_episodes: int = 0


class RobotSimulation(BaseModel):
    """Whether a simulated robot's simulation runs, who started it, and what it serves."""

    running: bool
    started_by: Literal["studio", "external"] | None = None
    owner_name: str = Field(..., description="Name the robot attaches to")
    http_url: str | None = None
    viewer_url: str | None = None
    viewer_proxy_url: str | None = Field(
        None, description="The 3D viewer through Studio, for the browser: a path on Studio's own address"
    )
    scene: str | None = None
    cameras: list[SimulationCameraInfo] = Field(default_factory=list)
    autopilot: SimulationAutopilot | None = Field(None, description="None when the simulation reports no autopilot")
    recording: SimulationRecording | None = Field(
        None, description="None when the simulation reports no automatic recording"
    )


class ProjectRobotSimulation(RobotSimulation):
    """A running simulation of one of a project's robots."""

    robot_id: UUID
    robot_name: str
    robot_type: str


class AddSimulationCamerasRequest(BaseModel):
    robot_id: UUID = Field(..., description="The simulated robot whose cameras to add")
    cameras: list[str] = Field(..., min_length=1, description="Names of the simulation's cameras to add")
    names: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Names for the project cameras this call creates, by simulation camera; a missing or blank one "
            "becomes '<robot name> <camera>'. A camera the project has already keeps its name."
        ),
    )


def simulation_proxy_path(project_id: UUID, robot_id: UUID) -> str:
    """Return the path under which Studio serves a robot's simulation to the browser.

    The simulation runs on its own ports, which the browser may not reach (e.g. in Docker, where only
    Studio's port is published): ``<path>/viewer/`` serves its 3D viewer, ``<path>/http/...`` some of
    its HTTP API.
    """
    return f"/api/projects/{project_id}/robots/{robot_id}/simulation"


class SimulationRelaySession(BaseModel):
    """The recording page's session, as it relays it to the simulation."""

    session_id: str | None = Field(None, max_length=64, description="The runtime session; a run stays on one")
    dataset_id: str | None = Field(None, max_length=64, description="The dataset loaded; a run stays on one")
    dataset: bool = Field(..., description="Whether a dataset is loaded to record into")
    recording: bool = Field(..., description="Whether an episode is being recorded")
    teleop: bool = Field(..., description="Whether teleoperation drives the follower")
    follower: str | None = Field(None, max_length=200)
    leader: str | None = Field(None, max_length=200)


class SimulationRelayResult(BaseModel):
    """How a relayed command went in the session."""

    request_id: str = Field(..., min_length=1, max_length=64, description="The simulation's id of the command")
    ok: bool
    error: str | None = Field(None, max_length=2000)


class SimulationRelayRequest(BaseModel):
    """One exchange of the recording page with the simulation's automatic recording.

    The first page to exchange relays a run; each sends its own ``client`` id. A command comes
    with every exchange until its result does, so the page runs each ``request_id`` once and sends
    its result until the command no longer comes.
    """

    client: str = Field(..., min_length=1, max_length=64, description="The relaying page's own id")
    session: SimulationRelaySession
    results: list[SimulationRelayResult] = Field(default_factory=list, max_length=16)


class SimulationRelayCommand(BaseModel):
    """A session command the simulation asks the recording page to run, as its own buttons do."""

    command: Literal["start_episode", "save_episode", "discard_episode"]
    request_id: str = Field(..., min_length=1, max_length=64)
    task: str | None = Field(None, max_length=200, description="With ``start_episode``: the episode's task")


class SimulationRelayReply(BaseModel):
    active: bool = Field(..., description="Whether the simulation's automatic recording runs, so keep relaying")
    owner: bool = Field(False, description="Whether this page relays the run; another page that does gets them")
    commands: list[SimulationRelayCommand] = Field(
        default_factory=list, description="Commands the simulation waits for the results of, oldest first"
    )
