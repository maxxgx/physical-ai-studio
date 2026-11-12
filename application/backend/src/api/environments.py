from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from fastapi.exceptions import HTTPException
from fastapi.responses import Response
from physicalai.config import Config

from api.dependencies import RobotClientFactoryDep, SimulationManagerDep, get_environment_service
from api.robot_simulation import resolve_export_cameras
from schemas.environment import (
    Environment,
    EnvironmentWithRelations,
    TeleoperatorAutopilotWithRobot,
    TeleoperatorRobotWithRobot,
)
from schemas.robot import ReadableRobot
from services.environment_service import EnvironmentService
from services.runtime_export import RUNTIME_FPS, build_runtime_config, runtime_config_change_me

router = APIRouter(prefix="/api/projects/{project_id}/environments", tags=["Project Environments"])

ProjectID = UUID


@router.get("")
async def list_project_environments(
    project_id: ProjectID,
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
) -> list[Environment]:
    """Fetch all environments."""
    return await environment_service.get_environment_list(project_id)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project_environment(
    project_id: ProjectID,
    environment: Environment,
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
) -> Environment:
    """Create a new environment."""
    return await environment_service.create_environment(project_id, environment)


@router.get("/{environment_id}")
async def get_project_environment(
    project_id: ProjectID,
    environment_id: UUID,
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
) -> EnvironmentWithRelations:
    """Get environment by id with eager loaded robots and cameras."""
    return await environment_service.get_environment_by_id(project_id, environment_id)


@router.get("/{environment_id}/runtime-config")
async def get_runtime_config(
    request: Request,
    project_id: ProjectID,
    environment_id: UUID,
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
    robot_client_factory: RobotClientFactoryDep,
    simulations: SimulationManagerDep,
) -> Response:
    """Download a runnable physicalai teleoperation configuration.

    With the simulation's autopilot as teleoperator, the leader is the virtual leader a session
    would build (``SimulationManager.autopilot_leader``), and the file says how to switch the
    autopilot to leading, which a session does itself.
    """
    environment = await environment_service.get_environment_by_id(project_id, environment_id)
    if len(environment.robots) != 1:
        raise HTTPException(status_code=400, detail="Runtime export requires exactly one follower robot")
    relation = environment.robots[0]
    header = ""
    leader: ReadableRobot | None = None
    if isinstance(relation.tele_operator, TeleoperatorAutopilotWithRobot):
        leader = simulations.autopilot_leader(relation.robot)
        header = _autopilot_header(leader)
    elif isinstance(relation.tele_operator, TeleoperatorRobotWithRobot):
        leader = relation.tele_operator.robot
    if leader is None:
        raise HTTPException(status_code=400, detail="Runtime teleoperation export requires a leader robot")

    try:
        document = await build_runtime_config(
            follower=relation.robot,
            leader=leader,
            cameras=await resolve_export_cameras(request, environment_service.session, project_id, environment.cameras),
            robot_factory=robot_client_factory,
            fps=RUNTIME_FPS,
            # The rig this describes does not have to be attached to export it.
            allow_stored_port=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    unresolved = runtime_config_change_me(document)
    comments = "".join(f"# CHANGE_ME: replace machine-specific device path {path}\n" for path in unresolved)
    return Response(
        content=header + comments + Config.from_dict(document).to_yaml(),
        media_type="application/yaml",
        headers={"Content-Disposition": 'attachment; filename="runtime.yaml"'},
    )


def _autopilot_header(leader: ReadableRobot) -> str:
    """Comment lines for an export whose leader is the simulation's autopilot.

    The address is built from the leader's validated port, never from the follower's stored
    ``http_url``: that is free text, which must not end up in a comment or a shell command.
    """
    port = int(leader.payload.http_port)
    return (
        "# The leader is the simulation's autopilot. It leads only in leader mode: set Autopilot to\n"
        "# 'Virtual leader' in the simulation's viewer before running, or run\n"
        f"#   curl -X POST http://127.0.0.1:{port}/autopilot -H 'Content-Type: application/json' "
        '-d \'{"mode": "leader"}\'\n'
    )


@router.put("/{environment_id}")
async def update_project_environment(
    project_id: ProjectID,
    environment_id: UUID,
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
    environment: Environment,
) -> EnvironmentWithRelations:
    """Update environment."""
    environment_with_id = environment.model_copy(update={"id": environment_id})

    return await environment_service.update_environment(
        project_id,
        environment_with_id,
    )


@router.delete("/{environment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project_environment(
    project_id: ProjectID,
    environment_id: UUID,
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
) -> None:
    """Delete an environment."""
    await environment_service.delete_environment(project_id, environment_id)
