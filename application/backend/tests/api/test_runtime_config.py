from types import SimpleNamespace
from uuid import uuid4

import pytest
import yaml
from fastapi import HTTPException
from physicalai.config import Config

from api.environments import get_runtime_config
from robots.catalog.registry import RobotCatalogRegistry
from robots.robot_client_factory import RobotClientFactory
from schemas.environment import EnvironmentWithRelations
from services.runtime_export import build_runtime_config
from services.simulation_manager import SimulationManager


def _environment(*, with_leader: bool) -> EnvironmentWithRelations:
    teleoperator = (
        {
            "type": "robot",
            "robot_id": str(uuid4()),
            "robot": {
                "id": str(uuid4()),
                "name": "Leader",
                "type": "SO101_Leader",
                "payload": {"connection_string": "/dev/ttyACM1", "serial_number": "leader"},
            },
        }
        if with_leader
        else {"type": "none"}
    )
    return EnvironmentWithRelations.model_validate(
        {
            "id": str(uuid4()),
            "name": "Test",
            "robots": [
                {
                    "robot": {
                        "id": str(uuid4()),
                        "name": "Follower",
                        "type": "SO101_Follower",
                        "payload": {"connection_string": "/dev/ttyACM0", "serial_number": "follower"},
                    },
                    "tele_operator": teleoperator,
                }
            ],
            "cameras": [],
        }
    )


async def test_runtime_config_download_is_yaml_attachment(mocker) -> None:
    environment = _environment(with_leader=True)
    environment_service = SimpleNamespace(
        get_environment_by_id=mocker.AsyncMock(return_value=environment), session=None
    )
    document = Config(
        "physicalai.runtime.RobotRuntime",
        {"robot": {"class_path": "physicalai.robot.so101.SO101", "init_args": {}}, "fps": 30.0},
    ).to_dict()
    build = mocker.patch("api.environments.build_runtime_config", return_value=document)

    response = await get_runtime_config(
        request=SimpleNamespace(),
        project_id=uuid4(),
        environment_id=environment.id,
        environment_service=environment_service,
        robot_client_factory=mocker.MagicMock(),
        simulations=mocker.MagicMock(),
    )

    assert response.media_type == "application/yaml"
    assert response.headers["content-disposition"] == 'attachment; filename="runtime.yaml"'
    assert b"physicalai.runtime.RobotRuntime" in response.body
    # A config can be exported for a rig that is not plugged in right now.
    assert build.call_args.kwargs["allow_stored_port"] is True


async def test_runtime_config_download_requires_leader(mocker) -> None:
    environment = _environment(with_leader=False)
    environment_service = SimpleNamespace(
        get_environment_by_id=mocker.AsyncMock(return_value=environment), session=None
    )

    with pytest.raises(HTTPException, match="leader robot"):
        await get_runtime_config(
            request=SimpleNamespace(),
            project_id=uuid4(),
            environment_id=environment.id,
            environment_service=environment_service,
            robot_client_factory=mocker.MagicMock(),
            simulations=mocker.MagicMock(),
        )


def _autopilot_environment(http_url: str) -> EnvironmentWithRelations:
    return EnvironmentWithRelations.model_validate(
        {
            "id": str(uuid4()),
            "name": "Sim bench",
            "robots": [
                {
                    "robot": {
                        "id": str(uuid4()),
                        "name": "Sim arm",
                        "type": "MuJoCo_SO101_Follower",
                        "payload": {"http_url": http_url},
                    },
                    "tele_operator": {"type": "autopilot"},
                }
            ],
            "cameras": [],
        }
    )


async def _export(mocker, environment: EnvironmentWithRelations) -> str:
    environment_service = SimpleNamespace(
        get_environment_by_id=mocker.AsyncMock(return_value=environment), session=None
    )
    response = await get_runtime_config(
        request=SimpleNamespace(),
        project_id=uuid4(),
        environment_id=environment.id,
        environment_service=environment_service,
        robot_client_factory=RobotClientFactory(robot_manager=mocker.MagicMock()),
        simulations=SimulationManager(RobotCatalogRegistry()),
    )
    return response.body.decode()


async def test_runtime_config_exports_the_autopilot_leader_and_how_to_switch_it_on(mocker) -> None:
    environment = _autopilot_environment("http://127.0.0.1:50488")
    environment_service = SimpleNamespace(
        get_environment_by_id=mocker.AsyncMock(return_value=environment), session=None
    )
    build = mocker.patch("api.environments.build_runtime_config", side_effect=build_runtime_config)

    response = await get_runtime_config(
        request=SimpleNamespace(),
        project_id=uuid4(),
        environment_id=environment.id,
        environment_service=environment_service,
        robot_client_factory=RobotClientFactory(robot_manager=mocker.MagicMock()),
        simulations=SimulationManager(RobotCatalogRegistry()),
    )

    leader = build.call_args.kwargs["leader"]
    assert (leader.name, leader.payload.http_port) == ("Sim arm autopilot", 50488)
    body = response.body.decode()
    assert body.startswith("# The leader is the simulation's autopilot.")
    assert "curl -X POST http://127.0.0.1:50488/autopilot" in body
    document = yaml.safe_load(body)
    leader_driver = document["init_args"]["action_source"]["init_args"]["leader"]["init_args"]["robot"]
    assert leader_driver["init_args"]["http_port"] == 50488


@pytest.mark.parametrize("http_url", ["http://127.0.0.1:50488\n", "http://127.0.0.1:50488/$(touch pwned)"])
async def test_the_autopilot_instructions_take_only_the_port_from_the_stored_address(mocker, http_url: str) -> None:
    body = await _export(mocker, _autopilot_environment(http_url))

    # The stored address is free text: none of it but the port reaches the comment or the command.
    assert "#   curl -X POST http://127.0.0.1:50488/autopilot -H" in body
    assert "$(" not in body
    yaml.safe_load(body)
