"""Map physicalai.robot SharedRobot errors to application BaseException types."""

from __future__ import annotations

from loguru import logger
from physicalai.robot import (
    RobotDeviceAlreadyOwned,
    RobotError,
    RobotNameConflict,
    RobotProtocolMismatch,
    RobotTransportError,
)

from exceptions import BaseException as AppBaseException
from exceptions import (
    RobotDeviceAlreadyOwnedError,
    RobotNameConflictError,
    RobotProtocolMismatchError,
    SharedRobotTransportError,
    SimulationNotRunningError,
)

# Prefix of the error a simulated robot raises on connect when no simulation runs under its name.
# Its message is written for users (it says how to start the simulation), so it is forwarded as is.
SIMULATION_NOT_RUNNING_PREFIX = "No MuJoCo simulation named "


def translate_robot_error(  # noqa: PLR0911 - one return per mapped error.
    exc: BaseException,
    *,
    robot_name: str | None = None,
) -> BaseException:
    """Map physicalai.robot errors to app BaseException at the integration boundary.

    User-facing copy lives on the app exceptions. Only structured fields are used:
    ``RobotDeviceAlreadyOwned.device_ids`` from physicalai, and ``robot_name`` from
    the Studio connect context. Library messages are never parsed or forwarded, except
    a simulated robot's "not running" error, which is written for users.
    """
    if isinstance(exc, AppBaseException):
        return exc

    if isinstance(exc, RobotDeviceAlreadyOwned):
        device_ids = tuple(exc.device_ids or ())
        return RobotDeviceAlreadyOwnedError(device_ids=device_ids or None)

    if isinstance(exc, RobotNameConflict):
        return RobotNameConflictError(robot_name=robot_name)

    if isinstance(exc, RobotProtocolMismatch):
        # physicalai.RobotProtocolMismatch does not expose supported/remote
        # protocol versions as structured fields — only in the message string.
        return RobotProtocolMismatchError()

    if type(exc) is RobotTransportError and str(exc).startswith(SIMULATION_NOT_RUNNING_PREFIX):
        return SimulationNotRunningError(str(exc), robot_name=robot_name)

    if isinstance(exc, RobotTransportError):
        logger.warning("Unmapped SharedRobot transport failure (phase={!r}): {}", exc.phase, exc)
    elif isinstance(exc, RobotError):
        logger.warning("Unmapped robot error: {}", exc)
    else:
        return exc

    return SharedRobotTransportError()
