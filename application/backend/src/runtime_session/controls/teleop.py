from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from robots.features import feature_names
from runtime_session.controls.base import Control
from runtime_session.store import ACTION

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physicalai.robot import Robot

    from runtime_session.bus import Bus
    from runtime_session.store import Store
    from schemas.runtime_session import TeleopControlConfig


class TeleopControl(Control):
    """Command the follower whatever the leader measures, due by the leader's next reading.

    The leader can be anywhere when teleop starts, so for the first ``grace_s``
    each command is stamped due at the end of the grace period: the robot worker
    closes the gap over it rather than jumping. It tracks the leader the whole
    time, so the follower is on it when the grace period ends.

    With ``autopilot``, the leader reads a simulation's autopilot, and reports in each
    reading whether the autopilot leads (sensor ``autopilot``). Teleop then stops if the
    autopilot stops leading (its scene changed, or it was switched off in the viewer), or
    does not start leading within ``AUTOPILOT_START_S``: the follower would only hold.
    """

    AUTOPILOT_START_S = 5.0

    def __init__(
        self,
        store: Store,
        leader: Robot,
        config: TeleopControlConfig,
        bus: Bus | None = None,
        *,
        autopilot: bool = False,
    ) -> None:
        leader_joints = tuple(feature_names(list(leader.joint_names)))
        if leader_joints != store.spec.joint_names:
            raise ValueError(
                f"The leader's joints {leader_joints} do not match the follower's {store.spec.joint_names}"
            )
        super().__init__(store, config, name="TeleopControl", hz=config.hz, bus=bus)
        self.leader = leader
        self.grace_s = config.grace_s
        self.started_at = 0.0
        self.autopilot = autopilot
        self.autopilot_led = False

    def on_start(self) -> None:
        self.started_at = time.monotonic()
        # A leader that reads over a connection idle since it connected (the simulation's virtual
        # leader keeps an HTTP connection, which the simulation closes after a few idle seconds)
        # fails its first read, and connects again for the next one.
        try:
            self.leader.get_observation()
        except ConnectionError as error:
            logger.info("The leader's first reading failed ({}); reading it again", error)
            self.leader.get_observation()

    def on_tick(self) -> None:
        observation = self.leader.get_observation()
        if self.autopilot:
            self._check_autopilot(observation.sensor_data)
        due = max(observation.timestamp + 1 / self.hz, self.started_at + self.grace_s)
        self.store.write(ACTION, observation.joint_positions, timestamp=due)

    def _check_autopilot(self, sensor_data: Mapping[str, Any] | None) -> None:
        """Raise once the autopilot stopped leading, or did not start in time; nothing if the leader does not say."""
        leads = (sensor_data or {}).get("autopilot")
        if leads is None:
            return
        if bool(np.asarray(leads).any()):
            self.autopilot_led = True
        elif self.autopilot_led:
            raise RuntimeError("The simulation's autopilot stopped leading (its scene changed, or it was switched off)")
        elif time.monotonic() - self.started_at > self.AUTOPILOT_START_S:
            raise RuntimeError("The simulation's autopilot did not start leading")
