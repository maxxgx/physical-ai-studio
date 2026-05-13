# Runtime System — Implementation Plan

This document is the implementation plan for `physicalai.runtime`. It refines the original [policy_runtime_design.md](./policy_runtime_design.md) based on codebase exploration, bug analysis, and architecture review.

Read [policy_runtime_design.md](./policy_runtime_design.md) first for API shape and ownership rules. This document covers what to build, in what order, and why.

## Reference Implementation

The golden reference is `physicalai/examples/runtime/inference_async.py` — a working async prototype with QueueMixer, InferenceThread, velocity clamping, and camera discovery. Every runtime component must match or exceed its behavior.

---

## Phase 1: Critical Bug Fixes (half day)

Fix bugs on the code path Phase 2 depends on. Phase 2 defines a public runtime contract — workarounds would calcify into permanent API shape. `predict_action_chunk()` currently raises `RuntimeError` without these fixes because Bug 2's inverted guard blocks the runtime's call to `model.predict_action_chunk(obs)`. This is not stylistic — it is a hard blocker.

### Bug 1: `use_action_queue` checks manifest, ignores runtime runner

**File**: `physicalai/src/physicalai/inference/model.py` — `use_action_queue` property

**Problem**: Reads `self.manifest.model.runner` class_path. Ignores `self.runner` passed at construction or set at runtime.

**Fix**: Check `isinstance(self.runner, ActionChunking)` instead:

```python
@property
def use_action_queue(self) -> bool:
    from physicalai.inference.runners.action_chunking import ActionChunking
    return isinstance(self.runner, ActionChunking)
```

### Bug 2: `select_action()` / `predict_action_chunk()` guards inverted

**File**: `physicalai/src/physicalai/inference/model.py`

**Problem**: `select_action()` raises when `not use_action_queue`. `predict_action_chunk()` raises when `use_action_queue`. Both are backwards — `select_action()` should be the generic one-action API, `predict_action_chunk()` should return raw chunks.

**Fix**: Remove guards entirely. Both methods should work for any runner (shape-stable contract per design doc §3):

| Runner          | `select_action()`  | `predict_action_chunk()` |
| --------------- | ------------------ | ------------------------ |
| single-pass     | runner output      | wrap as `(1, D)` chunk   |
| chunk-producing | pop one via cursor | runner output            |

### Bug 3: ACT export manifest declares SinglePass instead of ActionChunking

**File**: `library/src/physicalai/export/mixin_policy.py` — `_build_manifest()`

**Problem**: Checks `metadata.get("use_action_queue", False)` but ACT never sets this metadata flag. Manifest always gets `SinglePass` runner.

**Fix**: ACT export must pass `use_action_queue=True` and `chunk_size=<config.chunk_size>` in its metadata kwargs. Same fix needed for Pi0.5 (Bug 5).

### Deferred Bugs (document as issues, do not block Phase 2)

| Bug   | Summary                                                               | Why deferred                                            |
| ----- | --------------------------------------------------------------------- | ------------------------------------------------------- |
| Bug 4 | Manifest missing `hardware` section (`RobotSpec`, `CameraSpec`)       | Not on inference code path                              |
| Bug 5 | Pi0.5 export also declares SinglePass                                 | Same root cause as Bug 3, fix together                  |
| Bug 7 | Pi0.5 normalization not baked into graph (external pre/postprocessor) | By design — manifest `preprocessors_specs` handles this |
| Bug 8 | Pi0.5 denoising loop not exportable cleanly (11x graph size)          | Export concern, not runtime concern                     |
| Bug 9 | `OVTokenizer` may need `import openvino_tokenizers` for custom ops    | Needs verification — may already work via adapter       |

---

## Phase 2: Runtime System (2–3 days)

New package: `physicalai/src/physicalai/runtime/`

```text
physicalai/src/physicalai/runtime/
├── __init__.py              # exports: PolicyRuntime, SyncExecution, AsyncExecution,
│                            #          ActionQueue, LerpSmoother, ReplaceSmoother, RunStats
├── smoothers.py             # ChunkSmoother ABC, ReplaceSmoother, LerpSmoother
├── _action_queue.py         # ActionQueue (public via __init__, internal module)
├── execution.py             # Execution ABC, SyncExecution, AsyncExecution, WorkerDiedError
└── runtime.py               # PolicyRuntime, RunStats, default_observation_to_input
```

Dependency order: `smoothers.py` → `_action_queue.py` → `execution.py` → `runtime.py` → `__init__.py`

### Architectural Decisions

**ActionQueue is owned by PolicyRuntime, not hidden inside Execution.**

The original design doc keeps Execution (scheduling) and ActionQueue (buffering) as separate concerns. This is correct — when `AsyncExecution(transport="process")` or `RemoteExecution` arrive, they should push chunks into the same ActionQueue without duplicating buffer logic.

Users get a clean default API:

```python
runtime = PolicyRuntime(
    robot=robot,
    model=model,
    execution=AsyncExecution(threshold=0.5),
    fps=30,
)
runtime.run(duration_s=60)
```

Power users can override buffering:

```python
runtime = PolicyRuntime(
    robot=robot,
    model=model,
    execution=AsyncExecution(threshold=0.5),
    action_queue=ActionQueue(smoother=LerpSmoother(duration_frames=10)),
    fps=30,
)
```

**Execution is a scheduler, not a buffer.** It decides when/where inference runs and pushes results into ActionQueue. It does not own pop, remaining, or chunk_size.

**InferenceModel must NOT import ActionQueue.** Per design doc §4: if both layers need pop-from-chunk mechanics, they share `ActionChunkCursor`, not `ActionQueue`.

### 2.1 `smoothers.py`

Extracted from `QueueMixer.add()` in inference_async.py.

```python
class ChunkSmoother(ABC):
    """Merges a new action chunk into remaining actions from the previous chunk."""

    @abstractmethod
    def merge(
        self,
        remaining: np.ndarray,    # (R, action_dim) — unconsumed actions from previous chunk
        incoming: np.ndarray,     # (H, action_dim) — new chunk from inference
        offset: int,              # skip first N actions of incoming (latency compensation)
    ) -> np.ndarray:
        """Return merged actions array. Called by ActionQueue.push_chunk()."""
        ...


class ReplaceSmoother(ChunkSmoother):
    """Drop remaining actions, use incoming[offset:]."""

    def merge(self, remaining, incoming, offset):
        return incoming[offset:]


class LerpSmoother(ChunkSmoother):
    """Lerp-blend overlapping region, then append non-overlapping tail.

    Stateless merge — no hidden mutation. duration_frames is the fallback
    blending window used when offset is 0. When offset > 0, the blending
    window is computed from offset directly: lerp_dur = max(offset, 1).

    Matches QueueMixer.add() from inference_async.py:
    - Weights: w_i = max(1.0 - i / lerp_dur, 0.0) for old actions
    - Overlap region: blended = w * remaining + (1 - w) * incoming
    """

    def __init__(self, duration_frames: int = 5) -> None:
        self.duration_frames = duration_frames

    def merge(self, remaining, incoming, offset):
        lerp_dur = max(offset, 1) if offset > 0 else self.duration_frames

        incoming = incoming[offset:]
        n_remain = len(remaining)
        lerp_dur = min(n_remain, lerp_dur)

        weights = np.maximum(1.0 - np.arange(n_remain) / max(lerp_dur, 1), 0.0)
        weights = weights[:, np.newaxis]

        n_blend = min(n_remain, len(incoming))
        blended = weights[:n_blend] * remaining[:n_blend] + (1.0 - weights[:n_blend]) * incoming[:n_blend]

        return np.concatenate([blended, incoming[n_blend:]], axis=0).astype(np.float32)
```

Key: `offset` is not just "skip N actions." It is latency compensation — `offset = int(inference_latency * fps)`. The smoother must handle this, not the caller.

### 2.2 `_action_queue.py` (public API, internal module)

Thread-safe action buffer with smoother integration and hold telemetry.

```python
class ActionQueue:
    """Thread-safe action buffer with chunk smoothing and starvation telemetry.

    Public API — exported from physicalai.runtime. Power users can override
    the default ActionQueue on PolicyRuntime to customize smoothing behavior.
    Execution pushes chunks into it; PolicyRuntime pops actions from it.
    """

    def __init__(self, smoother: ChunkSmoother | None = None) -> None:
        self._smoother = smoother or ReplaceSmoother()
        self._lock = threading.Lock()
        self._queue: np.ndarray | None = None
        self._index: int = 0

        # Telemetry
        self._consecutive_holds: int = 0
        self._total_holds: int = 0
        self._total_pops: int = 0

    def push_chunk(self, chunk: np.ndarray, offset: int = 0) -> None:
        """Merge a new chunk into the queue. Thread-safe."""
        with self._lock:
            if self._queue is None or self._index >= len(self._queue):
                self._queue = chunk[offset:]
                self._index = 0
                return
            remaining = self._queue[self._index:]
            self._queue = self._smoother.merge(remaining, chunk, offset)
            self._index = 0

    def pop(self) -> np.ndarray | None:
        """Pop next action, or None if empty. Thread-safe."""
        with self._lock:
            if self._queue is None or self._index >= len(self._queue):
                self._consecutive_holds += 1
                self._total_holds += 1
                return None
            action = self._queue[self._index]
            self._index += 1
            self._consecutive_holds = 0
            self._total_pops += 1
            return action

    @property
    def remaining(self) -> int:
        with self._lock:
            if self._queue is None:
                return 0
            return max(len(self._queue) - self._index, 0)

    def below_threshold(self, threshold: int) -> bool:
        return self.remaining <= threshold

    def clear(self) -> None:
        with self._lock:
            self._queue = None
            self._index = 0

    # --- Telemetry ---
    @property
    def consecutive_holds(self) -> int:
        return self._consecutive_holds

    @property
    def total_holds(self) -> int:
        return self._total_holds

    @property
    def total_pops(self) -> int:
        return self._total_pops
```

### 2.3 `execution.py`

**Execution ABC** — scheduler only. Pushes chunks into ActionQueue, does not own pop/remaining.

```python
class Execution(ABC):
    """Decides when and where inference runs. Pushes results into ActionQueue."""

    @abstractmethod
    def start(self, model: InferenceModel, action_queue: ActionQueue) -> None:
        """Bind to model and queue. Called once before the loop."""
        ...

    @abstractmethod
    def maybe_request(self, observation: dict[str, np.ndarray]) -> None:
        """Check if new inference is needed. If so, run or schedule it."""
        ...

    @abstractmethod
    def warmup(self, sample_observation: dict[str, np.ndarray]) -> None:
        """Run one inference to discover chunk_size and seed the queue.

        After warmup():
        - action_queue has one chunk ready (robot starts moving immediately)
        - self.chunk_size is set
        - self.action_dim is set
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop scheduling. For async: signal thread, join with timeout."""
        ...

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Discovered after warmup(). Used to compute threshold."""
        ...
```

**SyncExecution** — blocks on inference when queue runs low.

```python
class SyncExecution(Execution):
    """Synchronous inference in the control thread."""

    def __init__(self) -> None:
        self._model: InferenceModel | None = None
        self._queue: ActionQueue | None = None
        self._chunk_size: int = 0

    def start(self, model, action_queue):
        self._model = model
        self._queue = action_queue

    def warmup(self, sample_observation):
        actions = self._model.predict_action_chunk(sample_observation)
        self._chunk_size = actions.shape[0]
        self._queue.push_chunk(actions, offset=0)

    def maybe_request(self, observation):
        if self._queue.below_threshold(1):  # refill when empty
            actions = self._model.predict_action_chunk(observation)
            self._queue.push_chunk(actions, offset=0)

    def stop(self):
        pass

    @property
    def chunk_size(self):
        return self._chunk_size
```

**AsyncExecution** — background thread with health monitoring. Maps to `InferenceThread` from inference_async.py.

```python
class WorkerDiedError(RuntimeError):
    """Raised when the inference worker thread dies unexpectedly."""
    pass


class AsyncExecution(Execution):
    """Async inference in a background thread.

    Thread architecture (matches inference_async.py):

        Control thread (main):              Inference thread (background):
        ─────────────────────               ────────────────────────────
        loop at fps:                        loop:
          obs = robot.get_observation()       wait for obs_slot
          execution.maybe_request(obs)        chunk = model.predict_action_chunk(obs)
          action = queue.pop()                offset = int(latency * fps)
          robot.send_action(action)           queue.push_chunk(chunk, offset)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        fps: int = 30,
        watchdog_timeout_s: float = 30.0,
        max_consecutive_holds: int | None = None,   # default: 3 * fps (3 seconds)
    ) -> None:
        self._threshold_frac = threshold
        self._fps = fps
        self._watchdog_timeout_s = watchdog_timeout_s
        self._max_consecutive_holds = max_consecutive_holds or 3 * fps

        # Set during start()
        self._model: InferenceModel | None = None
        self._queue: ActionQueue | None = None
        self._chunk_size: int = 0
        self._threshold_count: int = 0

        # Thread state
        self._lock = threading.Lock()
        self._obs_slot: dict | None = None
        self._obs_ready = threading.Event()
        self._running_inference = False
        self._request_time: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._death_cause: BaseException | None = None
        self._inference_count: int = 0

    def start(self, model, action_queue):
        self._model = model
        self._queue = action_queue
        self._thread = threading.Thread(target=self._run, name="InferenceThread", daemon=True)
        self._thread.start()

    def warmup(self, sample_observation):
        actions = self._model.predict_action_chunk(sample_observation)
        self._chunk_size = actions.shape[0]
        self._threshold_count = int(self._chunk_size * self._threshold_frac)
        self._queue.push_chunk(actions, offset=0)

    def maybe_request(self, observation):
        # Check for worker death — raise, don't silently continue
        if self._thread is not None and not self._thread.is_alive():
            if self._death_cause is not None:
                raise WorkerDiedError(
                    f"Inference thread died: {self._death_cause}"
                ) from self._death_cause

        # Check for stuck inference
        if self._busy_duration > self._watchdog_timeout_s:
            logger.warning(
                "Inference stuck for %.0fs — force resetting", self._busy_duration,
            )
            self._force_reset()

        # Submit if queue is low and worker is idle
        if self._queue.below_threshold(self._threshold_count):
            if not self._busy:
                # Defensive copy — observation may be reused by caller
                snapshot = {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in observation.items()
                }
                with self._lock:
                    self._obs_slot = snapshot
                    self._request_time = time.perf_counter()
                self._obs_ready.set()

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            self._obs_ready.set()   # unblock wait
            self._thread.join(timeout=10.0)

    @property
    def chunk_size(self):
        return self._chunk_size

    # --- Health properties ---

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def _busy(self) -> bool:
        with self._lock:
            return self._obs_slot is not None or self._running_inference

    @property
    def _busy_duration(self) -> float:
        with self._lock:
            if not (self._obs_slot is not None or self._running_inference):
                return 0.0
            return time.perf_counter() - self._request_time

    @property
    def inference_count(self) -> int:
        return self._inference_count

    # --- Internal ---

    def _force_reset(self) -> None:
        with self._lock:
            self._obs_slot = None
            self._running_inference = False
        logger.warning("Force reset — cleared stuck inference state")

    def _run(self) -> None:
        """Inference thread main loop."""
        try:
            while not self._stop_event.is_set():
                self._obs_ready.wait()
                self._obs_ready.clear()

                if self._stop_event.is_set():
                    return

                with self._lock:
                    obs = self._obs_slot
                    self._obs_slot = None
                    if obs is None:
                        continue
                    self._running_inference = True

                t0 = time.perf_counter()
                actions = self._model.predict_action_chunk(obs)
                latency = time.perf_counter() - t0

                offset = int(latency * self._fps)

                self._queue.push_chunk(actions, offset=offset)
                self._inference_count += 1

                with self._lock:
                    self._running_inference = False

        except Exception as e:
            self._death_cause = e
            logger.error("Inference thread died: %s", e, exc_info=True)
```

### 2.4 `runtime.py`

```python
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from physicalai.capture.camera import Camera, Frame
from physicalai.inference import InferenceModel
from physicalai.robot.interface import Robot, RobotObservation
from physicalai.runtime._action_queue import ActionQueue
from physicalai.runtime.execution import Execution, WorkerDiedError
from physicalai.runtime.smoothers import LerpSmoother

logger = logging.getLogger(__name__)

def default_observation_to_input(
    robot_obs: RobotObservation,
    camera_frames: dict[str, Frame],
) -> dict[str, Any]:
    """Default observation-to-model-input conversion.

    Maps:
    - Joint positions → "state" array
    - Camera frames → "images.{name}" arrays

    For Pi0.5 or other models needing custom keys (e.g. "task"),
    pass a custom obs_to_input callable to PolicyRuntime.
    """
    model_input: dict[str, Any] = {}

    # Collect joint positions into "state" vector
    if robot_obs.joint_positions:
        model_input["state"] = np.array([robot_obs.joint_positions], dtype=np.float32)

    # Map camera frames to "images.{name}"
    for name, frame in camera_frames.items():
        model_input[f"images.{name}"] = frame.data

    return model_input


class RuntimeCallback(Protocol):
    """Optional hook points in the PolicyRuntime control loop."""

    def before_send_action(self, *, action: np.ndarray, step: int) -> np.ndarray | None:
        """Called before sending action. Return modified action or None to use original."""
        ...

    def on_action_sent(self, *, action: np.ndarray, step: int) -> None:
        """Called after action is sent to robot."""
        ...

    def on_hold(self, *, step: int, holds: int) -> None:
        """Called when action queue is empty and robot holds last position."""
        ...


class PolicyRuntime:
    """Runs a policy on robot hardware.

    Loop shape (matches inference_async.py):
        obs = robot.get_observation()
        model_input = obs_to_input(obs, cameras)
        execution.maybe_request(model_input)
        action = action_queue.pop()
        if action is None: hold position
        robot.send_action(action)
        sleep_until_next_tick()
    """

    def __init__(
        self,
        robot: Robot,
        model: InferenceModel,
        execution: Execution,
        fps: float,
        cameras: Mapping[str, Camera] | None = None,
        action_queue: ActionQueue | None = None,
        obs_to_input: Callable[[RobotObservation, dict[str, Frame]], dict[str, Any]] | None = None,
        callbacks: Sequence[RuntimeCallback] = (),
    ) -> None:
        self._robot = robot
        self._model = model
        self._execution = execution
        self._fps = fps
        self._cameras = cameras or {}
        self._action_queue = action_queue or ActionQueue(smoother=LerpSmoother(duration_frames=5))
        self._obs_to_input = obs_to_input or default_observation_to_input
        self._callbacks = list(callbacks)

    def run(self, *, duration_s: float | None = None) -> RunStats:
        """Run the control loop.

        1. Warm up — run one inference, seed queue, discover chunk_size
        2. Loop — observe, maybe_request, pop, send, sleep
        3. Shutdown — stop execution, drain
        """
        # --- Init ---
        self._execution.start(self._model, self._action_queue)

        sample_obs = self._build_model_input()
        self._execution.warmup(sample_obs)

        goal_time = 1.0 / self._fps
        step = 0
        last_action: np.ndarray | None = None

        try:
            while True:
                if duration_s is not None and step * goal_time >= duration_s:
                    break

                loop_start = time.perf_counter()

                # 1. Observe
                obs = self._build_model_input()

                # 2. Maybe request inference
                self._execution.maybe_request(obs)

                # 3. Pop action
                action = self._action_queue.pop()
                if action is not None:
                    last_action = action
                else:
                    action = last_action
                    holds = self._action_queue.consecutive_holds
                    if holds == 1:
                        logger.warning("Queue empty — holding position")
                    elif holds % self._fps == 0:
                        logger.warning(
                            "Queue starvation: %d consecutive holds (%.1fs)",
                            holds, holds / self._fps,
                        )
                    self._invoke_callback("on_hold", step=step, holds=holds)

                if action is None:
                    # No warmup result and no previous action — skip
                    logger.error("No action available (warmup may have failed)")
                    continue

                # 4. Callbacks
                action = self._invoke_callback("before_send_action", action=action, step=step) or action

                # 5. Send
                self._robot.send_action(action)
                self._invoke_callback("on_action_sent", action=action, step=step)

                # 6. Timing
                elapsed = time.perf_counter() - loop_start
                sleep_time = goal_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                step += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except WorkerDiedError as e:
            logger.error("Worker died during runtime: %s", e)
            raise
        finally:
            self._shutdown(step)

        return RunStats(
            steps=step,
            total_pops=self._action_queue.total_pops,
            total_holds=self._action_queue.total_holds,
            inference_count=getattr(self._execution, "inference_count", 0),
        )

    def _build_model_input(self) -> dict[str, Any]:
        robot_obs = self._robot.get_observation()
        camera_frames = {name: cam.read_latest() for name, cam in self._cameras.items()}
        return self._obs_to_input(robot_obs, camera_frames)

    def _shutdown(self, step: int) -> None:
        """Robot and cameras must be connected before run(). Caller owns lifecycle."""
        # 1. Stop inference scheduling
        self._execution.stop()

        # 2. Drain remaining actions (up to 1s) for smooth stop
        remaining = self._action_queue.remaining
        drain_limit = min(remaining, int(self._fps))
        for _ in range(drain_limit):
            action = self._action_queue.pop()
            if action is not None:
                self._robot.send_action(action)
                time.sleep(1.0 / self._fps)

        logger.info(
            "Shutdown complete — %d steps, %d pops, %d holds",
            step, self._action_queue.total_pops, self._action_queue.total_holds,
        )

    def _invoke_callback(self, method: str, **kwargs):
        result = None
        for cb in self._callbacks:
            fn = getattr(cb, method, None)
            if fn is not None:
                result = fn(**kwargs)
        return result
```

### 2.5 Tests

```text
physicalai/tests/unit/runtime/
├── test_smoothers.py        # ReplaceSmoother, LerpSmoother: offset handling, lerp weights,
│                            #   dynamic duration, edge cases (empty remaining, offset > chunk)
├── test_action_queue.py     # push/pop, smoother integration, thread safety (concurrent
│                            #   push+pop from 2 threads), hold counters, clear()
├── test_execution.py        # SyncExecution: warmup seeds queue, maybe_request refills
│                            # AsyncExecution: mock model, health monitoring (alive/busy/
│                            #   busy_duration), WorkerDiedError propagation, force_reset
└── test_runtime.py          # PolicyRuntime with mock robot + mock model:
                             #   full loop, hold fallback, shutdown drain, callbacks,
                             #   WorkerDiedError propagation, duration_s limit
```

All tests use mock `InferenceModel` and mock `Robot` — no hardware, no exported models.

---

## Phase 3: CLI and Integration (1–2 days)

1. `physicalai run --config so101_pi05.yaml` CLI command
2. YAML config loader (`PolicyRuntime.from_config()`)
3. Observation builder (bridges `Robot` protocol + `Camera` → model input dict)
4. Migrate `inference_async.py` to use `PolicyRuntime` (becomes ~20 lines)

### Velocity clamping and camera discovery stay outside core

Velocity clamping (`max_speed`, `ramp_steps`, `commanded_pos` tracking) is SO-101-specific. Camera discovery (interactive selection, name mapping, blank cameras, flip) is app-specific.

These belong in:

- Example scripts (`examples/runtime/`)
- CLI helpers (`physicalai.cli.run`)
- User callbacks

Not in `PolicyRuntime` or `Execution`. If a reusable pattern emerges across 2+ robots, promote to a formal action-transform layer later.

---

## Phase 4: Advanced (later)

1. `AsyncExecution(transport="process")` for PyTorch CPU (GIL contention)
2. RTC guidance correction (requires split export from `library/`)
3. `RemoteExecution` + gRPC `PolicyServer` (see [policy_server_design.md](./policy_server_design.md))

---

## Component Mapping: inference_async.py → Library

| Script component                            | Library target                                             | Notes                                                   |
| ------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| `QueueMixer`                                | `ActionQueue` + `LerpSmoother`                             | Offset-aware blending extracted into smoother           |
| `QueueMixer.lerp_duration = max(offset, 1)` | `LerpSmoother.merge()` computes from offset                | Stateless — `duration_frames` is fallback when offset=0 |
| `InferenceThread`                           | `AsyncExecution`                                           | Same thread architecture: obs_slot + result push        |
| `InferenceThread.force_reset()`             | `AsyncExecution._force_reset()`                            | Clears stuck state                                      |
| `InferenceThread.busy_duration`             | `AsyncExecution._busy_duration`                            | Watchdog timeout trigger                                |
| `InferenceThread.alive`                     | `AsyncExecution.alive`                                     | Dead thread detection                                   |
| `get_full_observation()`                    | `default_observation_to_input()` + `obs_to_input` callable | Separates robot obs format from model input format      |
| `action_to_robot_dict()`                    | `Robot.send_action(ndarray)`                               | Robot protocol handles conversion                       |
| `main()` while-loop                         | `PolicyRuntime.run()`                                      | Same 5-step structure                                   |
| Velocity clamping + ramp                    | User callback or example code                              | Too robot-specific for core runtime                     |
| Camera discovery + `SharedCamera`           | User code / CLI (Phase 3)                                  | App-specific                                            |
| Warm-up inference + queue seeding           | `Execution.warmup()`                                       | Seeds queue so loop starts with actions                 |
| `inference_thread.alive` check + restart    | `AsyncExecution.maybe_request()` raises `WorkerDiedError`  | Raise instead of silent restart                         |
| `force_reset()` for stuck thread            | `AsyncExecution._force_reset()` via watchdog               | Auto-triggered when `busy_duration > timeout`           |
| Hold position + `hold_count`                | `ActionQueue.consecutive_holds` + `PolicyRuntime` logging  | Telemetry via queue counters                            |
| Two-backend support (torch/OV)              | `InferenceModel` abstraction                               | Runtime only calls `predict_action_chunk()`             |

---

## Design Gaps Addressed

Gaps identified during architecture review, with resolutions.

### Gap 1: Observation ownership in async

**Problem**: Main thread passes observation dict to `maybe_request()`. If main thread reuses camera buffers, inference thread reads corrupted data.

**Resolution**: `AsyncExecution.maybe_request()` performs defensive copy before submitting:

```python
snapshot = {
    k: v.copy() if isinstance(v, np.ndarray) else v
    for k, v in observation.items()
}
```

Cost: one dict of numpy copies per inference request (not per tick — only when threshold triggers). At 30fps with threshold=0.5 and chunk_size=50, that's roughly once every ~0.8s.

### Gap 2: Empty-queue telemetry

**Problem**: "Hold last action" silently masks queue starvation.

**Resolution**: `ActionQueue` tracks `consecutive_holds` and `total_holds`. `PolicyRuntime` logs warnings on first hold and every `fps` consecutive holds (1 per second). `on_hold` callback exposes starvation events to user code.

### Gap 3: Graceful shutdown

**Problem**: Hard stop can jerk the robot.

**Resolution**: `PolicyRuntime._shutdown()` drains up to 1 second of remaining actions at loop FPS. Beyond 1s, hard stop — the user pressed Ctrl+C for a reason. Robot and cameras stay connected here; caller owns connect/disconnect lifecycle.

### Gap 4: Error propagation

**Problem**: Inference thread exceptions silently swallowed — `pop_action()` returns None forever.

**Resolution**: `AsyncExecution` stores `_death_cause` on thread exception. `maybe_request()` checks `alive` and raises `WorkerDiedError` with original traceback preserved via `raise ... from`. PolicyRuntime catches and re-raises.

### Gap 5: Two-backend support

**Problem**: inference_async.py handles PyTorch and OpenVINO with different preprocessing.

**Resolution**: Not a runtime concern. `InferenceModel` abstracts backend differences. Runtime only calls `model.predict_action_chunk(obs)`. The torch bypass in inference_async.py exists because the script bypasses `InferenceModel` for direct Pi05 policy access — the library runtime won't need this.

---

## Resolved Questions

1. **`PolicyRuntime.run()` returns `RunStats`.** `@dataclass` with 4 ints (steps, total_pops, total_holds, inference_count). Useful for testing and logging.

2. **Warm-up happens inside `run()`.** User doesn't forget, no "did I call warmup?" failure mode.

3. **Dead worker raises `WorkerDiedError`.** Let caller decide recovery strategy. Auto-restart can mask systematic failures.

---

## Relationship to Existing Design Docs

This plan refines [policy_runtime_design.md](./policy_runtime_design.md). Key differences:

| Topic                           | Original design doc                                              | This plan                                                                                                  |
| ------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| ActionQueue visibility          | Public parameter on PolicyRuntime                                | Public parameter with sensible default. Internal `_action_queue.py` module, exported via `__init__`.       |
| Execution ABC                   | `start(action_queue, model)`, `maybe_request(obs, action_queue)` | `start(model, action_queue)`, `maybe_request(obs)` — queue stored internally on start, not passed per call |
| Warmup                          | `warmup(sample_observation, n=2)`                                | `warmup(sample_observation)` — one call, seeds queue                                                       |
| Health monitoring               | Not specified                                                    | First-class: alive, busy, busy_duration, WorkerDiedError, watchdog                                         |
| Smoothing                       | `LerpChunkSmoother`, `ReplaceMerger`                             | `LerpSmoother`, `ReplaceSmoother` — stateless merge, offset-aware                                          |
| Observation bridge              | Not specified                                                    | `obs_to_input` callable with `default_observation_to_input` fallback                                       |
| `predict_action_chunk()` return | `Mapping[str, Any]` with `"actions"` key                         | `np.ndarray` directly (matches actual implementation)                                                      |
| Bug fixes                       | Not applicable                                                   | Phase 1 prerequisite — bugs 1-3 on critical path                                                           |

All ownership rules, boundary constraints, and deferred-until-needed decisions from the original design doc remain in effect.
