from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    path: str
    backend: str = "auto"
    device: str = "auto"


class CameraConfig(BaseModel):
    type: str
    device_id: str
    model_config = ConfigDict(extra="allow")


class ExecutionConfig(BaseModel):
    mode: str = "async"
    threshold: float = 0.5
    watchdog_timeout_s: float = 30.0


class SmootherConfig(BaseModel):
    type: str = "lerp"
    duration_frames: int = 5


class RuntimeConfig(BaseModel):
    model: ModelConfig
    robot: dict[str, Any]
    cameras: dict[str, CameraConfig] = {}
    execution: ExecutionConfig = ExecutionConfig()
    smoother: SmootherConfig = SmootherConfig()
    fps: float = 30.0
    duration_s: float | None = None


def load_config(path: str | Path) -> RuntimeConfig:
    """Load and validate a runtime YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file {path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read config file {path}: {exc}") from exc

    if raw is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")

    try:
        return RuntimeConfig(**raw)
    except ValidationError as exc:
        logger.exception("Invalid runtime config in %s", path)
        raise ValueError(f"Invalid runtime config in {path}: {exc}") from exc
