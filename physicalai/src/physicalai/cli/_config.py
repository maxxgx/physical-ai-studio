from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML in config file {path}: {exc}"
        raise ValueError(msg) from exc
    except OSError as exc:
        msg = f"Failed to read config file {path}: {exc}"
        raise OSError(msg) from exc

    if raw is None:
        msg = f"Config file is empty: {path}"
        raise ValueError(msg)
    if not isinstance(raw, Mapping):
        msg = f"Config file must contain a mapping at top level: {path}"
        raise ValueError(msg)

    try:
        return RuntimeConfig(**raw)
    except ValidationError as exc:
        logger.exception("Invalid runtime config in %s", path)
        msg = f"Invalid runtime config in {path}: {exc}"
        raise ValueError(msg) from exc
