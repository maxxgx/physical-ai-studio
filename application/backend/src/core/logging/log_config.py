# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from settings import get_settings

settings = get_settings()
WORKERS_FOLDER = os.path.join(settings.log_dir, "workers")
JOBS_FOLDER = os.path.join(settings.log_dir, "jobs")


@dataclass
class LogConfig:
    """Configuration for logging behavior."""

    rotation: str = "10 MB"
    retention: str = "10 days"
    # Worker processes write one file each (workers/<Class>-<pid>-<start>.log). The backend removes
    # the oldest at startup: files older than this many days, then all but the newest ones.
    worker_files_max_age_days: int = 10
    worker_files_kept: int = 100
    # Rotated files a running worker keeps besides its current one; it removes older ones as it rotates.
    worker_rotated_files_kept: int = 10
    level: str = "DEBUG" if settings.debug else "INFO"
    serialize: bool = True
    log_folder: Path = settings.log_dir
    # Mapping of worker classes to their dedicated log files.
    # None key is used for application-level logs that don't belong to any specific worker.
    # A process worker of a class not listed here writes its own file instead (see setup_logging).
    worker_log_info: ClassVar[dict[str | None, str]] = {
        "TrainingWorker": "training.log",
        "DatasetImportWorker": "dataset_import.log",
        None: "app.log",
    }
