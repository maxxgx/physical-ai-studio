from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_config(path: str | Path):
    from physicalai.cli._config import load_config

    return load_config(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physicalai")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a policy from YAML config")
    run_parser.add_argument("--config", required=True, help="Path to YAML config file")
    run_parser.add_argument("--duration-s", type=float, help="Override YAML duration_s")
    run_parser.add_argument("--fps", type=float, help="Override YAML fps")
    run_parser.add_argument("--dry-run", action="store_true", help="Load config, print summary, exit")
    run_parser.set_defaults(func=handle_run)

    return parser


def handle_run(args: argparse.Namespace) -> None:
    try:
        config = _load_config(args.config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except ValueError as exc:
        print(f"error: invalid config: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if args.duration_s is not None:
        config.duration_s = args.duration_s
    if args.fps is not None:
        config.fps = args.fps

    if args.dry_run:
        print("Config loaded successfully")
        print(f"  model:      {config.model.path} (backend={config.model.backend})")
        print(f"  robot:      {config.robot}")
        print(f"  cameras:    {list(config.cameras.keys()) or 'none'}")
        print(f"  execution:  {config.execution.mode}")
        print(f"  fps:        {config.fps}")
        print(f"  duration_s: {config.duration_s}")
        return

    from physicalai.runtime.runtime import PolicyRuntime

    runtime = PolicyRuntime.from_config(config)

    try:
        runtime.robot.connect()
        for cam in runtime.cameras.values():
            cam.connect()
        stats = runtime.run(duration_s=config.duration_s)
        print(f"Run complete: {stats}")
    except ConnectionError as exc:
        print(f"error: connection failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        for cam in runtime.cameras.values():
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001
                pass
        try:
            runtime.robot.disconnect()
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(2)

    try:
        args.func(args)
    except KeyboardInterrupt:
        raise SystemExit(0) from None


if __name__ == "__main__":
    main()
