from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m physicalai.runtime.observer",
        description="Observe runtime telemetry from a running PolicyRuntime session",
    )
    parser.add_argument("--session-id", default=None, help="Filter to a specific session ID")
    parser.add_argument("--record", default=None, metavar="PATH", help="Record events to JSONL file")
    parser.add_argument("--no-console", action="store_true", help="Disable live console output")
    args = parser.parse_args(argv)

    try:
        from physicalai.runtime.observer._subscriber import TelemetrySubscriber
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("Install telemetry dependencies: pip install physicalai[telemetry]", file=sys.stderr)
        raise SystemExit(1) from None

    subscriber = TelemetrySubscriber(session_id=args.session_id)

    if not args.no_console:
        from physicalai.runtime.observer._console import ConsoleHandler

        subscriber.add_handler(ConsoleHandler())

    recorder = None
    if args.record:
        from pathlib import Path

        from physicalai.runtime.observer._recorder import RecorderHandler

        recorder = RecorderHandler(Path(args.record))
        subscriber.add_handler(recorder)

    subscriber.start()
    print(f"Observing telemetry (session={args.session_id or 'all'})... Press Ctrl+C to stop.")

    try:
        import signal

        signal.pause()
    except AttributeError:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.stop()
        if recorder:
            recorder.close()
        print("\nObserver stopped.")


if __name__ == "__main__":
    main()
