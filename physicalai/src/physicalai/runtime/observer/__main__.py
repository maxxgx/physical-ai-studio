from __future__ import annotations

import argparse


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
    except ImportError:
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


if __name__ == "__main__":
    main()
