# Steam Deck Controller → WebSocket Bridge

Reads the Steam Deck's integrated controller via `evdev` and broadcasts
normalized state to browser clients over WebSocket. Includes a built-in
visual test page.

## Prerequisites

Run **on the Steam Deck** (SteamOS / any Linux with the controller):

```bash
pip install physicalai[steamdeck]
# or: pip install evdev aiohttp
```

**Stop Steam** before running — Steam grabs exclusive access to the controller:

```bash
steam -shutdown   # or kill it from the task manager
```

## Quick start

```bash
python examples/steamdeck/server.py
```

Open `http://localhost:8080` in a browser (on the Deck or another machine on
the same network — use the Deck's IP instead of `localhost`).

## Options

| Flag | Default | Description |
|------|---------|------------|
| `--device` | auto-detect | evdev device path, e.g. `/dev/input/event6` |
| `--port` | `8080` | HTTP / WebSocket port |
| `--hz` | `60` | Broadcast rate in Hz |

## Architecture

```
Steam Deck Controller
        │
        ▼
   evdev (kernel)
        │
        ▼
  server.py (Python)
   ├─ reads events, normalizes axes to -1..1
   └─ broadcasts JSON over WebSocket at --hz rate
        │
        ▼
   Browser (index.html)
   └─ connects to ws://host:port/ws
   └─ renders sticks, triggers, trackpads, buttons
```

## Troubleshooting

**"Steam Deck controller not found"** — The auto-detection looks for a
device name containing "steam" and "deck" or "controller". If your device
has a different name, the server prints all available devices. Use
`--device /dev/input/eventN` to specify it manually.

**Permission denied** — You may need to run as root or add your user to the
`input` group: `sudo usermod -aG input $USER` (then log out and back in).
