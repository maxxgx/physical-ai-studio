from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi.websockets import WebSocketDisconnect
from loguru import logger
from physicalai.capture.errors import CaptureError, CaptureTimeoutError
from turbojpeg import TJPF_RGB, TurboJPEG

from schemas.project_camera import Camera
from utils.camera_factory import build_direct_camera
from workers.transport.worker_transport import WorkerTransport
from workers.transport_worker import TransportWorker, WorkerState, WorkerStatus

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera as CameraABC


tj = TurboJPEG()


class CameraWorker(TransportWorker[Camera]):
    """Orchestrates camera streaming over configurable transport."""

    def __init__(
        self,
        config: Camera,
        transport: WorkerTransport,
    ) -> None:
        super().__init__(transport)
        self.config = config
        cam: CameraABC | None = build_direct_camera(config)
        if cam is None:
            raise RuntimeError(f"Camera {config.id} not found.")
        self.cam = cam

    async def run(self) -> None:
        """Main worker loop."""
        try:
            await self.transport.connect()

            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, self.cam.connect)
            except CaptureError as exc:
                self.state = WorkerState.ERROR
                self.error_message = str(exc)
                logger.error(f"Failed to connect camera {self.config.name}: {exc}")
                await self.transport.send_json(
                    WorkerStatus(state=self.state, config=self.config, message=str(exc)).to_json()
                )
                return

            self.state = WorkerState.RUNNING
            await self.transport.send_json(
                WorkerStatus(
                    state=self.state,
                    config=self.config,
                    message="Camera connected",
                ).to_json()
            )

            await self.run_concurrent(
                asyncio.create_task(self._capture_loop()),
                asyncio.create_task(self._command_loop()),
            )

        except Exception as e:
            self.state = WorkerState.ERROR
            self.error_message = str(e)
            logger.error(f"Worker error: {e}")
            await self.transport.send_json(WorkerStatus(state=self.state, message=str(e)).to_json())
        finally:
            await self.shutdown()

    async def _capture_loop(self) -> None:
        """Continuously capture and send frames."""
        while not self._stop_requested:
            try:
                frame = await self.cam.async_read(timeout=1.0)
            except CaptureTimeoutError:
                continue
            except CaptureError as exc:
                logger.error(f"capture error on {self.config.fingerprint}: {exc}")
                break

            jpeg_bytes = tj.encode(frame.data, pixel_format=TJPF_RGB, quality=80)
            await self.transport.send_bytes(jpeg_bytes)

    async def _command_loop(self) -> None:
        """Handle incoming commands from client."""
        try:
            while not self._stop_requested:
                command = await self.transport.receive_command()
                if command:
                    await self._handle_command(command)
        except (WebSocketDisconnect, RuntimeError):
            self._stop_requested = True
        except asyncio.CancelledError:
            pass

    async def _handle_command(self, command: dict) -> None:
        """Handle a single command."""
        event = command.get("event")

        match event:
            case "ping":
                await self.transport.send_json(WorkerStatus(state=WorkerState.RUNNING, message="pong").to_json())
            case "disconnect":
                logger.info("Client requested disconnect")
                self._stop_requested = True

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info(f"Shutting down camera: {self.config.name}")
        self._stop_requested = True
        await super().shutdown()
        if self.cam is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.cam.disconnect)
            self.cam = None
