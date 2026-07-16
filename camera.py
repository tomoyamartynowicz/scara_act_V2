from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pyrealsense2 as rs


class CameraError(RuntimeError):
    pass


@dataclass
class CameraSample:
    frames: dict[str, np.ndarray]
    time: float
    rs_timestamp_ms: float
    frame_number: int


class RealSenseSource:
    def __init__(
        self,
        color_name: str = "wrist_d405",
        depth_name: str | None = None,
        size: tuple[int, int] = (640, 480),
        fps: int = 30,
        serial: str | None = None,
        warmup_frames: int = 30,
        timeout_ms: int = 1000,
        align_depth: bool = False,
    ) -> None:
        self.color_name = color_name
        self.depth_name = depth_name
        self.size = size
        self.fps = fps
        self.serial = serial
        self.warmup_frames = warmup_frames
        self.timeout_ms = timeout_ms
        self.align_depth = align_depth

        self.pipeline: rs.pipeline | None = None
        self.align: rs.align | None = None

    def start(self) -> "RealSenseSource":
        self.pipeline = rs.pipeline()
        config = rs.config()

        if self.serial:
            config.enable_device(self.serial)

        width, height = self.size

        config.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.rgb8,
            self.fps,
        )

        if self.depth_name:
            config.enable_stream(
                rs.stream.depth,
                width,
                height,
                rs.format.z16,
                self.fps,
            )

        try:
            self.pipeline.start(config)
        except Exception as exc:
            raise CameraError(
                f"Could not start RealSense camera: {exc}"
            ) from exc

        if self.depth_name and self.align_depth:
            self.align = rs.align(rs.stream.color)

        for _ in range(self.warmup_frames):
            self._wait_for_frames()

        return self

    def _wait_for_frames(self):
        if self.pipeline is None:
            raise CameraError("Camera has not been started")

        try:
            frames = self.pipeline.wait_for_frames(self.timeout_ms)
        except Exception as exc:
            raise CameraError(
                "Timed out waiting for RealSense frame"
            ) from exc

        if self.align is not None:
            frames = self.align.process(frames)

        return frames

    def read(self) -> CameraSample:
        # Gebruik dezelfde hostklok als de qpos-poller.
        wait_start = time.monotonic()
        frameset = self._wait_for_frames()
        wait_end = time.monotonic()

        color_frame = frameset.get_color_frame()
        if not color_frame:
            raise CameraError("Missing RGB frame")

        frames = {
            self.color_name: np.asanyarray(
                color_frame.get_data()
            ).copy()
        }

        if self.depth_name:
            depth_frame = frameset.get_depth_frame()

            if not depth_frame:
                raise CameraError("Missing depth frame")

            # Ruwe uint16-depth bewaren.
            frames[self.depth_name] = np.asanyarray(
                depth_frame.get_data()
            ).copy()

        return CameraSample(
            frames=frames,
            # Benadering van het moment waarop het frame beschikbaar kwam.
            time=0.5 * (wait_start + wait_end),
            rs_timestamp_ms=float(color_frame.get_timestamp()),
            frame_number=int(color_frame.get_frame_number()),
        )

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    def __enter__(self) -> "RealSenseSource":
        return self.start()

    def __exit__(self, *_args) -> None:
        self.close()
        