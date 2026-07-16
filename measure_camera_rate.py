#!/usr/bin/env python3

import threading
import time

import numpy as np
import pyrealsense2 as rs


SERIAL = "130322273198"
WIDTH = 640
HEIGHT = 480
FPS = 30
DURATION = 10.0


context = rs.context()

device = next(
    device
    for device in context.query_devices()
    if device.get_info(rs.camera_info.serial_number) == SERIAL
)

sensor = next(
    sensor
    for sensor in device.query_sensors()
    if sensor.get_info(rs.camera_info.name) == "Stereo Module"
)

matching_profile = None

for profile in sensor.get_stream_profiles():
    video = profile.as_video_stream_profile()

    if (
        profile.stream_type() == rs.stream.color
        and profile.format() == rs.format.yuyv
        and profile.fps() == FPS
        and video.width() == WIDTH
        and video.height() == HEIGHT
    ):
        matching_profile = profile
        break

if matching_profile is None:
    raise RuntimeError("Requested color profile not found")

queue = rs.frame_queue(100, keep_frames=True)

sensor.open(matching_profile)

if sensor.supports(rs.option.enable_auto_exposure):
    sensor.set_option(rs.option.enable_auto_exposure, 0)

if sensor.supports(rs.option.exposure):
    sensor.set_option(rs.option.exposure, 10_000)

if sensor.supports(rs.option.gain):
    sensor.set_option(rs.option.gain, 16)

timestamps = []
frame_numbers = []

try:
    sensor.start(queue)

    for _ in range(30):
        queue.wait_for_frame()

    start = time.monotonic()

    while time.monotonic() - start < DURATION:
        frame = queue.wait_for_frame()

        timestamps.append(float(frame.get_timestamp()))
        frame_numbers.append(int(frame.get_frame_number()))

    elapsed = time.monotonic() - start

finally:
    sensor.stop()
    sensor.close()

dt = np.diff(timestamps) / 1000.0
frame_steps = np.diff(frame_numbers)
dropped = np.maximum(frame_steps - 1, 0).sum()

print("\nResults")
print(f"Received: {len(timestamps)}")
print(f"Rate: {len(timestamps) / elapsed:.2f} Hz")
print(f"Dropped: {int(dropped)}")
print(f"Mean dt: {dt.mean() * 1000:.2f} ms")
print(f"Median dt: {np.median(dt) * 1000:.2f} ms")
print(f"P95 dt: {np.percentile(dt, 95) * 1000:.2f} ms")
print(f"Max dt: {dt.max() * 1000:.2f} ms")