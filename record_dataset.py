#!/usr/bin/env python3

import argparse
import bisect
import select
import sys
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path

import cv2
import h5py
import numpy as np

from camera import RealSenseSource
from constants import JOINT_NAMES
from tcs_client import TCSReadClient


class QposPoller:
    def __init__(self, robot):
        self.robot = robot
        self.samples = deque()
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None

    def start(self):
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def _poll(self):
        try:
            while not self.stop_event.is_set():
                t0 = time.monotonic()
                state = self.robot.get_joint_state()
                t1 = time.monotonic()

                if isinstance(state, dict):
                    qpos = [state[name] for name in JOINT_NAMES]
                else:
                    qpos = state

                qpos = np.asarray(qpos, dtype=np.float64)
                sample_time = 0.5 * (t0 + t1)

                with self.condition:
                    self.samples.append((sample_time, qpos))

                    while self.samples and self.samples[0][0] < sample_time - 3.0:
                        self.samples.popleft()

                    self.condition.notify_all()

        except BaseException as error:
            self.error = error
            with self.condition:
                self.condition.notify_all()

    def get_qpos(self, image_time, timeout=0.06, max_window=0.06):
        deadline = time.monotonic() + timeout

        with self.condition:
            while not self.samples or self.samples[-1][0] < image_time:
                if self.error is not None:
                    raise RuntimeError("Qpos poller stopped") from self.error

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None

                self.condition.wait(remaining)

            samples = list(self.samples)

        times = [sample[0] for sample in samples]
        index_after = bisect.bisect_left(times, image_time)

        if index_after == 0 or index_after == len(samples):
            return None

        time_before, qpos_before = samples[index_after - 1]
        time_after, qpos_after = samples[index_after]

        window = time_after - time_before
        if window <= 0 or window > max_window:
            return None

        alpha = (image_time - time_before) / window
        qpos = qpos_before + alpha * (qpos_after - qpos_before)

        nearest_offset = min(image_time - time_before, time_after - image_time)

        return qpos, nearest_offset, window

    def close(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2.0)

        self.robot.close()


class RawKeyboard:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self):
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        return sys.stdin.read(1).lower() if ready else None

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def empty_episode(camera_names):
    return {
        "images": {name: [] for name in camera_names},
        "qpos": [],
        "image_time": [],
        "rs_timestamp_ms": [],
        "frame_number": [],
        "qpos_offset": [],
        "qpos_window": [],
        "sync_skips": 0,
    }


def add_sample(episode, camera_sample, synced):
    qpos, offset, window = synced

    for name, image in camera_sample["frames"].items():
        episode["images"][name].append(image.copy())

    episode["qpos"].append(qpos.astype(np.float32))
    episode["image_time"].append(camera_sample["image_time"])
    episode["rs_timestamp_ms"].append(camera_sample["rs_timestamp_ms"])
    episode["frame_number"].append(camera_sample["frame_number"])
    episode["qpos_offset"].append(offset)
    episode["qpos_window"].append(window)


def next_episode_path(dataset_dir):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    numbers = []

    for path in dataset_dir.glob("episode_*.hdf5"):
        try:
            numbers.append(int(path.stem.split("_")[-1]))
        except ValueError:
            pass

    number = max(numbers, default=-1) + 1
    return dataset_dir / f"episode_{number}.hdf5"


def save_episode(episode, dataset_dir):
    path = next_episode_path(dataset_dir)

    qpos = np.asarray(episode["qpos"], dtype=np.float32)
    image_time = np.asarray(episode["image_time"], dtype=np.float64)
    frame_number = np.asarray(episode["frame_number"], dtype=np.int64)

    action = np.concatenate([qpos[1:], qpos[-1:]])

    frame_steps = np.diff(frame_number)
    dropped = int(np.maximum(frame_steps - 1, 0).sum())
    duration = image_time[-1] - image_time[0]
    actual_fps = (len(qpos) - 1) / duration

    with h5py.File(path, "w") as root:
        observations = root.create_group("observations")
        observations.create_dataset("qpos", data=qpos)

        images_group = observations.create_group("images")
        for name, images in episode["images"].items():
            images = np.stack(images)
            images_group.create_dataset(name, data=images, chunks=(1, *images.shape[1:]), compression="lzf")

        root.create_dataset("action", data=action)

        timestamps = root.create_group("timestamps")
        timestamps.create_dataset("image_time", data=image_time)
        timestamps.create_dataset("rs_timestamp_ms", data=episode["rs_timestamp_ms"])
        timestamps.create_dataset("frame_number", data=frame_number)
        timestamps.create_dataset("qpos_offset", data=episode["qpos_offset"])
        timestamps.create_dataset("qpos_window", data=episode["qpos_window"])

    print(
        f"Saved {path.name}: {len(qpos)} samples, "
        f"{actual_fps:.2f} Hz, {dropped} drops, "
        f"{episode['sync_skips']} sync skips"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/default"))
    parser.add_argument("--host", default="192.168.0.10")
    parser.add_argument("--port", type=int, default=10100)
    parser.add_argument("--camera-name", default="wrist_d405")
    parser.add_argument("--serial", default="130322273198")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", type=int, nargs=2, default=(640, 480))
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--align-depth", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--sync-timeout", type=float, default=0.06)
    parser.add_argument("--max-qpos-window", type=float, default=0.06)

    return parser.parse_args()


def main():
    args = parse_args()

    depth_name = None
    if args.depth:
        depth_name = f"{args.camera_name}_depth"

    camera_names = [args.camera_name]
    if depth_name is not None:
        camera_names.append(depth_name)

    camera = RealSenseSource(
        color_name=args.camera_name, depth_name=depth_name,
        size=tuple(args.size), fps=args.fps, serial=args.serial,
        align_depth=args.align_depth,
    ).start()

    robot = TCSReadClient(host=args.host, port=args.port, timeout=5.0, verbose=False)

    poller = QposPoller(robot).start()
    episode = None
    preview_enabled = args.preview

    print(
        "\n[r] start  [f] save  [c] cancel  "
        "[p] preview  [x] exit\n"
    )

    try:
        with RawKeyboard() as keyboard:
            while True:
                sample = camera.read()

                if preview_enabled:
                    image = sample["frames"][args.camera_name]
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    text = "IDLE"
                    if episode is not None:
                        text = f"REC {len(episode['qpos'])}"

                    cv2.putText(
                        image,
                        text,
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("Dataset recorder", image)
                    cv2.waitKey(1)

                if episode is not None:
                    synced = poller.get_qpos(
                        sample["image_time"], args.sync_timeout, args.max_qpos_window
                    )

                    if synced is None:
                        episode["sync_skips"] += 1
                    else:
                        add_sample(episode, sample, synced)

                key = keyboard.read()

                if key == "r":
                    if episode is None:
                        episode = empty_episode(camera_names)
                        print("Recording started")

                elif key == "f":
                    if episode is None:
                        print("No active recording")
                    elif len(episode["qpos"]) < 2:
                        print("Too few samples")
                    else:
                        save_episode(episode, args.dataset_dir)
                        episode = None

                elif key == "c":
                    episode = None
                    print("Recording cancelled")

                elif key == "p":
                    preview_enabled = not preview_enabled
                    if not preview_enabled:
                        cv2.destroyAllWindows()

                elif key in {"x", "q"}:
                    break

    finally:
        cv2.destroyAllWindows()
        camera.close()
        poller.close()


if __name__ == "__main__":
    main()