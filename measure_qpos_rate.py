#!/usr/bin/env python3

import argparse
import time

import numpy as np

from tcs_client import TCSCommandError, TCSReadClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.0.10")
    parser.add_argument("--port", type=int, default=10100)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    client = TCSReadClient(
        host=args.host,
        port=args.port,
        timeout=5.0,
        verbose=False,
    )

    sample_times = []
    command_durations = []
    failures = 0

    try:
        print(f"Warm-up: {args.warmup} metingen")

        for _ in range(args.warmup):
            client.get_joint_state()

        print(f"Meten gedurende {args.duration:.1f} seconden...")

        measurement_start = time.monotonic()
        deadline = measurement_start + args.duration

        while time.monotonic() < deadline:
            command_start = time.monotonic()

            try:
                client.get_joint_state()
            except (TCSCommandError, OSError, EOFError) as exc:
                failures += 1
                print(f"Meetfout: {exc}")
                continue

            command_end = time.monotonic()

            # Schatting van het moment waarop de qpos geldig was.
            sample_time = 0.5 * (command_start + command_end)

            sample_times.append(sample_time)
            command_durations.append(command_end - command_start)

        measurement_end = time.monotonic()

    finally:
        client.close()

    if len(sample_times) < 2:
        raise RuntimeError("Te weinig succesvolle metingen.")

    sample_times = np.asarray(sample_times)
    command_durations = np.asarray(command_durations)
    intervals = np.diff(sample_times)

    total_duration = measurement_end - measurement_start
    throughput = len(sample_times) / total_duration

    print("\nResultaten")
    print(f"Succesvolle samples:  {len(sample_times)}")
    print(f"Mislukte samples:     {failures}")
    print(f"Totale meettijd:      {total_duration:.3f} s")
    print(f"Gemiddelde poll rate: {throughput:.2f} Hz")

    print("\nTijd tussen qpos-samples")
    print(f"Gemiddeld: {intervals.mean() * 1000:.2f} ms")
    print(f"Mediaan:   {np.median(intervals) * 1000:.2f} ms")
    print(f"P95:       {np.percentile(intervals, 95) * 1000:.2f} ms")
    print(f"Maximum:   {intervals.max() * 1000:.2f} ms")

    print("\nDuur van één wherej-opdracht")
    print(f"Gemiddeld: {command_durations.mean() * 1000:.2f} ms")
    print(f"Mediaan:   {np.median(command_durations) * 1000:.2f} ms")
    print(f"P95:       {np.percentile(command_durations, 95) * 1000:.2f} ms")
    print(f"Maximum:   {command_durations.max() * 1000:.2f} ms")


if __name__ == "__main__":
    main()