#!/usr/bin/env python3
"""Animate W/X/Y/Z quaternion CSV samples as a rotating 3D body.

Usage: python visualize_quaternion_orientation.py [recording.csv] [--speed 1.0]
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider


ROOT = Path(__file__).resolve().parent
QUATERNION_COLUMNS = [f"wristMotion_quaternion{axis}" for axis in "WXYZ"]


def load_motion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Validate timestamps and normalize scalar-first quaternion samples."""
    timestamps, quaternions = [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = ["secondsElapsed", *QUATERNION_COLUMNS]
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected columns {', '.join(required)}")
        for line, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["secondsElapsed"])
                quaternion = tuple(float(row[column]) for column in QUATERNION_COLUMNS)
                norm = math.hypot(*quaternion)
                if not all(math.isfinite(v) for v in (timestamp, *quaternion)) or norm == 0:
                    raise ValueError("timestamp and quaternion must be finite; quaternion cannot be zero")
                if timestamps and timestamp <= timestamps[-1]:
                    raise ValueError("timestamps must strictly increase")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line}: {error}") from error
            timestamps.append(timestamp)
            quaternions.append([value / norm for value in quaternion])
    if not timestamps:
        raise ValueError(f"{path}: no samples")
    times = np.asarray(timestamps)
    return times - times[0], np.asarray(quaternions)


def rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Hamilton active rotation: body coordinates to reference coordinates."""
    w, x, y, z = quaternion
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def show_motion(path: Path, speed: float) -> None:
    timestamps, quaternions = load_motion(path)
    figure = plt.figure(figsize=(10, 8))
    figure.subplots_adjust(bottom=0.25)
    axes = figure.add_subplot(111, projection="3d")
    axes.set(xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), zlim=(-1.6, 1.6),
             xlabel="Reference X", ylabel="Reference Y", zlabel="Reference Z")
    axes.set_box_aspect((1, 1, 1))
    axes.set_title(path.name, fontsize=10)
    colors = ("tab:red", "tab:green", "tab:blue")
    body_axes = []
    for index, color in enumerate(colors):
        endpoint = np.eye(3)[index] * 1.4
        axes.plot([0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]],
                  color=color, linestyle="--", alpha=0.25)
        body_axes.append(axes.plot([], [], [], color=color, linewidth=3,
                                   label=f"Body {'XYZ'[index]}")[0])
    axes.legend(loc="upper left")
    # An asymmetric rectangular body makes rotation around each axis visible.
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-0.45, 0.45)
                         for z in (-0.2, 0.2)])
    edges = [(a, b) for a in range(8) for b in range(a+1, 8)
             if np.count_nonzero(vertices[a] != vertices[b]) == 1]
    body_lines = [axes.plot([], [], [], color="0.35", linewidth=2)[0] for _ in edges]
    readout = figure.text(0.1, 0.19, "", family="monospace", fontsize=10)
    slider = Slider(figure.add_axes((0.15, 0.13, 0.7, 0.025)), "Sample", 0,
                    max(1, len(timestamps)-1), valinit=0, valstep=1)
    play = Button(figure.add_axes((0.15, 0.04, 0.15, 0.05)), "Play")
    back = Button(figure.add_axes((0.33, 0.04, 0.15, 0.05)), "Previous")
    forward = Button(figure.add_axes((0.51, 0.04, 0.15, 0.05)), "Next")
    restart = Button(figure.add_axes((0.69, 0.04, 0.15, 0.05)), "Restart")
    state = {"playing": False, "anchor": 0.0, "index": 0}

    def render(index: int) -> None:
        index = min(int(index), len(timestamps)-1)
        state["index"] = index
        matrix = rotation_matrix(quaternions[index])
        rotated = vertices @ matrix.T
        for line, (a, b) in zip(body_lines, edges):
            points = rotated[[a, b]]
            line.set_data_3d(points[:, 0], points[:, 1], points[:, 2])
        for axis, line in enumerate(body_axes):
            endpoint = matrix[:, axis] * 1.4
            line.set_data_3d([0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]])
        w, x, y, z = quaternions[index]
        readout.set_text(f"Sample {index+1}/{len(timestamps)}   t={timestamps[index]:.3f}s   speed={speed:g}x\n"
                         f"W={w:+.4f}  X={x:+.4f}  Y={y:+.4f}  Z={z:+.4f}")
        figure.canvas.draw_idle()

    def pause() -> None:
        state["playing"] = False
        play.label.set_text("Play")

    def seek(value: float) -> None:
        render(int(value))
        state["anchor"] = time.perf_counter() - timestamps[state["index"]] / speed

    def toggle(_event) -> None:
        if state["playing"]:
            pause()
        else:
            if state["index"] == len(timestamps)-1:
                slider.set_val(0)
            state["anchor"] = time.perf_counter() - timestamps[state["index"]] / speed
            state["playing"] = True
            play.label.set_text("Pause")

    def step(delta: int) -> None:
        pause()
        slider.set_val(max(0, min(len(timestamps)-1, state["index"] + delta)))

    def tick(_frame) -> None:
        if state["playing"]:
            elapsed = (time.perf_counter() - state["anchor"]) * speed
            index = min(len(timestamps)-1, int(np.searchsorted(timestamps, elapsed, side="right"))-1)
            # Update without invoking seek, which would reset the playback clock.
            slider.eventson = False
            slider.set_val(max(0, index))
            slider.eventson = True
            render(max(0, index))
            if elapsed >= timestamps[-1]:
                pause()

    slider.on_changed(seek)
    play.on_clicked(toggle)
    back.on_clicked(lambda event: step(-1))
    forward.on_clicked(lambda event: step(1))
    restart.on_clicked(lambda event: (pause(), slider.set_val(0)))
    render(0)
    animation = FuncAnimation(figure, tick, interval=16, cache_frame_data=False)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path)
    parser.add_argument("--speed", type=float, default=1.0, help="playback multiplier (default: 1)")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    path = args.csv
    if path is None:
        path = next(iter(sorted((ROOT / "preprocessing" / "data").rglob("part_*.csv"))), None)
        if path is None:
            parser.error("provide a CSV path or run preprocessing first")
    try:
        show_motion(path, args.speed)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
