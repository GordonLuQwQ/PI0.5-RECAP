# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plot matched RTC and no-RTC PiperX joint trajectories."""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))


@dataclass(frozen=True)
class JointTrace:
    """One completed task's measured and commanded arm-joint trajectory."""

    directory: Path
    instruction: str
    pair: tuple[int, int]
    seed: int
    success: bool
    measured: np.ndarray
    targets: np.ndarray
    target_changes: np.ndarray

    @property
    def control_steps(self) -> int:
        """Return the number of actions applied to the environment."""
        return self.targets.shape[0]


def _load_trace(directory: Path) -> JointTrace:
    report = json.loads((directory / "report.json").read_text())
    if report.get("status") != "completed":
        raise ValueError(f"Trial is incomplete: {directory}")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as values:
        states = values["state"].astype(np.float64)
        targets = values["actions_requested"].astype(np.float64)
        final_state = values["final_state"].astype(np.float64)
    if states.ndim != 2 or targets.shape != states.shape or states.shape[1] != 7:
        raise ValueError(
            f"Expected state and action arrays shaped [steps, 7]: {directory}"
        )
    if final_state.shape != (1, 7):
        raise ValueError(f"Expected final_state shaped [1, 7]: {directory}")
    measured = np.concatenate((states[:, :6], final_state[:, :6]), axis=0)
    initial = np.load(directory / "initial_state.npy").astype(np.float64)
    np.testing.assert_allclose(measured[0], initial[0, :6], atol=1e-7, rtol=0)
    target_changes = np.diff(
        np.concatenate((initial[:, :6], targets[:, :6]), axis=0), axis=0
    )
    return JointTrace(
        directory=directory,
        instruction=report["instruction"],
        pair=tuple(report["pair"]),
        seed=int(report["scene_seed"]),
        success=bool(report["task_result"]["success"]),
        measured=measured,
        targets=targets[:, :6],
        target_changes=target_changes,
    )


def _assert_same_initial_scene(plain: Path, rtc: Path) -> None:
    np.testing.assert_allclose(
        np.load(plain / "initial_state.npy"),
        np.load(rtc / "initial_state.npy"),
        atol=1e-7,
        rtol=0,
        err_msg=f"Initial robot states differ for {plain.name}",
    )
    with (
        np.load(plain / "initial_truth.npz", allow_pickle=False) as plain_truth,
        np.load(rtc / "initial_truth.npz", allow_pickle=False) as rtc_truth,
    ):
        if set(plain_truth.files) != set(rtc_truth.files):
            raise ValueError(f"Initial scene fields differ for {plain.name}")
        for key in plain_truth.files:
            np.testing.assert_allclose(
                plain_truth[key],
                rtc_truth[key],
                atol=1e-7,
                rtol=0,
                err_msg=f"Initial scene field {key!r} differs for {plain.name}",
            )


def _write_csv(path: Path, plain: JointTrace, rtc: JointTrace) -> None:
    fields = ["step"]
    for mode in ("no_rtc", "rtc"):
        for quantity in ("measured", "target", "target_change"):
            fields.extend(f"{mode}_{quantity}_{joint}" for joint in JOINT_NAMES)
    rows = max(plain.control_steps, rtc.control_steps) + 1

    def values_at(array: np.ndarray, step: int) -> list[float | str]:
        if step >= len(array):
            return [""] * 6
        return [float(value) for value in array[step]]

    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for step in range(rows):
            row: list[float | int | str] = [step]
            for trace in (plain, rtc):
                row.extend(values_at(trace.measured, step))
                row.extend(values_at(trace.targets, step))
                row.extend(values_at(trace.target_changes, step))
            writer.writerow(row)


def _plot(path: Path, plain: JointTrace, rtc: JointTrace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        6, 3, figsize=(17, 19), sharex="col", constrained_layout=True
    )
    modes = ((plain, "no RTC", "#2878b5"), (rtc, "RTC", "#d94841"))
    for joint, name in enumerate(JOINT_NAMES):
        for trace, label, color in modes:
            measured_steps = np.arange(len(trace.measured))
            action_steps = np.arange(trace.control_steps)
            axes[joint, 0].plot(
                measured_steps,
                trace.measured[:, joint],
                color=color,
                linewidth=1.2,
                label=label,
            )
            axes[joint, 1].plot(
                action_steps,
                trace.targets[:, joint],
                color=color,
                linewidth=1.0,
            )
            axes[joint, 2].plot(
                action_steps,
                trace.target_changes[:, joint],
                color=color,
                linewidth=0.9,
            )
        axes[joint, 0].set_ylabel(f"{name} (rad)")
        axes[joint, 2].axhline(0.0, color="#777777", linewidth=0.6)
        for axis in axes[joint]:
            axis.grid(alpha=0.2)
    axes[0, 0].set_title("Measured joint angle")
    axes[0, 1].set_title("Requested joint target")
    axes[0, 2].set_title("Target change per control step")
    for axis in axes[-1]:
        axis.set_xlabel("Environment control step")
    figure.legend(
        handles=[
            Line2D([0], [0], color=color, label=label) for _, label, color in modes
        ],
        loc="upper right",
    )
    figure.suptitle(
        f"{plain.instruction} | seed={plain.seed}\n"
        f"no RTC: {plain.control_steps} steps, success={plain.success}; "
        f"RTC: {rtc.control_steps} steps, success={rtc.success}",
        fontsize=14,
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _trace_metrics(trace: JointTrace) -> dict:
    measured_changes = np.diff(trace.measured, axis=0)
    return {
        "control_steps": trace.control_steps,
        "success": trace.success,
        "max_abs_measured_change_rad": np.max(
            np.abs(measured_changes), axis=0
        ).tolist(),
        "p95_abs_measured_change_rad": np.percentile(
            np.abs(measured_changes), 95, axis=0
        ).tolist(),
        "max_abs_target_change_rad": np.max(
            np.abs(trace.target_changes), axis=0
        ).tolist(),
        "p95_abs_target_change_rad": np.percentile(
            np.abs(trace.target_changes), 95, axis=0
        ).tolist(),
    }


def generate_joint_comparisons(root: Path, output: Path | None = None) -> Path:
    """Validate paired trials and write their joint plots, CSVs, and summary."""
    root = root.resolve()
    plain_root = root / "no_rtc" / "standard"
    rtc_root = root / "rtc" / "standard"
    plain_trials = {
        path.name: path
        for path in plain_root.glob("pair*_seed*")
        if (path / "trajectory.npz").is_file()
    }
    rtc_trials = {
        path.name: path
        for path in rtc_root.glob("pair*_seed*")
        if (path / "trajectory.npz").is_file()
    }
    if plain_trials.keys() != rtc_trials.keys():
        missing_plain = sorted(rtc_trials.keys() - plain_trials.keys())
        missing_rtc = sorted(plain_trials.keys() - rtc_trials.keys())
        raise ValueError(
            f"Paired trials differ; missing no-RTC={missing_plain}, missing RTC={missing_rtc}"
        )
    if not plain_trials:
        raise ValueError(f"No completed paired trials found under {root}")

    output = (output or root / "joint_comparison").resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for name in sorted(plain_trials):
        plain_dir, rtc_dir = plain_trials[name], rtc_trials[name]
        _assert_same_initial_scene(plain_dir, rtc_dir)
        plain, rtc = _load_trace(plain_dir), _load_trace(rtc_dir)
        if (plain.instruction, plain.pair, plain.seed) != (
            rtc.instruction,
            rtc.pair,
            rtc.seed,
        ):
            raise ValueError(f"Trial metadata differs for {name}")
        plot_path = output / f"{name}_joint_curves.png"
        csv_path = output / f"{name}_joint_curves.csv"
        _plot(plot_path, plain, rtc)
        _write_csv(csv_path, plain, rtc)
        comparisons.append(
            {
                "trial": name,
                "instruction": plain.instruction,
                "pair": list(plain.pair),
                "scene_seed": plain.seed,
                "initial_scene_matches": True,
                "plot": plot_path.name,
                "csv": csv_path.name,
                "no_rtc": _trace_metrics(plain),
                "rtc": _trace_metrics(rtc),
            }
        )

    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(
            {"joint_names": JOINT_NAMES, "comparisons": comparisons},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    cards = "".join(
        f"<h2>{html.escape(row['instruction'])}</h2>"
        f"<p>seed {row['scene_seed']} · "
        f'<a href="{row["csv"]}">CSV</a></p>'
        f'<img src="{row["plot"]}" style="max-width:100%">'
        for row in comparisons
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        "<title>RTC 与 no-RTC 关节曲线</title>"
        "<h1>相同场景下的 RTC 与 no-RTC 关节曲线</h1>" + cards
    )
    return summary_path


def main() -> None:
    """Parse command-line arguments and generate all paired plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Paired evaluation root")
    parser.add_argument("--output", type=Path, help="Plot output directory")
    args = parser.parse_args()
    summary = generate_joint_comparisons(args.root, args.output)
    print(summary)


if __name__ == "__main__":
    main()
