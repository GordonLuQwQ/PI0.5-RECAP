"""Replay all failed policy trials as a two-view LeRobot value dataset."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .collect import write_json
from .export_positive_recap_lerobot import CAMERA_FEATURES, _features
from .protocol import PAIRS
from .recover_policy_failures import restore_cameras, restore_snapshot
from .runtime import init_genesis


def _load_failures(
    source: Path,
    *,
    expected_completed: int | None = None,
    expected_successes: int | None = None,
    expected_failures: int | None = None,
) -> tuple[dict, list[dict]]:
    """Validate the evaluation suite and select every completed failure."""
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing policy-rollout manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    specification = manifest.get("specification", {})
    declared_pairs = tuple(tuple(pair) for pair in specification.get("pairs", ()))
    if declared_pairs != PAIRS:
        raise ValueError(f"Expected pair order {PAIRS}, got {declared_pairs}")
    if tuple(specification.get("cameras", ())) != ("third_person", "wrist"):
        raise ValueError("Source rollout must use [third_person, wrist]")

    records = []
    completed = 0
    successes = 0
    for report_path in sorted(source.glob("pair*_seed*/report.json")):
        report = json.loads(report_path.read_text())
        if report.get("status") != "completed":
            continue
        completed += 1
        if report.get("task_result", {}).get("success", False):
            successes += 1
            continue

        directory = report_path.parent
        for filename in ("trajectory.npz", "initial_truth.npz", "initial_state.npy"):
            if not (directory / filename).is_file():
                raise FileNotFoundError(f"Failed policy trial is incomplete: {directory / filename}")
        pair = tuple(report["pair"])
        if pair not in PAIRS:
            raise ValueError(f"Unknown task pair in {report_path}: {pair}")
        task_result = report.get("task_result", {})
        records.append(
            {
                "directory": directory,
                "pair": pair,
                "instruction": str(report["instruction"]),
                "scene_seed": int(report["scene_seed"]),
                "failure": str(task_result.get("failure") or "unknown"),
                "control_hz": int(report["protocol"]["control_hz"]),
            }
        )

    for expected, actual, label in (
        (expected_completed, completed, "completed policy trials"),
        (expected_successes, successes, "successful policy trials"),
        (expected_failures, len(records), "failed policy trials"),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"Expected {expected} {label}, found {actual}")
    if not records:
        raise ValueError("The rollout set contains no failed policy trials")
    if len({record["control_hz"] for record in records}) != 1:
        raise ValueError("Failed policy trials disagree about control frequency")
    records.sort(key=lambda row: (PAIRS.index(row["pair"]), row["scene_seed"]))
    return manifest, records


def _add_failure_frame(
    dataset,
    observation: dict,
    action: np.ndarray,
    task: str,
) -> None:
    dataset.add_frame(
        {
            "observation.state": observation["state"][0].astype(np.float32),
            "observation.images.image": np.moveaxis(observation["third_person"][0], -1, 0),
            "observation.images.image2": np.moveaxis(observation["wrist"][0], -1, 0),
            "action": np.asarray(action, dtype=np.float32),
            "is_success": np.asarray([False], dtype=bool),
            "task": task,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="Camera calibration at its current downloaded location")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/piperx_policy_failures")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--encoder-queue-maxsize", type=int, default=1024)
    parser.add_argument(
        "--state-atol",
        type=float,
        help=(
            "Optional diagnostic hard limit for replay state drift. By default drift is recorded "
            "and the original completed evaluation report supplies the outcome label."
        ),
    )
    parser.add_argument("--max-failures", type=int)
    parser.add_argument("--expected-completed", type=int)
    parser.add_argument("--expected-successes", type=int)
    parser.add_argument("--expected-failures", type=int)
    args = parser.parse_args()
    args.rollouts = args.rollouts.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error("Output must be a new directory")
    if args.state_atol is not None and args.state_atol <= 0:
        parser.error("--state-atol must be positive")
    if args.max_failures is not None and args.max_failures <= 0:
        parser.error("--max-failures must be positive")

    manifest, records = _load_failures(
        args.rollouts,
        expected_completed=args.expected_completed,
        expected_successes=args.expected_successes,
        expected_failures=args.expected_failures,
    )
    if args.max_failures is not None:
        records = records[: args.max_failures]

    calibration = (args.calibration or Path(manifest["specification"]["calibration"])).expanduser().resolve()
    if not calibration.is_file():
        parser.error(f"Missing camera calibration: {calibration}")

    try:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise SystemExit("LeRobot and its video dependencies are required in the Genesis environment") from error

    gs = init_genesis(args.backend)
    from .env import StackingEnv

    env = StackingEnv(gs, num_envs=1, cameras=True)
    restore_cameras(env, calibration, tuple(CAMERA_FEATURES))
    features = _features(env)
    features["is_success"] = {"dtype": "bool", "shape": (1,), "names": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        robot_type="piperx",
        fps=records[0]["control_hz"],
        features=features,
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=args.encoder_threads,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
        rgb_encoder=RGBEncoderConfig(
            vcodec="h264",
            pix_fmt="yuv420p",
            g=2,
            crf=18,
            preset="fast",
        ),
    )

    started = time.perf_counter()
    task_counts: Counter = Counter()
    reason_counts: Counter = Counter()
    total_frames = 0
    replay_max_state_error = 0.0
    replay_max_final_state_error = 0.0
    completed = False
    try:
        for index, record in enumerate(records, start=1):
            restore_snapshot(
                env,
                args.rollouts,
                record["directory"],
                record["scene_seed"],
            )
            with np.load(record["directory"] / "trajectory.npz", allow_pickle=False) as episode:
                states = episode["state"].astype(np.float32)
                actions = episode["actions_applied"].astype(np.float32)
                final_state = episode["final_state"].astype(np.float32)
            if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != 7:
                raise ValueError(f"Invalid policy trace shape in {record['directory']}")

            for step, action in enumerate(actions):
                observation = env.observe(camera_names=tuple(CAMERA_FEATURES))
                state_error = float(np.max(np.abs(observation["state"][0] - states[step])))
                replay_max_state_error = max(replay_max_state_error, state_error)
                if args.state_atol is not None and state_error > args.state_atol:
                    raise RuntimeError(
                        f"Replay diverged at {record['directory']} step {step}: "
                        f"state error {state_error:.8g} > {args.state_atol:.8g}"
                    )
                _add_failure_frame(
                    dataset,
                    observation,
                    action,
                    record["instruction"],
                )
                env.step(action[None])

            final_error = float(np.max(np.abs(env.state() - final_state)))
            replay_max_final_state_error = max(replay_max_final_state_error, final_error)
            if args.state_atol is not None and final_error > args.state_atol:
                raise RuntimeError(
                    f"Final replay state diverged for {record['directory']}: {final_error:.8g} > {args.state_atol:.8g}"
                )
            dataset.save_episode()
            task_counts[record["instruction"]] += 1
            reason_counts[record["failure"]] += 1
            total_frames += len(actions)
            print(
                json.dumps(
                    {
                        "source": "policy_failure",
                        "episode": index,
                        "total": len(records),
                        "instruction": record["instruction"],
                        "seed": record["scene_seed"],
                        "failure": record["failure"],
                        "frames": len(actions),
                    }
                ),
                flush=True,
            )

        dataset.finalize()
        completed = True
    finally:
        if not completed:
            try:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                dataset.finalize()
            except Exception as error:  # noqa: BLE001
                print(json.dumps({"cleanup_warning": str(error)}), flush=True)
        env.close()

    summary = {
        "kind": "piperx_complete_policy_failures_edited",
        "repo_id": args.repo_id,
        "policy_rollouts": str(args.rollouts),
        "failure_episodes": len(records),
        "failure_frames": total_frames,
        "failures_by_task": dict(task_counts),
        "failures_by_reason": dict(reason_counts),
        "is_success": False,
        "outcome_source": "original_policy_report",
        "replay_max_state_error": replay_max_state_error,
        "replay_max_final_state_error": replay_max_final_state_error,
        "camera_calibration": str(calibration),
        "control_hz": records[0]["control_hz"],
        "elapsed_wall_seconds": time.perf_counter() - started,
    }
    write_json(args.output / "failure_export.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
