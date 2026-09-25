"""Build a positive-only LeRobot set from policy successes and IK corrections.

Successful policy trials contain states and actions but no full-rate two-view
video.  This exporter restores their persisted Genesis reset snapshots,
replays the applied actions, renders both training cameras, then appends only
the ``is_expert == 1`` suffixes from the correction collection.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .collect import write_json
from .protocol import PAIRS, instruction
from .recover_policy_failures import restore_cameras
from .runtime import init_genesis

CAMERA_FEATURES = {
    "third_person": "observation.images.image",
    "wrist": "observation.images.image2",
}
JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_closure",
]
POSITIVE_SUFFIX = "\nAdvantage: positive"
ADVANTAGE_TAG = "piperx_all_true_edited"


def _load_successes(
    source: Path,
    *,
    expected_completed: int | None = None,
    expected_successes: int | None = None,
) -> tuple[dict, list[dict]]:
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
    for report_path in sorted(source.glob("pair*_seed*/report.json")):
        report = json.loads(report_path.read_text())
        if report.get("status") != "completed":
            continue
        completed += 1
        if not report.get("task_result", {}).get("success", False):
            continue
        directory = report_path.parent
        for filename in ("trajectory.npz", "initial_truth.npz", "initial_state.npy"):
            if not (directory / filename).is_file():
                raise FileNotFoundError(f"Successful trial is incomplete: {directory / filename}")
        pair = tuple(report["pair"])
        if pair not in PAIRS:
            raise ValueError(f"Unknown task pair in {report_path}: {pair}")
        records.append(
            {
                "directory": directory,
                "pair": pair,
                "instruction": str(report["instruction"]),
                "scene_seed": int(report["scene_seed"]),
            }
        )
    if expected_completed is not None and completed != expected_completed:
        raise ValueError(f"Expected {expected_completed} completed policy trials, found {completed}")
    if expected_successes is not None and len(records) != expected_successes:
        raise ValueError(f"Expected {expected_successes} successful policy trials, found {len(records)}")
    records.sort(key=lambda row: (PAIRS.index(row["pair"]), row["scene_seed"]))
    return manifest, records


def _load_corrections(source: Path) -> tuple[dict, list[dict]]:
    collection_path = source / "collection.json"
    if not collection_path.is_file():
        raise FileNotFoundError(f"Missing correction metadata: {collection_path}")
    collection = json.loads(collection_path.read_text())
    if collection.get("kind") != "policy_prefix_ik_correction":
        raise ValueError(f"Not a correction collection: {source}")
    if tuple(collection.get("cameras", ())) != ("third_person", "wrist"):
        raise ValueError("Correction data must contain [third_person, wrist]")

    records = []
    for record_path in sorted(source.glob("episode_*.json")):
        record = json.loads(record_path.read_text())
        if not record.get("saved", False):
            continue
        archive = source / record["archive"]
        if not archive.is_file():
            raise FileNotFoundError(f"Missing correction archive: {archive}")
        records.append({**record, "archive_path": archive})
    if not records:
        raise ValueError(f"No saved correction episodes found in {source}")
    return collection, records


def _features(env) -> dict:
    result = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
    }
    for camera_name, feature_name in CAMERA_FEATURES.items():
        width, height = env.cameras[camera_name].res
        result[feature_name] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return result


def _add_frame(
    dataset,
    *,
    state: np.ndarray,
    third_person: np.ndarray,
    wrist: np.ndarray,
    action: np.ndarray,
    task: str,
) -> None:
    dataset.add_frame(
        {
            "observation.state": np.asarray(state, dtype=np.float32),
            "observation.images.image": np.moveaxis(third_person, -1, 0),
            "observation.images.image2": np.moveaxis(wrist, -1, 0),
            "action": np.asarray(action, dtype=np.float32),
            "task": task,
        }
    )


def _restore_snapshot_batch(env, source: Path, records: list[dict]) -> None:
    """Restore one persisted single-environment reset snapshot per vector row."""
    import torch

    if len(records) != env.num_envs:
        raise ValueError(f"Expected {env.num_envs} reset records, got {len(records)}")
    snapshots: list[dict[str, np.ndarray]] = []
    for record in records:
        path = source / "reset_states" / f"{record['scene_seed']}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing persisted reset state: {path}")
        with np.load(path, allow_pickle=False) as values:
            snapshots.append({name: values[name].copy() for name in values.files})

    env.scene.reset()
    state = env.scene.get_state()
    for solver_index, solver_state in enumerate(state.solvers_state):
        if solver_state is None:
            continue
        for name, target in vars(solver_state).items():
            if not isinstance(target, torch.Tensor):
                continue
            key = f"{solver_index}_{name}"
            rows = [snapshot[key] for snapshot in snapshots]
            if any(row.shape[0] != 1 for row in rows):
                raise ValueError(f"Reset snapshot {key} is not single-environment data")
            combined = np.concatenate(rows, axis=0)
            if tuple(combined.shape) != tuple(target.shape):
                raise ValueError(
                    f"Reset snapshot {key} shape {combined.shape} does not match "
                    f"batched Genesis state {tuple(target.shape)}"
                )
            target.copy_(torch.as_tensor(combined, device=target.device, dtype=target.dtype))

    env.last_action = np.concatenate([snapshot["last_action"] for snapshot in snapshots], axis=0).astype(np.float32)
    env.scene.reset(state)
    env.initial = env.truth()

    for env_index, record in enumerate(records):
        directory = Path(record["directory"])
        with np.load(directory / "initial_truth.npz", allow_pickle=False) as saved:
            for key in ("positions", "quaternions", "tcp", "tcp_quaternion"):
                np.testing.assert_allclose(env.initial[key][env_index], saved[key][0], atol=2e-6, rtol=0)
        saved_state = np.load(directory / "initial_state.npy")
        np.testing.assert_allclose(env.state()[env_index], saved_state[0], atol=2e-6, rtol=0)


def _load_policy_trace(
    record: dict,
    *,
    third_person_shape: tuple[int, int, int],
    wrist_shape: tuple[int, int, int],
) -> dict[str, Any]:
    """Load one successful policy trace and allocate its replay frame buffers."""
    with np.load(Path(record["directory"]) / "trajectory.npz", allow_pickle=False) as episode:
        states = episode["state"].astype(np.float32)
        actions = episode["actions_applied"].astype(np.float32)
        final_state = episode["final_state"].astype(np.float32)
    if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != 7:
        raise ValueError(f"Invalid policy trace shape in {record['directory']}")
    if final_state.shape == (1, 7):
        final_state = final_state[0]
    if final_state.shape != (7,):
        raise ValueError(f"Invalid final state shape in {record['directory']}")
    return {
        "record": record,
        "expected_states": states,
        "actions": actions,
        "final_state": final_state,
        "states": np.empty_like(states),
        "third_person": np.empty((len(actions), *third_person_shape), dtype=np.uint8),
        "wrist": np.empty((len(actions), *wrist_shape), dtype=np.uint8),
        "max_state_error": 0.0,
        "final_state_error": None,
    }


def _replay_policy_batch(
    env,
    source: Path,
    records: list[dict],
    *,
    state_atol: float | None,
) -> list[dict[str, Any]]:
    """Replay different-length policy traces concurrently in one Genesis scene."""
    if not records or len(records) > env.num_envs:
        raise ValueError("Replay batch must contain between one and num_envs records")
    padded_records = records + [records[-1]] * (env.num_envs - len(records))
    _restore_snapshot_batch(env, source, padded_records)
    traces = [
        _load_policy_trace(
            record,
            third_person_shape=(
                env.cameras["third_person"].res[1],
                env.cameras["third_person"].res[0],
                3,
            ),
            wrist_shape=(
                env.cameras["wrist"].res[1],
                env.cameras["wrist"].res[0],
                3,
            ),
        )
        for record in records
    ]
    max_steps = max(len(trace["actions"]) for trace in traces)

    for step in range(max_steps):
        observation = env.observe(camera_names=tuple(CAMERA_FEATURES))
        actions = env.last_action.copy()
        for env_index, trace in enumerate(traces):
            if step >= len(trace["actions"]):
                continue
            state_error = float(np.max(np.abs(observation["state"][env_index] - trace["expected_states"][step])))
            trace["max_state_error"] = max(trace["max_state_error"], state_error)
            if state_atol is not None and state_error > state_atol:
                raise RuntimeError(
                    f"Replay diverged at {trace['record']['directory']} step {step}: "
                    f"state error {state_error:.8g} > {state_atol:.8g}"
                )
            trace["states"][step] = observation["state"][env_index]
            trace["third_person"][step] = observation["third_person"][env_index]
            trace["wrist"][step] = observation["wrist"][env_index]
            actions[env_index] = trace["actions"][step]

        env.step(actions)
        ending = [env_index for env_index, trace in enumerate(traces) if step + 1 == len(trace["actions"])]
        if ending:
            current_state = env.state()
            for env_index in ending:
                trace = traces[env_index]
                final_error = float(np.max(np.abs(current_state[env_index] - trace["final_state"])))
                trace["final_state_error"] = final_error
                if state_atol is not None and final_error > state_atol:
                    raise RuntimeError(
                        f"Final replay state diverged for "
                        f"{trace['record']['directory']}: {final_error:.8g} > "
                        f"{state_atol:.8g}"
                    )

    return traces


def _write_true_advantages(output: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_files = sorted((output / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {output / 'data'}")
    tables = [pq.read_table(path, columns=["episode_index", "frame_index"]) for path in parquet_files]
    table = pa.concat_tables(tables, promote_options="default")
    count = len(table)
    sidecar = pa.table(
        {
            "episode_index": table.column("episode_index"),
            "frame_index": table.column("frame_index"),
            "advantage_continuous": pa.array(np.ones(count, dtype=np.float32)),
            "advantage": pa.array(np.ones(count, dtype=np.bool_)),
        }
    )
    tagged = output / "meta" / f"advantages_{ADVANTAGE_TAG}.parquet"
    default = output / "meta" / "advantages.parquet"
    pq.write_table(sidecar, tagged)
    pq.write_table(sidecar, default)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="Camera calibration at its current downloaded location")
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/piperx_positive_recap")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--num-envs", type=int, default=1)
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
    parser.add_argument("--max-policy-successes", type=int)
    parser.add_argument("--max-corrections", type=int)
    parser.add_argument("--expected-completed", type=int)
    parser.add_argument("--expected-successes", type=int)
    args = parser.parse_args()
    args.rollouts = args.rollouts.expanduser().resolve()
    args.corrections = args.corrections.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error("Output must be a new directory")
    if args.state_atol is not None and args.state_atol <= 0:
        parser.error("--state-atol must be positive")
    if min(args.num_envs, args.encoder_threads, args.encoder_queue_maxsize) <= 0:
        parser.error("--num-envs, --encoder-threads and --encoder-queue-maxsize must be positive")
    for value, name in (
        (args.max_policy_successes, "--max-policy-successes"),
        (args.max_corrections, "--max-corrections"),
    ):
        if value is not None and value < 0:
            parser.error(f"{name} must be non-negative")

    manifest, policy_records = _load_successes(
        args.rollouts,
        expected_completed=args.expected_completed,
        expected_successes=args.expected_successes,
    )
    correction_collection, correction_records = _load_corrections(args.corrections)
    if args.max_policy_successes is not None:
        policy_records = policy_records[: args.max_policy_successes]
    if args.max_corrections is not None:
        correction_records = correction_records[: args.max_corrections]
    if not policy_records and not correction_records:
        parser.error("At least one policy success or correction must be selected")

    calibration = (args.calibration or Path(manifest["specification"]["calibration"])).expanduser().resolve()
    if not calibration.is_file():
        parser.error(f"Missing camera calibration: {calibration}")

    try:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise SystemExit("LeRobot and pyarrow are required in the Genesis environment") from error

    gs = init_genesis(args.backend)
    from .env import StackingEnv

    env = StackingEnv(gs, num_envs=args.num_envs, cameras=True)
    restore_cameras(env, calibration, tuple(CAMERA_FEATURES))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        robot_type="piperx",
        fps=20,
        features=_features(env),
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=args.encoder_threads,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=2, crf=18, preset="fast"),
    )

    started = time.perf_counter()
    policy_counts: Counter = Counter()
    correction_counts: Counter = Counter()
    policy_frames = 0
    correction_frames = 0
    replay_max_state_error = 0.0
    replay_max_final_state_error = 0.0
    completed = False
    try:
        for batch_start in range(0, len(policy_records), env.num_envs):
            batch_records = policy_records[batch_start : batch_start + env.num_envs]
            traces = _replay_policy_batch(
                env,
                args.rollouts,
                batch_records,
                state_atol=args.state_atol,
            )
            for local_index, trace in enumerate(traces):
                record = trace["record"]
                for state, third_person, wrist, action in zip(
                    trace["states"],
                    trace["third_person"],
                    trace["wrist"],
                    trace["actions"],
                    strict=True,
                ):
                    _add_frame(
                        dataset,
                        state=state,
                        third_person=third_person,
                        wrist=wrist,
                        action=action,
                        task=record["instruction"],
                    )
                dataset.save_episode()
                policy_counts[record["instruction"]] += 1
                policy_frames += len(trace["actions"])
                replay_max_state_error = max(replay_max_state_error, trace["max_state_error"])
                replay_max_final_state_error = max(replay_max_final_state_error, trace["final_state_error"])
                print(
                    json.dumps(
                        {
                            "source": "policy_success",
                            "episode": batch_start + local_index + 1,
                            "total": len(policy_records),
                            "instruction": record["instruction"],
                            "seed": record["scene_seed"],
                            "frames": len(trace["actions"]),
                            "parallel_envs": len(traces),
                        }
                    ),
                    flush=True,
                )
            del traces

        offset = len(policy_records)
        for index, record in enumerate(correction_records, start=1):
            with np.load(record["archive_path"], allow_pickle=False) as episode:
                expert = episode["is_expert"].astype(bool)
                indices = np.flatnonzero(expert)
                if not len(indices) or not np.array_equal(indices, np.arange(indices[0], len(expert))):
                    raise ValueError(f"Correction {record['archive_path']} has no contiguous expert suffix")
                if indices[0] != int(record["policy_steps"]) or len(indices) != int(record["ik_steps"]):
                    raise ValueError(f"Correction metadata disagrees with {record['archive_path']}")
                # NPZ members are compressed. Materialize each member once;
                # indexing ``episode[name]`` in the loop would decompress the
                # complete video again for every frame.
                states = episode["state"][indices]
                third_person = episode["third_person"][indices]
                wrist = episode["wrist"][indices]
                actions = episode["actions"][indices]
                for state, main_image, wrist_image, action in zip(states, third_person, wrist, actions, strict=True):
                    _add_frame(
                        dataset,
                        state=state,
                        third_person=main_image,
                        wrist=wrist_image,
                        action=action,
                        task=record["instruction"],
                    )
            dataset.save_episode()
            correction_counts[record["instruction"]] += 1
            correction_frames += len(indices)
            print(
                json.dumps(
                    {
                        "source": "ik_expert_suffix",
                        "episode": offset + index,
                        "total": offset + len(correction_records),
                        "instruction": record["instruction"],
                        "frames": len(indices),
                    }
                ),
                flush=True,
            )

        dataset.finalize()
        advantage_rows = _write_true_advantages(args.output)
        total_frames = policy_frames + correction_frames
        if advantage_rows != total_frames:
            raise RuntimeError(f"Advantage rows ({advantage_rows}) != exported frames ({total_frames})")
        completed = True
    finally:
        if not completed:
            try:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                dataset.finalize()
            except Exception as error:  # noqa: BLE001
                print(
                    json.dumps({"cleanup_warning": str(error)}),
                    flush=True,
                )
        env.close()

    summary = {
        "kind": "piperx_positive_recap_sft_edited",
        "repo_id": args.repo_id,
        "policy_rollouts": str(args.rollouts),
        "corrections": str(args.corrections),
        "policy_success_episodes": len(policy_records),
        "policy_success_frames": policy_frames,
        "policy_successes_by_task": dict(policy_counts),
        "expert_suffix_episodes": len(correction_records),
        "expert_suffix_frames": correction_frames,
        "expert_suffixes_by_task": dict(correction_counts),
        "total_episodes": len(policy_records) + len(correction_records),
        "total_frames": policy_frames + correction_frames,
        "advantage": True,
        "advantage_tag": ADVANTAGE_TAG,
        "training_prompt_suffix": POSITIVE_SUFFIX,
        "replay_max_state_error": replay_max_state_error,
        "replay_max_final_state_error": replay_max_final_state_error,
        "replay_num_envs": args.num_envs,
        "encoder_threads": args.encoder_threads,
        "camera_calibration": str(calibration),
        "control_hz": int(correction_collection["control_hz"]),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "evaluation_warning": (
            "The policy-success and correction scenes use former evaluation seeds; "
            "evaluate the resulting policy on a new untouched seed list."
        ),
    }
    write_json(args.output / "positive_export.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
