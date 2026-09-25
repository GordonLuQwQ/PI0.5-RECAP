"""Export completed PiperX policy rollouts as one labeled LeRobot dataset.

The RLinf evaluator records the observation before every applied action and one
final video frame after the last action.  This exporter pairs the first ``T``
video frames with the ``T`` state/action rows from ``trajectory.npz`` and stores
the evaluator's terminal result in ``is_success`` for every frame.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .collect import write_json
from .protocol import PAIRS, instruction

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


def _load_records(source: Path, expected_per_pair: int) -> tuple[dict, list[dict]]:
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing rollout manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    specification = manifest.get("specification", {})
    declared_pairs = tuple(tuple(pair) for pair in specification.get("pairs", ()))
    if declared_pairs != PAIRS:
        raise ValueError(f"Expected pair order {PAIRS}, got {declared_pairs}")
    if tuple(specification.get("cameras", ())) != tuple(CAMERA_FEATURES):
        raise ValueError("Source rollout must use [third_person, wrist]")
    if tuple(specification.get("video_cameras", ())) != tuple(CAMERA_FEATURES):
        raise ValueError("Both rollout cameras must have saved videos")
    if int(specification.get("video_stride", 0)) != 1:
        raise ValueError("LeRobot export requires video_stride=1")

    records: list[dict] = []
    pair_counts: Counter = Counter()
    control_rates: set[int] = set()
    for report_path in sorted(source.glob("pair*_seed*/report.json")):
        report = json.loads(report_path.read_text())
        if report.get("status") != "completed":
            continue
        pair = tuple(report["pair"])
        if pair not in PAIRS:
            raise ValueError(f"Unknown task pair in {report_path}: {pair}")
        expected_instruction = instruction(pair)
        if report.get("instruction") != expected_instruction:
            raise ValueError(f"Instruction does not match pair in {report_path}")

        directory = report_path.parent
        trajectory_path = directory / "trajectory.npz"
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"Missing rollout trajectory: {trajectory_path}")
        videos = report.get("videos", {})
        video_paths = {}
        for camera in CAMERA_FEATURES:
            video = videos.get(camera)
            if not isinstance(video, dict):
                raise FileNotFoundError(f"Missing {camera} video metadata in {report_path}")
            video_path = directory / str(video["file"])
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing rollout video: {video_path}")
            video_paths[camera] = video_path

        control_hz = int(report["protocol"]["control_hz"])
        if any(float(videos[name]["fps"]) != control_hz for name in CAMERA_FEATURES):
            raise ValueError(f"Video rate does not match control rate in {report_path}")
        control_rates.add(control_hz)
        success = bool(report.get("task_result", {}).get("success", False))
        records.append(
            {
                "directory": directory,
                "trajectory": trajectory_path,
                "videos": video_paths,
                "pair": pair,
                "instruction": expected_instruction,
                "scene_seed": int(report["scene_seed"]),
                "success": success,
                "failure": report.get("task_result", {}).get("failure"),
                "control_hz": control_hz,
            }
        )
        pair_counts[pair] += 1

    expected_total = len(PAIRS) * expected_per_pair
    if len(records) != expected_total:
        raise ValueError(f"Expected {expected_total} completed rollouts, found {len(records)}")
    wrong_counts = {instruction(pair): pair_counts[pair] for pair in PAIRS if pair_counts[pair] != expected_per_pair}
    if wrong_counts:
        raise ValueError(f"Expected {expected_per_pair} rollouts per pair, got {wrong_counts}")
    if len(control_rates) != 1:
        raise ValueError(f"Rollouts disagree about control frequency: {sorted(control_rates)}")
    records.sort(key=lambda row: (PAIRS.index(row["pair"]), row["scene_seed"]))
    return manifest, records


def _decode_rgb(path: Path) -> tuple[Iterator[np.ndarray], object]:
    import av

    container = av.open(str(path))
    stream = container.streams.video[0]
    return (frame.to_ndarray(format="rgb24") for frame in container.decode(stream)), container


def _features(camera_shapes: dict[str, tuple[int, int]]) -> dict:
    features = {
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
        "is_success": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for camera, feature_name in CAMERA_FEATURES.items():
        height, width = camera_shapes[camera]
        features[feature_name] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return features


def _video_shape(path: Path) -> tuple[int, int]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return int(stream.codec_context.height), int(stream.codec_context.width)


def _export_episode(dataset, record: dict, expected_shapes: dict[str, tuple[int, int]]) -> int:
    with np.load(record["trajectory"], allow_pickle=False) as trajectory:
        states = trajectory["state"].astype(np.float32)
        actions = trajectory["actions_applied"].astype(np.float32)
    if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != 7:
        raise ValueError(f"Invalid state/action shape in {record['trajectory']}")
    if not len(actions):
        raise ValueError(f"Empty rollout trajectory: {record['trajectory']}")

    decoders = {}
    containers = {}
    try:
        for camera, path in record["videos"].items():
            decoder, container = _decode_rgb(path)
            decoders[camera] = decoder
            containers[camera] = container

        for step, (state, action) in enumerate(zip(states, actions, strict=True)):
            frames = {}
            for camera, decoder in decoders.items():
                try:
                    frame = next(decoder)
                except StopIteration as error:
                    raise ValueError(f"{record['videos'][camera]} ended before action step {step}") from error
                if frame.shape[:2] != expected_shapes[camera] or frame.dtype != np.uint8:
                    raise ValueError(
                        f"Unexpected video frame in {record['videos'][camera]}: "
                        f"shape={frame.shape}, dtype={frame.dtype}"
                    )
                frames[camera] = np.moveaxis(frame, -1, 0)

            dataset.add_frame(
                {
                    "observation.state": state,
                    "observation.images.image": frames["third_person"],
                    "observation.images.image2": frames["wrist"],
                    "action": action,
                    "is_success": np.asarray([record["success"]], dtype=bool),
                    "task": record["instruction"],
                }
            )

        # RecordedEnv stores exactly one final observation after the last action.
        for camera, decoder in decoders.items():
            try:
                next(decoder)
            except StopIteration as error:
                raise ValueError(f"Missing final frame in {record['videos'][camera]}") from error
            try:
                next(decoder)
            except StopIteration:
                continue
            raise ValueError(f"Extra frames in {record['videos'][camera]}")
    finally:
        for container in containers.values():
            container.close()

    dataset.save_episode()
    return len(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/piperx_policy_rollouts_60")
    parser.add_argument("--expected-per-pair", type=int, default=10)
    parser.add_argument("--encoder-threads", type=int, default=8)
    parser.add_argument("--encoder-queue-maxsize", type=int, default=1024)
    args = parser.parse_args()
    args.rollouts = args.rollouts.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error("Output must be a new directory")
    if min(args.expected_per_pair, args.encoder_threads, args.encoder_queue_maxsize) <= 0:
        parser.error("Counts and encoder settings must be positive")

    manifest, records = _load_records(args.rollouts, args.expected_per_pair)
    camera_shapes = {camera: _video_shape(records[0]["videos"][camera]) for camera in CAMERA_FEATURES}
    for record in records:
        for camera, video_path in record["videos"].items():
            if _video_shape(video_path) != camera_shapes[camera]:
                raise ValueError(f"Video resolution differs from the dataset: {video_path}")

    try:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise SystemExit("LeRobot and PyAV are required for rollout export") from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        robot_type="piperx",
        fps=records[0]["control_hz"],
        features=_features(camera_shapes),
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
    task_successes: Counter = Counter()
    failure_counts: Counter = Counter()
    total_frames = 0
    completed = False
    try:
        for index, record in enumerate(records, start=1):
            frames = _export_episode(dataset, record, camera_shapes)
            task_counts[record["instruction"]] += 1
            task_successes[record["instruction"]] += int(record["success"])
            if not record["success"]:
                failure_counts[str(record["failure"] or "unknown")] += 1
            total_frames += frames
            print(
                json.dumps(
                    {
                        "episode": index,
                        "total": len(records),
                        "instruction": record["instruction"],
                        "seed": record["scene_seed"],
                        "success": record["success"],
                        "frames": frames,
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

    summary = {
        "kind": "piperx_policy_rollouts_lerobot",
        "repo_id": args.repo_id,
        "policy_rollouts": str(args.rollouts),
        "checkpoint": manifest["specification"]["checkpoint"],
        "episodes": len(records),
        "frames": total_frames,
        "episodes_by_task": dict(task_counts),
        "successes_by_task": dict(task_successes),
        "failures_by_reason": dict(failure_counts),
        "successes": sum(task_successes.values()),
        "failures": len(records) - sum(task_successes.values()),
        "has_is_success": True,
        "cameras": list(CAMERA_FEATURES),
        "control_hz": records[0]["control_hz"],
        "encoder_threads": args.encoder_threads,
        "elapsed_wall_seconds": time.perf_counter() - started,
    }
    write_json(args.output / "rollout_export.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
