"""Convert successful NPZ demonstrations into a LeRobot v3 video dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

CAMERA_FEATURES = {
    "third_person": "observation.images.image",
    "wrist": "observation.images.image2",
}
ACTION_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_closure",
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=tuple(CAMERA_FEATURES),
        help="Views to export; defaults to every view recorded during collection",
    )
    parser.add_argument("--encoder-threads", type=int, default=4)
    parser.add_argument("--encoder-queue-maxsize", type=int, default=1024)
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("Output must be a new directory")

    collection = json.loads((source / "collection.json").read_text())
    records = [json.loads(path.read_text()) for path in sorted(source.glob("episode_*.json"))]
    if not records:
        parser.error("Input contains no successful episodes")
    cameras = tuple(args.cameras or collection["cameras"])
    fps = int(records[0]["protocol"]["control_hz"])

    try:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise SystemExit("LeRobot is required for export. Install this package in a LeRobot environment.") from error

    with np.load(source / records[0]["archive"], allow_pickle=False) as episode:
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ACTION_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ACTION_NAMES,
            },
        }
        for camera in cameras:
            height, width, channels = episode[camera].shape[1:]
            features[CAMERA_FEATURES[camera]] = {
                "dtype": "video",
                "shape": (height, width, channels),
                "names": ("height", "width", "channels"),
            }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        robot_type="piperx",
        fps=fps,
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

    pair_counts: Counter[str] = Counter()
    total_frames = 0
    metadata_rows = []
    for episode_index, record in enumerate(records):
        with np.load(source / record["archive"], allow_pickle=False) as episode:
            states = episode["state"].astype(np.float32)
            actions = episode["actions"].astype(np.float32)
            images = {camera: episode[camera] for camera in cameras}
            for frame_index, (state, action) in enumerate(zip(states, actions, strict=True)):
                frame = {
                    "observation.state": state,
                    "action": action,
                    "task": record["instruction"],
                }
                for camera in cameras:
                    frame[CAMERA_FEATURES[camera]] = images[camera][frame_index]
                dataset.add_frame(frame)
        dataset.save_episode()
        pair_counts[record["instruction"]] += 1
        total_frames += len(actions)
        metadata_rows.append(record)
        print(
            json.dumps(
                {
                    "episode": episode_index + 1,
                    "total": len(records),
                    "instruction": record["instruction"],
                    "frames": len(actions),
                }
            ),
            flush=True,
        )

    dataset.finalize()
    metadata_dir = output / "collection_metadata"
    metadata_dir.mkdir()
    for filename in (
        "collection.json",
        "protocol.json",
        "provenance.json",
        "summary.json",
        "cameras.json",
        "attempts.jsonl",
    ):
        path = source / filename
        if path.is_file():
            shutil.copy2(path, metadata_dir / filename)
    with (metadata_dir / "episodes.jsonl").open("w") as stream:
        for record in metadata_rows:
            stream.write(json.dumps(record) + "\n")

    summary = {
        "source": str(source),
        "repo_id": args.repo_id,
        "episodes": len(records),
        "frames": total_frames,
        "fps": fps,
        "cameras": list(cameras),
        "episodes_per_instruction": dict(pair_counts),
        "action_semantics": "absolute q1..q6 joint targets plus gripper closure",
    }
    _write_json(output / "export.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
