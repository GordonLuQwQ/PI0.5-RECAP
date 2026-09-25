"""Export only the IK-labelled suffixes of corrective trajectories to LeRobot."""

import argparse
import json
from pathlib import Path

import numpy as np

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--encoder-queue-maxsize", type=int, default=1024)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Export saved successful corrections even when collection quotas were not met",
    )
    args = parser.parse_args()
    args.input, args.output = args.input.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error("Output must be a new directory")
    collection_path = args.input / "collection.json"
    summary_path = args.input / "summary.json"
    if not collection_path.is_file() or not summary_path.is_file():
        parser.error("Input is not a completed correction collection")
    collection = json.loads(collection_path.read_text())
    summary = json.loads(summary_path.read_text())
    if collection.get("kind") != "policy_prefix_ik_correction":
        parser.error("Input is not a correction collection")
    if not summary.get("target_reached") and not args.allow_partial:
        parser.error("Correction collection is incomplete")
    records = [json.loads(path.read_text()) for path in sorted(args.input.glob("episode_*.json"))]
    if not records:
        parser.error("No saved correction episodes were found")
    if not args.allow_partial and len(records) != sum(collection["quotas"]):
        parser.error("Saved episode count does not match the declared quotas")
    cameras = tuple(collection["cameras"])

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise SystemExit("LeRobot is unavailable. Run this exporter with the lerobot_pi05 environment.") from error

    with np.load(args.input / records[0]["archive"], allow_pickle=False) as sample:
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
        }
        for camera in cameras:
            height, width, channels = sample[camera].shape[1:]
            if channels != 3:
                raise ValueError(f"Expected RGB frames for {camera}")
            features[CAMERA_FEATURES[camera]] = {
                "dtype": "video",
                "shape": (channels, height, width),
                "names": ["channels", "height", "width"],
            }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        robot_type="piperx",
        fps=int(collection["control_hz"]),
        features=features,
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=args.encoder_threads,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
    )
    exported_frames = 0
    for episode_index, record in enumerate(records):
        with np.load(args.input / record["archive"], allow_pickle=False) as episode:
            expert = episode["is_expert"].astype(bool)
            indices = np.flatnonzero(expert)
            if not len(indices) or not np.array_equal(indices, np.arange(indices[0], len(expert))):
                raise ValueError(f"Episode {episode_index} does not contain one contiguous expert suffix")
            if indices[0] != record["policy_steps"] or len(indices) != record["ik_steps"]:
                raise ValueError(f"Episode {episode_index} intervention metadata is inconsistent")
            states = episode["state"]
            actions = episode["actions"]
            images = {camera: episode[camera] for camera in cameras}
            for frame_index in indices:
                frame = {
                    "observation.state": states[frame_index].astype(np.float32),
                    "action": actions[frame_index].astype(np.float32),
                    "task": record["instruction"],
                }
                for camera in cameras:
                    frame[CAMERA_FEATURES[camera]] = np.moveaxis(images[camera][frame_index], -1, 0)
                dataset.add_frame(frame)
            dataset.save_episode()
            exported_frames += len(indices)
            print(
                json.dumps(
                    {
                        "episode": episode_index + 1,
                        "total": len(records),
                        "instruction": record["instruction"],
                        "expert_frames": len(indices),
                    }
                ),
                flush=True,
            )
    dataset.finalize()
    (args.output / "correction_export.json").write_text(
        json.dumps(
            {
                "source": str(args.input),
                "repo_id": args.repo_id,
                "episodes": len(records),
                "requested_episodes": sum(collection["quotas"]),
                "target_reached": bool(summary.get("target_reached")),
                "frames": exported_frames,
                "selection": "is_expert == 1",
                "excluded_policy_prefix_frames": sum(record["policy_steps"] for record in records),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
