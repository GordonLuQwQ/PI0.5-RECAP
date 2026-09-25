"""Collect successful teacher demonstrations as RGB, state, and action NPZ archives."""

import argparse
import json
import time
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .env import DEFAULT_URDF
from .protocol import (
    PAIRS,
    TASK_SETS,
    Protocol,
    instruction,
    manifest,
    training_pairs,
    write_manifest,
)
from .runtime import init_genesis


def save_archive(path, arrays):
    """Lossless NPZ, using fast compression to keep camera recording affordable."""
    temporary = path.with_suffix(".npz.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for key, value in arrays.items():
            with archive.open(key + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.ascontiguousarray(value), allow_pickle=False)
    temporary.replace(path)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--num-envs", type=int, default=5)
    parser.add_argument("--episodes-per-pair", type=int, default=1)
    parser.add_argument("--max-attempts-per-pair", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1000, help="First seed within the declared training pool")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task-set",
        choices=TASK_SETS,
        default="benchmark",
        help="benchmark: five training pairs with one held out; all: all six ordered pairs",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=[instruction(p) for p in PAIRS],
        help="Optional subset of instructions from --task-set; default collects the whole set",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=("third_person", "wrist"),
        default=["third_person", "wrist"],
        help="RGB views to record; camera poses are taken unchanged from env.py",
    )
    args = parser.parse_args()
    args.urdf = args.urdf.expanduser().resolve()
    permitted_pairs = training_pairs(args.task_set)
    tasks = args.tasks or [instruction(p) for p in permitted_pairs]
    if len(set(tasks)) != len(tasks) or len(set(args.cameras)) != len(args.cameras):
        parser.error("Tasks and cameras must not contain duplicates")
    if not set(tasks).issubset(instruction(p) for p in permitted_pairs):
        parser.error("This task is held out in benchmark mode; use --task-set all for six-pair training")
    pairs = tuple(p for task in tasks for p in permitted_pairs if instruction(p) == task)
    camera_names = args.cameras
    if min(args.num_envs, args.episodes_per_pair, args.max_attempts_per_pair) < 1:
        parser.error("Counts must be positive")
    if args.episodes_per_pair > args.max_attempts_per_pair or args.max_attempts_per_pair > 1000:
        parser.error("Require episodes-per-pair <= max-attempts-per-pair <= 1000 (declared training seed pool)")
    seeds = list(range(args.seed_start, args.seed_start + args.max_attempts_per_pair))
    if not set(seeds).issubset(manifest()["training_scene_seeds"]):
        parser.error("All attempted seeds must be in the declared training pool 1000..1999")
    if args.output.exists():
        parser.error("Output must be a new directory; previous data will not be overwritten")
    args.output.mkdir(parents=True)
    protocol = Protocol()
    specification = {
        "task_set": args.task_set,
        "tasks": tasks,
        "pairs": [list(p) for p in pairs],
        "episodes_per_pair": args.episodes_per_pair,
        "cameras": camera_names,
        "backend": args.backend,
        "num_envs": args.num_envs,
        "scene_seed_pool": seeds,
        "scene_sampling": {
            "name": "uniform_xy_yaw_with_separation",
            "objects": "red cube, red cylinder, blue cube",
            "center_x_range_m": list(protocol.x_range),
            "center_y_range_m": list(protocol.y_range),
            "center_z_m": protocol.table_top + protocol.object_height / 2 + 0.001,
            "yaw_range_rad": [-float(np.pi), float(np.pi)],
            "roll_pitch_rad": [0.0, 0.0],
            "minimum_center_distance_m": protocol.min_center_distance,
        },
        "urdf": str(args.urdf),
        "default_training_cameras": camera_names,
        "note": (
            "Six-pair training mode: no task pair is reserved for unseen-pair generalization."
            if args.task_set == "all"
            else "Selected training tasks from the five-pair benchmark; the sixth pair remains held out."
        ),
    }
    counts, attempts = Counter(), Counter()
    used_seeds = {p: set() for p in pairs}
    episode_index = 0
    write_manifest(args.output / "protocol.json", task_set=args.task_set)
    write_json(args.output / "collection.json", specification)
    gs = init_genesis(args.backend)
    from .env import TCP_LOCAL, StackingEnv
    from .planning import PlannerConfig
    from .success import SuccessMonitor
    from .teacher import TeacherConfig, demonstration

    env = StackingEnv(gs, num_envs=args.num_envs, cameras=True, urdf=args.urdf)
    start_time = time.perf_counter()
    provenance = {
        "teacher_config": asdict(TeacherConfig()),
        "planner_config": asdict(PlannerConfig()),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    write_json(args.output / "provenance.json", provenance)

    def write_summary():
        summary = {
            "kind": "scripted_teacher_collection",
            "has_images": True,
            "cameras": camera_names,
            "episodes_per_pair": args.episodes_per_pair,
            "scene_sampling": specification["scene_sampling"],
            "successes_saved": {instruction(p): counts[p] for p in pairs},
            "attempts": {instruction(p): attempts[p] for p in pairs},
            "balanced_target_reached": all(counts[p] == args.episodes_per_pair for p in pairs),
            "elapsed_wall_seconds": time.perf_counter() - start_time,
            "note": "Teacher collection results are not learned-policy evaluation results.",
        }
        write_json(args.output / "summary.json", summary)
        print(json.dumps({"progress": summary}), flush=True)
        return summary

    try:
        write_summary()
        while any(counts[p] < args.episodes_per_pair and attempts[p] < args.max_attempts_per_pair for p in pairs):
            batch, batch_seeds, padding = [], [], []
            for _ in range(args.num_envs):
                pending = [
                    p
                    for p in pairs
                    if counts[p] + batch.count(p) < args.episodes_per_pair and attempts[p] < args.max_attempts_per_pair
                ]
                if not pending:
                    # Fill fixed-size Genesis batches; padding is logged but is never training data.
                    batch.append(pairs[0])
                    batch_seeds.append(seeds[0])
                    padding.append(True)
                    continue
                pair = min(pending, key=lambda p: (counts[p] + batch.count(p), attempts[p], pairs.index(p)))
                seed = next(seed for seed in seeds if seed not in used_seeds[pair])
                used_seeds[pair].add(seed)
                batch.append(pair)
                batch_seeds.append(seed)
                padding.append(False)
                attempts[pair] += 1
            print(
                json.dumps(
                    {
                        "batch_start": {
                            "tasks": [instruction(p) for p in batch],
                            "seeds": batch_seeds,
                            "padding": padding,
                        }
                    }
                ),
                flush=True,
            )
            env.reset(batch_seeds)
            if not (args.output / "cameras.json").exists():
                write_json(
                    args.output / "cameras.json",
                    {name: metadata for name, metadata in env.camera_metadata().items() if name in camera_names},
                )
            monitor = SuccessMonitor(batch, env.initial, env.protocol)
            teacher_failures, events = {}, []

            def on_failure(indices, reason, detail, failures=teacher_failures):
                failures.update({int(i): {"failure": reason, "detail": detail} for i in indices})

            frames, actions, phases = [], [], []
            for action, phase in demonstration(env, batch, on_failure=on_failure, events=events):
                frames.append(env.observe(camera_names=camera_names))  # o_t BEFORE a_t.
                phases.append(phase)

                def update_monitor(current_action=action, current_monitor=monitor):
                    current_monitor.update(
                        env.truth(contacts=True),
                        current_action[:, 6],
                        env.protocol.physics_dt,
                    )

                applied = env.step(
                    action,
                    on_physics_step=update_monitor,
                )
                actions.append(applied)
                if len(actions) % 100 == 0:
                    print(json.dumps({"batch_step": len(actions), "phase": phase}), flush=True)
            observations = {key: np.stack([f[key] for f in frames]) for key in frames[0]} if frames else {}
            del frames
            actions = np.stack(actions) if actions else np.empty((0, env.num_envs, 7), dtype=np.float32)
            final_state = env.state()
            for i, result in enumerate(monitor.results()):
                if i in teacher_failures:
                    result.update(success=False, **teacher_failures[i])
                saved = result["success"] and not padding[i]
                record = {
                    **result,
                    "pair": batch[i],
                    "instruction": instruction(batch[i]),
                    "scene_seed": batch_seeds[i],
                    "env_index": i,
                    "num_envs": args.num_envs,
                    "backend": args.backend,
                    "padding": padding[i],
                    "saved": saved,
                    "frames": len(actions),
                    "initial_positions": env.initial["positions"][i].tolist(),
                    "initial_quaternions": env.initial["quaternions"][i].tolist(),
                    "initial_tcp": env.initial["tcp"][i].tolist(),
                    "scene_sampling": specification["scene_sampling"],
                    "protocol": asdict(env.protocol),
                    "tcp_local": TCP_LOCAL,
                    "has_images": True,
                    "cameras": camera_names,
                    "urdf": str(env.urdf),
                    "teacher_events": [event for event in events if event["env"] == i],
                }
                if saved:
                    name = f"episode_{episode_index:06d}"
                    record["archive"] = name + ".npz"
                    save_archive(
                        args.output / record["archive"],
                        {
                            **{key: values[:, i] for key, values in observations.items()},
                            "actions": actions[:, i],
                            "phases": np.asarray(phases),
                            "final_state": final_state[i],
                        },
                    )
                    write_json(args.output / (name + ".json"), record)
                    counts[batch[i]] += 1
                    episode_index += 1
                with (args.output / "attempts.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(
                    json.dumps(
                        {k: record[k] for k in ("instruction", "scene_seed", "success", "failure", "padding", "saved")}
                    ),
                    flush=True,
                )
            del observations, actions
            write_summary()
        summary = write_summary()
        if not summary["balanced_target_reached"]:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
