"""Replay failed policy prefixes, switch to IK online, and save corrections."""

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .collect import save_archive, write_json
from .intervention import FailureWatchdogConfig, OnlineFailureWatchdog
from .protocol import PAIRS, instruction
from .runtime import init_genesis

SUPPORTED_SOURCE_FAILURES = {
    "never_reached",
    "never_closed_gripper",
    "missed_grasp",
    "lifted_then_dropped",
}


def parse_quotas(value):
    try:
        quotas = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("quotas must be comma-separated integers") from error
    if len(quotas) != len(PAIRS) or any(item < 0 for item in quotas):
        raise argparse.ArgumentTypeError(f"quotas must contain {len(PAIRS)} non-negative values in PAIRS order")
    return quotas


def load_candidates(source, quotas, *, allow_partial=False):
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    declared_pairs = tuple(tuple(pair) for pair in manifest["specification"]["pairs"])
    if declared_pairs != PAIRS:
        raise ValueError(f"Expected source pair order {PAIRS}, got {declared_pairs}")

    candidates = {pair: [] for pair in PAIRS}
    for report_path in sorted(source.glob("pair*_seed*/report.json")):
        report = json.loads(report_path.read_text())
        result = report.get("task_result", {})
        if report.get("status") != "completed" or result.get("success"):
            continue
        if result.get("failure") not in SUPPORTED_SOURCE_FAILURES:
            continue
        pair = tuple(report["pair"])
        if pair not in candidates:
            continue
        trajectory = report_path.parent / "trajectory.npz"
        initial_truth = report_path.parent / "initial_truth.npz"
        if not trajectory.is_file() or not initial_truth.is_file():
            raise FileNotFoundError(f"Incomplete source trial: {report_path.parent}")
        candidates[pair].append(
            {
                "pair": list(pair),
                "seed": int(report["scene_seed"]),
                "instruction": report["instruction"],
                "terminal_failure": result["failure"],
                "directory": str(report_path.parent),
            }
        )
    for pair, quota in zip(PAIRS, quotas, strict=True):
        candidates[pair].sort(key=lambda row: row["seed"])
        if len(candidates[pair]) < quota and not allow_partial:
            raise ValueError(
                f"{instruction(pair)} needs {quota} episodes but only {len(candidates[pair])} supported failures exist"
            )
    return manifest, candidates


def restore_snapshot(env, source, trial_directory, seed):
    """Restore the persisted Genesis reset state used by the source rollout."""
    import torch

    path = source / "reset_states" / f"{seed}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing persisted reset state: {path}")
    env.scene.reset()
    snapshot = env.scene.get_state()
    with np.load(path, allow_pickle=False) as values:
        for index, state in enumerate(snapshot.solvers_state):
            if state is None:
                continue
            for name, value in vars(state).items():
                if isinstance(value, torch.Tensor):
                    value.copy_(
                        torch.as_tensor(
                            values[f"{index}_{name}"],
                            device=value.device,
                            dtype=value.dtype,
                        )
                    )
        env.last_action = values["last_action"].copy()
    env.scene.reset(snapshot)
    env.initial = env.truth()

    with np.load(Path(trial_directory) / "initial_truth.npz", allow_pickle=False) as saved:
        for key in ("positions", "quaternions", "tcp", "tcp_quaternion"):
            np.testing.assert_allclose(env.initial[key], saved[key], atol=2e-6, rtol=0)
    np.testing.assert_allclose(env.state(), np.load(Path(trial_directory) / "initial_state.npy"), atol=2e-6, rtol=0)
    return env.initial


def restore_cameras(env, calibration_path, camera_names):
    calibration = json.loads(calibration_path.read_text())
    for name in camera_names:
        saved, camera = calibration[name], env.cameras[name]
        if list(camera.res) != saved["resolution_wh"] or camera.fov != saved["fov_degrees"]:
            raise ValueError(f"Camera geometry no longer matches calibration: {name}")
        np.testing.assert_allclose(camera.intrinsics, saved["intrinsics"], atol=1e-6, rtol=0)
        if saved["attached_link"] is None:
            transform = np.asarray(saved["initial_camera_to_world_opengl"]).reshape(-1, 4, 4)[0]
            camera.set_pose(transform=transform)
        else:
            np.testing.assert_allclose(
                env.camera_mounts[name]["camera_to_link_opengl"],
                saved["camera_to_link_opengl"],
                atol=1e-8,
                rtol=0,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Completed six-task policy rollout suite containing manifest.json"
    )
    parser.add_argument(
        "--output", type=Path, help="New raw correction dataset directory (required unless --plan-only)"
    )
    parser.add_argument(
        "--quotas",
        type=parse_quotas,
        default=parse_quotas("6,6,9,7,3,4"),
        help="Saved corrections for PAIRS order 01,02,10,12,20,21",
    )
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--cameras", nargs="+", choices=("third_person", "wrist"))
    parser.add_argument("--calibration", type=Path, help="Camera calibration at its current downloaded location")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Keep available successful corrections if candidates are exhausted before quotas are met",
    )
    parser.add_argument(
        "--plan-only", action="store_true", help="Validate quotas and print candidate episodes without starting Genesis"
    )
    args = parser.parse_args()
    args.source = args.source.resolve()
    manifest, candidates = load_candidates(args.source, args.quotas, allow_partial=args.allow_partial)
    camera_names = tuple(args.cameras or manifest["specification"]["cameras"])
    calibration = (args.calibration or Path(manifest["specification"]["calibration"])).expanduser().resolve()
    plan = {
        "source": str(args.source),
        "calibration": str(calibration),
        "pair_order": [list(pair) for pair in PAIRS],
        "tasks": [instruction(pair) for pair in PAIRS],
        "quotas": {instruction(pair): quota for pair, quota in zip(PAIRS, args.quotas, strict=True)},
        "eligible_source_failures": sorted(SUPPORTED_SOURCE_FAILURES),
        "candidate_counts": {instruction(pair): len(candidates[pair]) for pair in PAIRS},
        "candidates": {instruction(pair): rows for pair, rows in candidates.items()},
        "watchdog": asdict(FailureWatchdogConfig(window_seconds=args.window_seconds)),
        "training_rule": "Keep the full trace for audit; train only frames where is_expert == 1.",
        "evaluation_seed_note": (
            "These source trials use evaluation seeds. If corrections enter training, final policy "
            "evaluation must use a new untouched seed list."
        ),
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    if args.output is None:
        parser.error("--output is required unless --plan-only is used")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("Output must be a new directory; corrections are never appended or overwritten")
    if args.window_seconds <= 0:
        parser.error("--window-seconds must be positive")
    if len(set(camera_names)) != len(camera_names):
        parser.error("Camera names must not contain duplicates")
    if not calibration.is_file():
        parser.error(f"Missing source camera calibration: {calibration}")

    args.output.mkdir(parents=True)
    write_json(args.output / "selection.json", plan)
    write_json(
        args.output / "collection.json",
        {
            "kind": "policy_prefix_ik_correction",
            "source_rollouts": str(args.source),
            "pairs": [list(pair) for pair in PAIRS],
            "quotas": list(args.quotas),
            "cameras": list(camera_names),
            "control_hz": 20,
            "policy_prefix_label": "is_expert=0",
            "ik_suffix_label": "is_expert=1",
            "trainable_frames": "is_expert == 1 only",
            "source_seed_status": "Previously used evaluation seeds; exclude them from future final evaluation.",
        },
    )

    gs = init_genesis(args.backend)
    from .env import DEFAULT_URDF, TCP_LOCAL, StackingEnv
    from .planning import PlannerConfig
    from .success import SuccessMonitor
    from .teacher import TeacherConfig, demonstration

    env = StackingEnv(gs, num_envs=1, cameras=True)
    restore_cameras(env, calibration, camera_names)
    write_json(
        args.output / "cameras.json",
        {name: value for name, value in env.camera_metadata().items() if name in camera_names},
    )
    watchdog_config = FailureWatchdogConfig(window_seconds=args.window_seconds)
    teacher_config = TeacherConfig()
    write_json(
        args.output / "provenance.json",
        {
            "watchdog_config": asdict(watchdog_config),
            "teacher_config": asdict(teacher_config),
            "planner_config": asdict(PlannerConfig()),
            "source_manifest": manifest,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )

    counts, attempts = Counter(), Counter()
    episode_index = 0
    started = time.perf_counter()

    def checkpoint_summary():
        summary = {
            "kind": "policy_prefix_ik_correction",
            "successes_saved": {instruction(pair): counts[pair] for pair in PAIRS},
            "targets": {instruction(pair): quota for pair, quota in zip(PAIRS, args.quotas, strict=True)},
            "attempts": {instruction(pair): attempts[pair] for pair in PAIRS},
            "target_reached": all(counts[pair] == quota for pair, quota in zip(PAIRS, args.quotas, strict=True)),
            "elapsed_wall_seconds": time.perf_counter() - started,
            "trainable_frames": "is_expert == 1 only",
        }
        write_json(args.output / "summary.json", summary)
        print(json.dumps({"progress": summary}), flush=True)
        return summary

    try:
        checkpoint_summary()
        for pair, quota in zip(PAIRS, args.quotas, strict=True):
            if quota == 0:
                continue
            for candidate in candidates[pair]:
                if counts[pair] >= quota:
                    break
                attempts[pair] += 1
                directory = Path(candidate["directory"])
                initial = restore_snapshot(env, args.source, directory, candidate["seed"])
                monitor = SuccessMonitor([pair], initial, env.protocol)
                watchdog = OnlineFailureWatchdog(pair, initial, watchdog_config)
                frames, actions, phases, expert_flags = [], [], [], []

                def execute(
                    action,
                    phase,
                    expert,
                    detect_failure,
                    _frames=frames,
                    _actions=actions,
                    _phases=phases,
                    _expert_flags=expert_flags,
                    _monitor=monitor,
                    _watchdog=watchdog,
                ):
                    _frames.append(env.observe(camera_names=camera_names))  # o_t before a_t
                    _phases.append(phase)
                    _expert_flags.append(expert)

                    def update_monitors():
                        truth = env.truth(contacts=True)
                        closure = env.state()[:, 6]
                        _monitor.update(truth, closure, env.protocol.physics_dt)
                        if detect_failure and not _monitor.success[0]:
                            _watchdog.update(truth, closure, env.protocol.physics_dt)

                    applied = env.step(np.asarray(action, dtype=np.float32), on_physics_step=update_monitors)
                    _actions.append(applied[0].copy())

                with np.load(directory / "trajectory.npz", allow_pickle=False) as trajectory:
                    source_actions = trajectory["actions_applied"].copy()
                for action in source_actions:
                    execute(action[None], "policy_replay", False, True)
                    if watchdog.trigger is not None or monitor.success[0]:
                        break

                policy_steps = len(actions)
                teacher_failures, teacher_events = {}, []

                def on_failure(indices, reason, detail, _teacher_failures=teacher_failures):
                    _teacher_failures.update({int(index): {"failure": reason, "detail": detail} for index in indices})

                recovery_mode = None
                if watchdog.trigger is not None and not monitor.success[0]:
                    truth = env.truth(contacts=True)
                    source = pair[0]
                    current_lift = float(truth["positions"][0, source, 2] - initial["positions"][0, source, 2])
                    tcp_distance = float(np.linalg.norm(truth["positions"][0, source] - truth["tcp"][0]))
                    held_by_both_fingers = bool(np.all(truth["finger_contacts"][0, source]))
                    resume_grasped = bool(
                        watchdog.trigger["lifted_at_seconds"] is not None
                        and current_lift >= teacher_config.lift_min_height
                        and tcp_distance <= teacher_config.dropped_distance
                        and held_by_both_fingers
                    )
                    if resume_grasped and tcp_distance <= teacher_config.secure_grasp_distance:
                        recovery_mode = "continue_grasped"
                    elif resume_grasped:
                        recovery_mode = "lower_and_regrasp"
                    else:
                        recovery_mode = "regrasp"
                    for action, phase in demonstration(
                        env,
                        [pair],
                        on_failure=on_failure,
                        events=teacher_events,
                        config=teacher_config,
                        start_with_grasped_source=resume_grasped,
                    ):
                        execute(action, "ik/" + phase, True, False)

                result = monitor.results()[0]
                saved = bool(
                    watchdog.trigger is not None
                    and not teacher_failures
                    and result["success"]
                    and len(actions) > policy_steps
                )
                record = {
                    **result,
                    "saved": saved,
                    "pair": list(pair),
                    "instruction": instruction(pair),
                    "scene_seed": candidate["seed"],
                    "source_trial": str(directory),
                    "source_terminal_failure": candidate["terminal_failure"],
                    "intervention": watchdog.trigger,
                    "recovery_mode": recovery_mode,
                    "policy_steps": policy_steps,
                    "ik_steps": len(actions) - policy_steps,
                    "frames": len(actions),
                    "teacher_failure": teacher_failures.get(0),
                    "teacher_events": teacher_events,
                    "initial_positions": initial["positions"][0].tolist(),
                    "initial_quaternions": initial["quaternions"][0].tolist(),
                    "initial_tcp": initial["tcp"][0].tolist(),
                    "protocol": asdict(env.protocol),
                    "tcp_local": TCP_LOCAL,
                    "cameras": list(camera_names),
                    "urdf": str(DEFAULT_URDF),
                }
                if saved:
                    observations = {key: np.stack([frame[key][0] for frame in frames]) for key in frames[0]}
                    name = f"episode_{episode_index:06d}"
                    record["archive"] = name + ".npz"
                    save_archive(
                        args.output / record["archive"],
                        {
                            **observations,
                            "actions": np.asarray(actions, dtype=np.float32),
                            "phases": np.asarray(phases),
                            "is_expert": np.asarray(expert_flags, dtype=np.uint8),
                            "final_state": env.state()[0],
                        },
                    )
                    write_json(args.output / (name + ".json"), record)
                    counts[pair] += 1
                    episode_index += 1
                with (args.output / "attempts.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(
                    json.dumps(
                        {
                            "instruction": instruction(pair),
                            "scene_seed": candidate["seed"],
                            "source_failure": candidate["terminal_failure"],
                            "intervention": None if watchdog.trigger is None else watchdog.trigger["reason"],
                            "policy_steps": policy_steps,
                            "ik_steps": len(actions) - policy_steps,
                            "success": result["success"],
                            "saved": saved,
                        }
                    ),
                    flush=True,
                )
                del frames, actions, phases, expert_flags
                checkpoint_summary()
        summary = checkpoint_summary()
        if not summary["target_reached"] and not args.allow_partial:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
