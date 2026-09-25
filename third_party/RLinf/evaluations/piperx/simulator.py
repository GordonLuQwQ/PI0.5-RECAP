# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the existing calibrated PiperX scene in its Genesis Python environment."""

import fcntl
import html
import logging
import sys
import time
import traceback
from dataclasses import asdict
from multiprocessing.connection import Connection

import numpy as np
from bridge import BRIDGE_VERSION, decode_packet, encode_packet, new_trace, save_trace
from common import RecordedEnv, restore_cameras
from ledger import (
    ensure_manifest,
    instruction,
    prepare_trial,
    update_summary,
    write_json,
)
from piperx_data_engine.env import StackingEnv
from piperx_data_engine.runtime import init_genesis
from placement import PlacementMonitor
from settings import (
    CAMERAS,
    CHECKPOINT,
    FAST_CHUNK_STEP,
    MAX_CONTROL_STEPS,
    OUTPUT,
    PAIRS,
    RTC_CONFIG,
    SEEDS,
    SUITE,
    TOTAL_TRIALS,
    VIDEO_FAILURE_ONLY,
    trial_plan,
)

logger = logging.getLogger(__name__)


class Simulation:
    """Execute single control steps; RTC scheduling remains in RTCEnvWorker."""

    def __init__(self, trials: list) -> None:
        self.trials, self.index = trials, -1
        self.digest = ensure_manifest()
        self.env = StackingEnv(init_genesis("gpu", seed=0), num_envs=1, cameras=True)
        self.recorded = None
        self.report = None
        self.active = False

    def reset_scene(self, seed: int) -> None:
        """Restore the same settled physics state for every task of a seed."""
        import torch

        cache = OUTPUT / "reset_states"
        cache.mkdir(exist_ok=True)
        path = cache / f"{seed}.npz"
        with (cache / f"{seed}.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.env.scene.reset()
            if path.exists():
                snapshot = self.env.scene.get_state()
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
                    self.env.last_action = values["last_action"].copy()
                self.env.scene.reset(snapshot)
            else:
                self.env.reset([seed])
                values = {"last_action": self.env.last_action.copy()}
                for index, state in enumerate(self.env.scene.get_state().solvers_state):
                    if state is not None:
                        values.update(
                            {
                                f"{index}_{name}": value.detach().cpu().numpy().copy()
                                for name, value in vars(state).items()
                                if isinstance(value, torch.Tensor)
                            }
                        )
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **values)
                temporary.replace(path)
        self.env.initial = self.env.truth()

    def dispatch(self, operation: str, arrays: dict, meta: dict):
        if operation == "reset":
            self.index += 1
            trial = self.trials[self.index]
            self.report = None
            self.recorded = None
            if trial is None:
                self.active = False
                self.reset_scene(SEEDS[0])
                restore_cameras(self.env)
                self.steps = 0
                self.task = instruction(tuple(PAIRS[0]))
                return self.env.observe(camera_names=CAMERAS), {
                    "active": False,
                    "instruction": self.task,
                    "trial": None,
                }

            self.active = True
            pair, seed = tuple(trial["pair"]), trial["seed"]
            self.directory = prepare_trial(OUTPUT, pair, seed)
            self.reset_scene(seed)
            cameras = restore_cameras(self.env)
            initial = self.env.truth()
            np.savez_compressed(self.directory / "initial_truth.npz", **initial)
            np.save(self.directory / "initial_state.npy", self.env.state())
            write_json(self.directory / "cameras.json", cameras)
            self.monitor = PlacementMonitor([pair], initial, self.env.protocol)
            self.trace, self.steps = new_trace(), 0
            self.task = instruction(pair)
            self.report = {
                "status": "running",
                "signature": self.digest,
                "pair": list(pair),
                "scene_seed": seed,
                "instruction": self.task,
                "suite": SUITE,
                "controller": "rlinf_official_rtc"
                if RTC_CONFIG["enabled"]
                else "rlinf_official_chunk",
                "checkpoint": str(CHECKPOINT),
                "rtc": RTC_CONFIG,
                "protocol": asdict(self.env.protocol),
                "max_control_steps": MAX_CONTROL_STEPS,
                "fast_chunk_step": FAST_CHUNK_STEP,
            }
            write_json(self.directory / "report.json", self.report)
            self.recorded = RecordedEnv(self.env, self.directory)
            self.started = time.perf_counter()
            logger.info(
                "Starting %d/%d %s: %s",
                self.index + 1,
                len(self.trials),
                trial["directory"],
                self.task,
            )
            return self.recorded.current_observation(), {
                "active": True,
                "instruction": self.task,
                "trial": trial["directory"],
            }

        if operation in {"step", "chunk_step"}:
            action_batch = arrays["actions"]
            if not self.active:
                horizon = 1 if operation == "step" else action_batch.shape[1]
                reply = {
                    "instruction": self.task,
                    "success": False,
                    "done": True,
                    "steps": 0,
                }
                if operation == "chunk_step":
                    reply["step_successes"] = [False] * horizon
                    reply["step_dones"] = [True] * horizon
                return self.env.observe(camera_names=CAMERAS), reply
            if operation == "step":
                if action_batch.shape != (1, 7):
                    raise ValueError(
                        "Expected one absolute PiperX action with shape (1, 7)"
                    )
                action_batch = action_batch[:, None, :]
            elif (
                action_batch.ndim != 3
                or action_batch.shape[:1] != (1,)
                or action_batch.shape[2] != 7
            ):
                raise ValueError(
                    "Expected absolute PiperX action chunk with shape (1, H, 7)"
                )
            if not np.isfinite(action_batch).all():
                raise ValueError("PiperX actions must be finite")

            successes, dones = [], []
            done = False
            success = bool(self.monitor.success[0])
            for index in range(action_batch.shape[1]):
                if not done:
                    success, done = self.execute_action(action_batch[:, index])
                successes.append(success)
                dones.append(done)

            obs = self.recorded.current_observation()
            reply = {
                "instruction": self.task,
                "success": success,
                "done": done,
                "steps": self.steps,
            }
            if operation == "chunk_step":
                reply["step_successes"] = successes
                reply["step_dones"] = dones
            return obs, reply

        if operation == "finish":
            if not self.active:
                return {}, {
                    "progress": "",
                    "remaining_batches": len(self.trials) - self.index - 1,
                }
            self.report["clipped_control_steps"] = save_trace(
                self.directory, self.trace, self.env.state()
            )
            requested = np.asarray(self.trace["actions_requested"])
            applied = np.asarray(self.trace["actions_applied"])
            joint_excess = np.abs(requested[:, :6] - applied[:, :6])
            self.report["joint_clipped_control_steps"] = int(
                np.any(joint_excess > 1e-5, axis=1).sum()
            )
            self.report["max_joint_limit_excess_rad"] = float(joint_excess.max())
            self.report["executed_control_steps"] = self.steps
            self.report["task_result"] = self.monitor.results()[0]
            self.report["status"] = "validating_videos"
            write_json(self.directory / "report.json", self.report)
            keep_video = not (
                VIDEO_FAILURE_ONLY and self.report["task_result"]["success"]
            )
            self.report["videos"] = self.recorded.finish(keep_video=keep_video)
            self.recorded = None
            self.report["wall_seconds"] = time.perf_counter() - self.started
            videos = "".join(
                f'<h2>{name}</h2><video controls src="{name}.mp4" style="max-width:720px"></video>'
                for name in self.report["videos"]
            )
            (self.directory / "watch.html").write_text(
                '<!doctype html><meta charset="utf-8">'
                f"<h1>{html.escape(self.task)}</h1><p>{self.report['controller']} · success={self.report['task_result']['success']}</p>"
                + videos
            )
            self.report["status"] = "completed"
            write_json(self.directory / "report.json", self.report)
            summary = update_summary(OUTPUT, self.digest)
            progress = f"{summary['completed_trials']}/{TOTAL_TRIALS} {self.directory.name}: success={self.report['task_result']['success']}"
            logger.info("%s", progress)
            return {}, {
                "progress": progress,
                "remaining_batches": len(self.trials) - self.index - 1,
            }
        raise ValueError(f"Unknown operation: {operation}")

    def execute_action(self, action: np.ndarray) -> tuple[bool, bool]:
        """Apply one control target and update trace and success state."""
        tick = time.perf_counter()
        if action.shape != (1, 7) or not np.isfinite(action).all():
            raise ValueError(
                "Expected finite absolute PiperX joint/closure action (1, 7)"
            )
        self.trace["state"].append(self.env.state()[0].copy())
        self.trace["actions_requested"].append(action[0].copy())
        self.trace["wall_action_seconds"].append(tick - self.started)

        def monitor():
            self.monitor.update(
                self.env.truth(contacts=True),
                self.env.state()[:, 6],
                self.env.protocol.physics_dt,
            )

        applied = self.recorded.step(action, on_physics_step=monitor)
        self.trace["actions_applied"].append(applied[0].copy())
        self.steps += 1
        success = bool(self.monitor.success[0])
        return success, success or self.steps >= MAX_CONTROL_STEPS

    def close(self) -> None:
        if self.recorded is not None:
            for writer in self.recorded.writers.values():
                writer.close()
        self.env.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    connection = Connection(int(sys.argv[1]))
    simulation = None
    try:
        while True:
            arrays, meta = decode_packet(connection.recv_bytes())
            try:
                if meta["operation"] == "initialize":
                    trials = trial_plan()
                    simulation = Simulation(trials)
                    result, reply = (
                        {},
                        {"ready": True, "remaining_batches": len(trials)},
                    )
                else:
                    result, reply = simulation.dispatch(meta["operation"], arrays, meta)
            except Exception:  # noqa: BLE001 -- send simulator failures to the owning worker
                detail = traceback.format_exc()
                logger.error("%s", detail)
                if simulation is not None and simulation.report is not None:
                    simulation.report.update(status="error", error=detail)
                    write_json(simulation.directory / "report.json", simulation.report)
                    update_summary(
                        OUTPUT, simulation.digest, status="interrupted", error=detail
                    )
                result, reply = {}, {"error": detail}
            connection.send_bytes(
                encode_packet(result, {"version": BRIDGE_VERSION, **reply})
            )
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        if simulation is not None:
            simulation.close()
        connection.close()


if __name__ == "__main__":
    main()
