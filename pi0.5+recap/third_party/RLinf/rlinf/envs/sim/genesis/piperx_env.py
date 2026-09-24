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

"""Expose independent PiperX Genesis scenes as one batched RLinf environment."""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.utils.logging import get_logger


class PiperXEnv:
    """Bridge one batched policy call to independent Genesis scene processes."""

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ) -> None:
        del worker_info
        if total_num_processes != 1 or seed_offset != 0:
            raise ValueError("PiperX evaluation requires one RLinf environment worker")
        if num_envs != int(cfg.piperx.get("parallel_envs", 1)):
            raise ValueError("total_num_envs and piperx.parallel_envs must match")
        if cfg.auto_reset or not cfg.is_eval:
            raise ValueError("PiperX supports evaluation with auto_reset=False")
        from evaluations.piperx.bridge import decode_packet, encode_packet

        self._decode_packet, self._encode_packet = decode_packet, encode_packet
        self.cfg = cfg
        self.num_envs = num_envs
        self.is_start = True
        self._connections: list[Connection] = []
        self._processes: list[subprocess.Popen] = []
        self._executor = ThreadPoolExecutor(max_workers=num_envs)
        self._closed = False
        self._done = np.zeros(num_envs, dtype=bool)
        self._active = np.ones(num_envs, dtype=bool)
        self._success = np.zeros(num_envs, dtype=bool)
        self._steps = np.zeros(num_envs, dtype=np.int64)
        self._last_arrays: list[dict[str, np.ndarray] | None] = [None] * num_envs
        self._instructions = [""] * num_envs
        self._last_step = None
        self.logger = get_logger()

        output = Path(cfg.piperx.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        config_path = output / "env_config.json"
        config_path.write_text(
            json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)
        )

        try:
            for slot in range(num_envs):
                self._start_simulator(slot, config_path, output)
            replies = self._rpc_all("initialize")
            remaining = {reply[1]["remaining_batches"] for reply in replies.values()}
            if remaining != {cfg.rollout_epoch}:
                raise ValueError(
                    "Pending batch count changed; relaunch evaluations/piperx/run.sh"
                )
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _start_simulator(self, slot: int, config_path: Path, output: Path) -> None:
        parent, child = socket.socketpair()
        child_env = os.environ.copy()
        child_env.pop("VIRTUAL_ENV", None)
        child_env.pop("LD_LIBRARY_PATH", None)
        child_env["LD_PRELOAD"] = self.cfg.piperx.libstdcpp_path
        child_env["PYOPENGL_PLATFORM"] = self.cfg.piperx.opengl_platform
        child_env["PIPERX_EVAL_CONFIG"] = str(config_path)
        child_env["PIPERX_WORKER_SLOT"] = str(slot)
        child_env["PIPERX_WORKER_COUNT"] = str(self.num_envs)
        child_env["PYTHONPATH"] = str(self.cfg.piperx.project_path)
        log_name = "genesis.log" if self.num_envs == 1 else f"genesis_slot{slot}.log"
        try:
            with (output / log_name).open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [
                        self.cfg.piperx.python_path,
                        self.cfg.piperx.bridge_script,
                        str(child.fileno()),
                    ],
                    cwd=self.cfg.piperx.project_path,
                    env=child_env,
                    pass_fds=(child.fileno(),),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            self._processes.append(process)
            self._connections.append(Connection(parent.detach()))
        finally:
            parent.close()
            child.close()

    def _rpc_one(
        self, slot: int, operation: str, arrays: dict | None = None
    ) -> tuple[dict, dict]:
        from evaluations.piperx.bridge import BRIDGE_VERSION

        connection = self._connections[slot]
        connection.send_bytes(
            self._encode_packet(
                arrays or {}, {"version": BRIDGE_VERSION, "operation": operation}
            )
        )
        timeout = 1200 if operation == "initialize" else 180
        if not connection.poll(timeout):
            log_name = (
                "genesis.log"
                if self.num_envs == 1
                else f"genesis_slot{slot}.log"
            )
            raise TimeoutError(
                f"PiperX slot {slot} {operation} timed out; see "
                f"{self.cfg.piperx.output_dir}/{log_name}"
            )
        result, reply = self._decode_packet(connection.recv_bytes())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return result, reply

    def _rpc_all(
        self,
        operation: str,
        arrays_by_slot: dict[int, dict] | None = None,
        slots: list[int] | None = None,
    ) -> dict[int, tuple[dict, dict]]:
        selected = list(range(self.num_envs)) if slots is None else slots
        arrays_by_slot = arrays_by_slot or {}
        futures = {
            slot: self._executor.submit(
                self._rpc_one, slot, operation, arrays_by_slot.get(slot)
            )
            for slot in selected
        }
        try:
            return {slot: future.result() for slot, future in futures.items()}
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _merge_arrays(arrays_by_slot: list[dict[str, np.ndarray]]) -> dict:
        keys = arrays_by_slot[0].keys()
        return {
            key: np.concatenate([arrays[key] for arrays in arrays_by_slot], axis=0)
            for key in keys
        }

    @staticmethod
    def observation(arrays: dict, instructions: list[str] | str) -> dict:
        """Return batched CPU RGB images, measured states and instructions."""
        from evaluations.piperx.bridge import validate_observation

        validate_observation(arrays, ("third_person", "wrist"))
        if isinstance(instructions, str):
            instructions = [instructions]
        if len(instructions) != arrays["state"].shape[0]:
            raise ValueError("Instruction count must match the observation batch")
        return {
            "main_images": torch.from_numpy(arrays["third_person"]),
            "wrist_images": torch.from_numpy(arrays["wrist"]),
            "states": torch.from_numpy(arrays["state"]),
            "task_descriptions": instructions,
        }

    def _batched_observation(self) -> dict:
        arrays = self._merge_arrays(
            [arrays for arrays in self._last_arrays if arrays is not None]
        )
        return self.observation(arrays, self._instructions)

    def reset(self) -> tuple[dict, dict]:
        """Start the next pending batch, one trial per Genesis process."""
        replies = self._rpc_all("reset")
        for slot, (arrays, reply) in replies.items():
            self._last_arrays[slot] = arrays
            self._instructions[slot] = reply["instruction"]
            self._active[slot] = bool(reply["active"])
            self._done[slot] = not self._active[slot]
            self._success[slot] = False
            self._steps[slot] = 0
            if reply["active"]:
                self.logger.info(
                    "PiperX slot %d %s: %s",
                    slot,
                    reply["trial"],
                    reply["instruction"],
                )
        self._last_step = None
        return self._batched_observation(), {}

    def _episode_info(self) -> dict:
        return {
            "episode": {
                "success_once": torch.from_numpy(self._success.astype(np.float32)),
                "return": torch.from_numpy(self._success.astype(np.float32)),
                "episode_len": torch.from_numpy(self._steps.astype(np.float32)),
            }
        }

    def step(self, actions: torch.Tensor | np.ndarray) -> tuple:
        """Execute one absolute joint target in every unfinished scene."""
        value = (
            actions.detach().cpu().numpy()
            if torch.is_tensor(actions)
            else np.asarray(actions)
        ).astype(np.float32)
        if value.shape != (self.num_envs, 7):
            raise ValueError(
                f"Expected PiperX actions ({self.num_envs}, 7), got {value.shape}"
            )
        slots = np.flatnonzero(~self._done).tolist()
        if slots:
            replies = self._rpc_all(
                "step",
                {slot: {"actions": value[slot : slot + 1]} for slot in slots},
                slots,
            )
            for slot, (arrays, reply) in replies.items():
                self._last_arrays[slot] = arrays
                self._instructions[slot] = reply["instruction"]
                self._success[slot] = bool(reply["success"])
                self._done[slot] = bool(reply["done"])
                self._steps[slot] = reply["steps"]
        info = self._episode_info() if self._done.any() else {}
        self._last_step = (
            self._batched_observation(),
            torch.from_numpy(self._success.astype(np.float32)),
            torch.from_numpy(self._success.copy()),
            torch.from_numpy(self._done & ~self._success),
            info,
        )
        return self._last_step

    def chunk_step(self, actions: torch.Tensor | np.ndarray) -> tuple:
        """Execute an action chunk with one bridge request per active scene."""
        if self.cfg.piperx.get("fast_chunk_step", False):
            return self._fast_chunk_step(actions)
        rows = [self.step(actions[:, index]) for index in range(actions.shape[1])]
        return (
            [row[0] for row in rows],
            *(torch.stack([row[index] for row in rows], dim=1) for index in (1, 2, 3)),
            [row[4] for row in rows],
        )

    def _fast_chunk_step(self, actions: torch.Tensor | np.ndarray) -> tuple:
        """Execute each scene's chunk concurrently across simulator processes."""
        value = (
            actions.detach().cpu().numpy()
            if torch.is_tensor(actions)
            else np.asarray(actions)
        ).astype(np.float32)
        if value.ndim != 3 or value.shape[0] != self.num_envs or value.shape[2] != 7:
            raise ValueError(
                f"Expected PiperX chunks ({self.num_envs}, H, 7), got {value.shape}"
            )
        horizon = value.shape[1]
        step_successes = np.broadcast_to(
            self._success[:, None], (self.num_envs, horizon)
        ).copy()
        step_dones = np.broadcast_to(
            self._done[:, None], (self.num_envs, horizon)
        ).copy()
        slots = np.flatnonzero(~self._done).tolist()
        if slots:
            replies = self._rpc_all(
                "chunk_step",
                {slot: {"actions": value[slot : slot + 1]} for slot in slots},
                slots,
            )
            for slot, (arrays, reply) in replies.items():
                self._last_arrays[slot] = arrays
                self._instructions[slot] = reply["instruction"]
                self._success[slot] = bool(reply["success"])
                self._done[slot] = bool(reply["done"])
                self._steps[slot] = reply["steps"]
                step_successes[slot] = reply["step_successes"]
                step_dones[slot] = reply["step_dones"]
                if reply["steps"] % 100 == 0:
                    self.logger.info(
                        "PiperX slot %d step %d: %s",
                        slot,
                        reply["steps"],
                        reply["instruction"],
                    )

        observation = self._batched_observation()
        rows = []
        for index in range(horizon):
            successes = step_successes[:, index]
            dones = step_dones[:, index]
            rows.append(
                (
                    observation,
                    torch.from_numpy(successes.astype(np.float32)),
                    torch.from_numpy(successes.copy()),
                    torch.from_numpy(dones & ~successes),
                    self._episode_info() if dones.any() else {},
                )
            )
        self._last_step = rows[-1]
        return (
            [row[0] for row in rows],
            *(torch.stack([row[index] for row in rows], dim=1) for index in (1, 2, 3)),
            [row[4] for row in rows],
        )

    def update_reset_state_ids(self) -> None:
        """Finalize every scene in the current batch before the next reset."""
        replies = self._rpc_all("finish")
        remaining = set()
        for _, reply in replies.values():
            if reply["progress"]:
                self.logger.info("%s", reply["progress"])
            remaining.add(reply["remaining_batches"])
        if len(remaining) != 1:
            raise RuntimeError("PiperX simulator slots lost batch alignment")
        if remaining == {0}:
            self.close()

    def close(self) -> None:
        """Release all bridge connections and simulator child processes."""
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            connection.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
        for process in self._processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
