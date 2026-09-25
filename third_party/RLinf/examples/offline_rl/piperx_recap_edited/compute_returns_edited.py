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

"""Write RECAP reward and return sidecars without changing LeRobot data files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from reward_edited import ReCapRewardSpecEdited, discounted_returns, episode_rewards


def _load_tasks(root: Path) -> dict[int, str]:
    """Read task names from LeRobot v3 parquet or the older JSONL file."""
    parquet = root / "meta" / "tasks.parquet"
    if parquet.is_file():
        rows = pq.read_table(parquet, columns=["task_index", "task"]).to_pylist()
        return {int(row["task_index"]): str(row["task"]) for row in rows}

    jsonl = root / "meta" / "tasks.jsonl"
    if jsonl.is_file():
        rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
        return {int(row["task_index"]): str(row["task"]) for row in rows}
    return {}


def _read_frames(root: Path, *, require_success: bool) -> dict[str, np.ndarray]:
    """Read only the small scalar columns needed to label each frame."""
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found below {root / 'data'}")

    tables = []
    for file in files:
        available = set(pq.ParquetFile(file).schema_arrow.names)
        required = {"episode_index", "frame_index", "task_index"}
        missing = required - available
        if missing:
            raise KeyError(f"{file} is missing required columns: {sorted(missing)}")
        if require_success and "is_success" not in available:
            raise KeyError(
                f"{file} has no is_success column. A rollout dataset must record "
                "the final outcome for every episode."
            )
        columns = ["episode_index", "frame_index", "task_index"]
        if "task" in available:
            columns.append("task")
        if "is_success" in available:
            columns.append("is_success")
        tables.append(pq.read_table(file, columns=columns))

    table = pa.concat_tables(tables, promote_options="default")
    result: dict[str, np.ndarray] = {
        name: np.asarray(table.column(name).to_pylist()) for name in table.column_names
    }
    order = np.lexsort((result["frame_index"], result["episode_index"]))
    return {name: values[order] for name, values in result.items()}


def _label_dataset(
    root: Path,
    *,
    dataset_type: str,
    tag: str,
    spec: ReCapRewardSpecEdited,
) -> dict[str, Any]:
    """Compute labels for one SFT or rollout dataset and write one sidecar."""
    if dataset_type not in {"sft", "rollout"}:
        raise ValueError(f"Unsupported dataset type {dataset_type!r}")
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {root}")

    columns = _read_frames(root, require_success=dataset_type == "rollout")
    tasks = _load_tasks(root)
    episode_indices = columns["episode_index"].astype(np.int64)
    frame_indices = columns["frame_index"].astype(np.int64)
    task_indices = columns["task_index"].astype(np.int64)
    unique_episodes, starts, lengths = np.unique(
        episode_indices, return_index=True, return_counts=True
    )

    returns = np.empty(len(episode_indices), dtype=np.float32)
    rewards = np.empty(len(episode_indices), dtype=np.float32)
    successes = 0
    for episode, start, length in zip(
        unique_episodes.tolist(), starts.tolist(), lengths.tolist(), strict=True
    ):
        episode_slice = slice(start, start + length)
        expected_frames = np.arange(length, dtype=np.int64)
        if not np.array_equal(frame_indices[episode_slice], expected_frames):
            raise ValueError(
                f"Episode {episode} frame_index must be contiguous 0..{length - 1}"
            )
        success = True
        if dataset_type == "rollout":
            flags = columns["is_success"][episode_slice].astype(bool)
            success = bool(flags[-1])
        successes += int(success)
        episode_reward = episode_rewards(length, success=success, spec=spec)
        rewards[episode_slice] = episode_reward
        returns[episode_slice] = discounted_returns(episode_reward, spec.gamma)

    if "task" in columns:
        prompts = [str(value) for value in columns["task"]]
    else:
        prompts = [tasks.get(int(index), "perform the task") for index in task_indices]

    sidecar = pa.table(
        {
            "episode_index": pa.array(episode_indices),
            "frame_index": pa.array(frame_indices),
            "return": pa.array(returns),
            "reward": pa.array(rewards),
            "prompt": pa.array(prompts, type=pa.string()),
        }
    )
    output = root / "meta" / f"returns_{tag}.parquet"
    pq.write_table(sidecar, output)

    stats = {
        "dataset_path": str(root),
        "dataset_type": dataset_type,
        "episodes": int(len(unique_episodes)),
        "successes": int(successes),
        "failures": int(len(unique_episodes) - successes),
        "frames": int(len(returns)),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "sidecar": str(output),
    }
    stats_path = root / "meta" / f"returns_{tag}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def _reward_spec(config: dict[str, Any]) -> ReCapRewardSpecEdited:
    reward = config["reward"]
    value_model = config["value_model"]
    return ReCapRewardSpecEdited(
        step_reward=float(reward["step_reward"]),
        success_terminal_reward=float(reward["success_terminal_reward"]),
        failure_terminal_reward=float(reward["failure_terminal_reward"]),
        gamma=float(reward["gamma"]),
        num_bins=int(value_model["num_bins"]),
        v_min=float(value_model["v_min"]),
        v_max=float(value_model["v_max"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_edited.yaml"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    spec = _reward_spec(config)
    tag = str(config["data"]["returns_tag"])

    summaries = []
    for entry in config["data"]["datasets"]:
        path = entry.get("path")
        if path is None:
            if entry.get("type") == "rollout" and config["data"].get(
                "require_rollout_dataset", False
            ):
                raise ValueError(
                    "Set data.datasets rollout.path in config_edited.yaml before "
                    "computing RECAP labels."
                )
            continue
        summaries.append(
            _label_dataset(
                Path(path).expanduser().resolve(),
                dataset_type=str(entry["type"]),
                tag=tag,
                spec=spec,
            )
        )

    print(json.dumps({"reward_spec": spec.__dict__, "datasets": summaries}, indent=2))


if __name__ == "__main__":
    main()
