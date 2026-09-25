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

"""Read PiperX LeRobot v3 frames for the EDITED Pi0.5 value critic."""

from __future__ import annotations

import dataclasses
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler


def _concat_parquet(files: list[Path], columns: list[str] | None = None) -> pa.Table:
    if not files:
        raise FileNotFoundError("No parquet files were found")
    return pa.concat_tables(
        [pq.read_table(file, columns=columns) for file in files],
        promote_options="default",
    )


class _VideoFrameReaderEdited:
    """Decode random MP4 frames while reusing a bounded set of open files."""

    def __init__(self, max_open_files: int = 32) -> None:
        if max_open_files <= 0:
            raise ValueError("max_open_files must be positive")
        self.max_open_files = int(max_open_files)
        self._containers: OrderedDict[Path, av.container.InputContainer] = OrderedDict()

    def _container(self, video_path: Path) -> av.container.InputContainer:
        container = self._containers.pop(video_path, None)
        if container is None:
            container = av.open(str(video_path))
        self._containers[video_path] = container
        while len(self._containers) > self.max_open_files:
            _, stale = self._containers.popitem(last=False)
            stale.close()
        return container

    def decode(
        self, video_path: Path, timestamp: float, tolerance_s: float
    ) -> np.ndarray:
        """Decode the nearest frame using the same seek rule as LeRobot PyAV."""
        loaded: list[tuple[float, np.ndarray]] = []
        container = self._container(video_path)
        stream = container.streams.video[0]
        offset = max(0, round(timestamp / float(stream.time_base)) - 1)
        container.seek(offset, backward=True, any_frame=False, stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * stream.time_base)
            loaded.append((frame_time, frame.to_ndarray(format="rgb24")))
            if frame_time >= timestamp:
                break

        if not loaded:
            raise RuntimeError(
                f"No frame decoded from {video_path} at {timestamp:.6f}s"
            )
        frame_time, image = min(loaded, key=lambda item: abs(item[0] - timestamp))
        error = abs(frame_time - timestamp)
        if error > tolerance_s:
            raise RuntimeError(
                f"Nearest frame in {video_path} is {error:.6f}s from requested "
                f"timestamp {timestamp:.6f}s; tolerance is {tolerance_s:.6f}s"
            )
        return np.ascontiguousarray(image)

    def close(self) -> None:
        """Close every cached PyAV container."""
        containers = getattr(self, "_containers", None)
        while containers:
            _, container = containers.popitem()
            container.close()

    def __getstate__(self) -> dict[str, Any]:
        """Drop process-local handles when a DataLoader worker pickles us."""
        return {"max_open_files": self.max_open_files, "_containers": OrderedDict()}

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors may run after PyAV modules begin interpreter teardown.
            pass


def _decode_rgb_frame(
    video_path: Path, timestamp: float, tolerance_s: float
) -> np.ndarray:
    """One-shot frame decoder retained for direct checks and small utilities."""
    reader = _VideoFrameReaderEdited(max_open_files=1)
    try:
        return reader.decode(video_path, timestamp, tolerance_s)
    finally:
        reader.close()


class PiperXValueFramesEdited(Dataset):
    """Flat frame dataset joined with an EDITED RECAP return sidecar."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        returns_tag: str,
        third_person_key: str,
        wrist_key: str,
        state_key: str,
        episode_start: int | None = None,
        episode_stop: int | None = None,
        video_tolerance_s: float = 0.03,
        video_cache_size: int = 32,
    ) -> None:
        self.root = Path(dataset_path).expanduser().resolve()
        self.third_person_key = third_person_key
        self.wrist_key = wrist_key
        self.state_key = state_key
        self.video_tolerance_s = float(video_tolerance_s)
        self.video_reader = _VideoFrameReaderEdited(video_cache_size)

        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot info.json not found: {info_path}")
        self.info = json.loads(info_path.read_text())
        for key in (third_person_key, wrist_key, state_key):
            if key not in self.info["features"]:
                raise KeyError(f"Dataset {self.root} has no feature {key!r}")

        data_files = sorted((self.root / "data").rglob("*.parquet"))
        table = _concat_parquet(
            data_files,
            columns=[state_key, "episode_index", "frame_index", "timestamp"],
        )
        self.states = np.asarray(table.column(state_key).to_pylist(), dtype=np.float32)
        self.episode_indices = table.column("episode_index").to_numpy().astype(np.int64)
        self.frame_indices = table.column("frame_index").to_numpy().astype(np.int64)
        self.timestamps = table.column("timestamp").to_numpy().astype(np.float64)

        selection = np.ones(len(self.episode_indices), dtype=bool)
        if episode_start is not None:
            selection &= self.episode_indices >= int(episode_start)
        if episode_stop is not None:
            selection &= self.episode_indices < int(episode_stop)
        if not selection.any():
            raise ValueError(
                f"Episode selection [{episode_start}, {episode_stop}) has no frames "
                f"in {self.root}"
            )
        self.states = self.states[selection]
        self.episode_indices = self.episode_indices[selection]
        self.frame_indices = self.frame_indices[selection]
        self.timestamps = self.timestamps[selection]

        sidecar_path = self.root / "meta" / f"returns_{returns_tag}.parquet"
        if not sidecar_path.is_file():
            raise FileNotFoundError(
                f"Return sidecar not found: {sidecar_path}. Run "
                "compute_returns_edited.py first."
            )
        sidecar = pq.read_table(
            sidecar_path,
            columns=["episode_index", "frame_index", "return", "prompt"],
        )
        self.raw_returns, self.prompts = self._align_sidecar(sidecar)
        self.video_rows = self._load_video_rows()

    def _align_sidecar(self, sidecar: pa.Table) -> tuple[np.ndarray, np.ndarray]:
        side_episode = sidecar.column("episode_index").to_numpy().astype(np.int64)
        side_frame = sidecar.column("frame_index").to_numpy().astype(np.int64)
        if np.any(side_episode < 0) or np.any(side_frame < 0):
            raise ValueError("episode_index and frame_index must be non-negative")
        side_keys = (side_episode << 32) | side_frame
        data_keys = (self.episode_indices << 32) | self.frame_indices
        order = np.argsort(side_keys)
        sorted_keys = side_keys[order]
        positions = np.searchsorted(sorted_keys, data_keys)
        in_range = positions < len(sorted_keys)
        matches = np.zeros_like(in_range, dtype=bool)
        matches[in_range] = sorted_keys[positions[in_range]] == data_keys[in_range]
        if not matches.all():
            first = int(np.flatnonzero(~matches)[0])
            raise KeyError(
                "Return sidecar does not contain dataset frame "
                f"(episode={self.episode_indices[first]}, frame={self.frame_indices[first]})"
            )
        aligned = order[positions]
        returns = sidecar.column("return").to_numpy().astype(np.float32)[aligned]
        prompts = np.asarray(sidecar.column("prompt").to_pylist(), dtype=object)[
            aligned
        ]
        return returns, prompts

    def _load_video_rows(self) -> dict[int, dict[str, tuple[Path, float]]]:
        episode_files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        keys = (self.third_person_key, self.wrist_key)
        columns = ["episode_index"]
        for key in keys:
            columns.extend(
                [
                    f"videos/{key}/chunk_index",
                    f"videos/{key}/file_index",
                    f"videos/{key}/from_timestamp",
                ]
            )
        table = _concat_parquet(episode_files, columns=columns)
        rows: dict[int, dict[str, tuple[Path, float]]] = {}
        template = str(self.info["video_path"])
        for row in table.to_pylist():
            episode = int(row["episode_index"])
            rows[episode] = {}
            for key in keys:
                chunk = int(row[f"videos/{key}/chunk_index"])
                file_index = int(row[f"videos/{key}/file_index"])
                relative = template.format(
                    video_key=key,
                    chunk_index=chunk,
                    file_index=file_index,
                )
                video_path = self.root / relative
                if not video_path.is_file():
                    raise FileNotFoundError(f"Video not found: {video_path}")
                rows[episode][key] = (
                    video_path,
                    float(row[f"videos/{key}/from_timestamp"]),
                )
        return rows

    def __len__(self) -> int:
        return len(self.episode_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = int(self.episode_indices[index])
        local_timestamp = float(self.timestamps[index])
        video = self.video_rows.get(episode)
        if video is None:
            raise KeyError(f"Episode {episode} is absent from meta/episodes")

        third_path, third_start = video[self.third_person_key]
        wrist_path, wrist_start = video[self.wrist_key]
        return {
            "main_image": self.video_reader.decode(
                third_path, third_start + local_timestamp, self.video_tolerance_s
            ),
            "wrist_image": self.video_reader.decode(
                wrist_path, wrist_start + local_timestamp, self.video_tolerance_s
            ),
            "state": self.states[index],
            "prompt": str(self.prompts[index]),
            "raw_return": self.raw_returns[index],
            "episode_index": episode,
            "frame_index": int(self.frame_indices[index]),
        }


class PiperXPolicyFramesEdited(Dataset):
    """Read LeRobot v3 observations and edge-padded action chunks for Pi0.5."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        action_horizon: int,
        advantage_path: str | Path | None = None,
        third_person_key: str = "observation.images.image",
        wrist_key: str = "observation.images.image2",
        state_key: str = "observation.state",
        action_key: str = "action",
        episode_start: int | None = None,
        episode_stop: int | None = None,
        video_tolerance_s: float = 0.03,
        video_cache_size: int = 32,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self.root = Path(dataset_path).expanduser().resolve()
        self.action_horizon = int(action_horizon)
        self.third_person_key = third_person_key
        self.wrist_key = wrist_key
        self.video_tolerance_s = float(video_tolerance_s)
        self.video_reader = _VideoFrameReaderEdited(video_cache_size)

        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot info.json not found: {info_path}")
        self.info = json.loads(info_path.read_text())
        required_features = (third_person_key, wrist_key, state_key, action_key)
        for key in required_features:
            if key not in self.info["features"]:
                raise KeyError(f"Dataset {self.root} has no feature {key!r}")

        table = _concat_parquet(
            sorted((self.root / "data").rglob("*.parquet")),
            columns=[
                state_key,
                action_key,
                "episode_index",
                "frame_index",
                "timestamp",
                "task_index",
            ],
        )
        self.states = np.asarray(table.column(state_key).to_pylist(), dtype=np.float32)
        self.actions = np.asarray(
            table.column(action_key).to_pylist(), dtype=np.float32
        )
        self.episode_indices = table.column("episode_index").to_numpy().astype(np.int64)
        self.frame_indices = table.column("frame_index").to_numpy().astype(np.int64)
        self.timestamps = table.column("timestamp").to_numpy().astype(np.float64)
        task_indices = table.column("task_index").to_numpy().astype(np.int64)

        selection = np.ones(len(self.episode_indices), dtype=bool)
        if episode_start is not None:
            selection &= self.episode_indices >= int(episode_start)
        if episode_stop is not None:
            selection &= self.episode_indices < int(episode_stop)
        if not selection.any():
            raise ValueError(
                f"Episode selection [{episode_start}, {episode_stop}) has no frames "
                f"in {self.root}"
            )
        self.states = self.states[selection]
        self.actions = self.actions[selection]
        self.episode_indices = self.episode_indices[selection]
        self.frame_indices = self.frame_indices[selection]
        self.timestamps = self.timestamps[selection]
        task_indices = task_indices[selection]
        if not len(self.episode_indices):
            raise ValueError(f"Policy dataset has no frames: {self.root}")

        task_rows = pq.read_table(
            self.root / "meta" / "tasks.parquet", columns=["task_index", "task"]
        ).to_pylist()
        tasks = {int(row["task_index"]): str(row["task"]) for row in task_rows}
        try:
            self.prompts = np.asarray(
                [tasks[int(index)] for index in task_indices], dtype=object
            )
        except KeyError as error:
            raise KeyError(
                f"Unknown task_index in {self.root}: {error.args[0]}"
            ) from error

        self.advantages = np.ones(len(self.episode_indices), dtype=bool)
        if advantage_path is not None:
            advantage_table = pq.read_table(
                Path(advantage_path).expanduser().resolve(),
                columns=["episode_index", "frame_index", "advantage"],
            )
            advantage_episode = (
                advantage_table.column("episode_index").to_numpy().astype(np.int64)
            )
            advantage_frame = (
                advantage_table.column("frame_index").to_numpy().astype(np.int64)
            )
            advantage_keys = (advantage_episode << 32) | advantage_frame
            data_keys = (self.episode_indices << 32) | self.frame_indices
            order = np.argsort(advantage_keys)
            sorted_keys = advantage_keys[order]
            positions = np.searchsorted(sorted_keys, data_keys)
            in_range = positions < len(sorted_keys)
            matches = np.zeros_like(in_range, dtype=bool)
            matches[in_range] = sorted_keys[positions[in_range]] == data_keys[in_range]
            if not matches.all():
                first = int(np.flatnonzero(~matches)[0])
                raise KeyError(
                    "Advantage sidecar does not contain dataset frame "
                    f"(episode={self.episode_indices[first]}, frame={self.frame_indices[first]})"
                )
            self.advantages = (
                advantage_table.column("advantage").to_numpy(zero_copy_only=False)[
                    order[positions]
                ]
            ).astype(bool)

        self.episode_ends: dict[int, int] = {}
        starts = np.r_[0, np.flatnonzero(np.diff(self.episode_indices)) + 1]
        ends = np.r_[starts[1:], len(self.episode_indices)]
        for start, end in zip(starts, ends, strict=True):
            episode = int(self.episode_indices[start])
            expected = np.arange(end - start, dtype=np.int64)
            if not np.array_equal(self.frame_indices[start:end], expected):
                raise ValueError(
                    f"Episode {episode} frame_index must be contiguous 0..{end - start - 1}"
                )
            self.episode_ends[episode] = int(end)
        self.video_rows = self._load_video_rows()

    def _load_video_rows(self) -> dict[int, dict[str, tuple[Path, float]]]:
        episode_files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        keys = (self.third_person_key, self.wrist_key)
        columns = ["episode_index"]
        for key in keys:
            columns.extend(
                [
                    f"videos/{key}/chunk_index",
                    f"videos/{key}/file_index",
                    f"videos/{key}/from_timestamp",
                ]
            )
        table = _concat_parquet(episode_files, columns=columns)
        rows: dict[int, dict[str, tuple[Path, float]]] = {}
        template = str(self.info["video_path"])
        for row in table.to_pylist():
            episode = int(row["episode_index"])
            rows[episode] = {}
            for key in keys:
                relative = template.format(
                    video_key=key,
                    chunk_index=int(row[f"videos/{key}/chunk_index"]),
                    file_index=int(row[f"videos/{key}/file_index"]),
                )
                video_path = self.root / relative
                if not video_path.is_file():
                    raise FileNotFoundError(f"Video not found: {video_path}")
                rows[episode][key] = (
                    video_path,
                    float(row[f"videos/{key}/from_timestamp"]),
                )
        return rows

    def __len__(self) -> int:
        return len(self.episode_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = int(self.episode_indices[index])
        video = self.video_rows.get(episode)
        if video is None:
            raise KeyError(f"Episode {episode} is absent from meta/episodes")
        last_index = self.episode_ends[episode] - 1
        action_indices = np.minimum(
            index + np.arange(self.action_horizon, dtype=np.int64), last_index
        )
        timestamp = float(self.timestamps[index])
        third_path, third_start = video[self.third_person_key]
        wrist_path, wrist_start = video[self.wrist_key]
        return {
            "observation/image": self.video_reader.decode(
                third_path, third_start + timestamp, self.video_tolerance_s
            ),
            "observation/wrist_image": self.video_reader.decode(
                wrist_path, wrist_start + timestamp, self.video_tolerance_s
            ),
            "observation/state": self.states[index],
            "actions": self.actions[action_indices],
            "prompt": str(self.prompts[index]),
            "advantage": np.bool_(self.advantages[index]),
        }


@dataclasses.dataclass(frozen=True)
class TokenizePositiveAdvantageEdited:
    """Tokenize the base and positive prompts without decoding images twice."""

    tokenizer: Any
    positive_prompt_suffix: str
    discrete_state_input: bool = False

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        prompt = data.pop("prompt", None)
        if prompt is None:
            raise ValueError("Prompt is required")
        if not isinstance(prompt, str):
            prompt = prompt.item()

        state = None
        if self.discrete_state_input:
            state = data.get("state")
            if state is None:
                raise ValueError("State is required for discrete-state tokenization")

        tokens, token_mask = self.tokenizer.tokenize(prompt, state)
        positive_tokens, positive_token_mask = self.tokenizer.tokenize(
            prompt + self.positive_prompt_suffix,
            state,
        )
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "tokenized_positive_prompt": positive_tokens,
            "tokenized_positive_prompt_mask": positive_token_mask,
        }


class AdvantagePreservingDatasetEdited(Dataset):
    """Restore frame-aligned advantage labels after OpenPI repacking."""

    def __init__(self, transformed_dataset: Any, advantages: np.ndarray) -> None:
        self.transformed_dataset = transformed_dataset
        self.advantages = np.asarray(advantages, dtype=bool)
        if len(self.transformed_dataset) != len(self.advantages):
            raise ValueError(
                "Transformed dataset and advantage labels must have equal length"
            )

    def __len__(self) -> int:
        return len(self.transformed_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.transformed_dataset[index])
        sample["advantage"] = np.bool_(self.advantages[index])
        return sample


def collate_value_frames_edited(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack raw frames while keeping prompts as Python strings."""
    return {
        "main_images": np.stack([sample["main_image"] for sample in samples]),
        "wrist_images": np.stack([sample["wrist_image"] for sample in samples]),
        "states": np.stack([sample["state"] for sample in samples]).astype(np.float32),
        "task_descriptions": [sample["prompt"] for sample in samples],
        "raw_returns": torch.as_tensor(
            [sample["raw_return"] for sample in samples], dtype=torch.float32
        ),
        "episode_indices": torch.as_tensor(
            [sample["episode_index"] for sample in samples], dtype=torch.int64
        ),
        "frame_indices": torch.as_tensor(
            [sample["frame_index"] for sample in samples], dtype=torch.int64
        ),
    }


def build_datasets_edited(config: dict[str, Any]) -> list[PiperXValueFramesEdited]:
    """Construct every configured dataset that has a concrete path."""
    data = config["data"]
    camera_keys = data["camera_keys"]
    datasets = []
    for entry in data["datasets"]:
        if entry.get("path") is None:
            continue
        datasets.append(
            PiperXValueFramesEdited(
                entry["path"],
                returns_tag=data["returns_tag"],
                third_person_key=camera_keys["third_person"],
                wrist_key=camera_keys["wrist"],
                state_key=data["state_key"],
                video_tolerance_s=float(data.get("video_tolerance_s", 0.03)),
                video_cache_size=int(data.get("video_cache_size", 32)),
            )
        )
    if not datasets:
        raise ValueError("No concrete dataset paths are configured")
    return datasets


def build_weighted_sampler_edited(
    datasets: list[PiperXValueFramesEdited],
    entries: list[dict[str, Any]],
    *,
    num_samples: int,
    seed: int,
) -> WeightedRandomSampler:
    """Give each configured dataset its declared mixture probability."""
    concrete_entries = [entry for entry in entries if entry.get("path") is not None]
    if len(concrete_entries) != len(datasets):
        raise ValueError("Dataset entries and constructed datasets disagree")
    weights = []
    for dataset, entry in zip(datasets, concrete_entries, strict=True):
        dataset_weight = float(entry.get("weight", 1.0))
        if dataset_weight <= 0:
            raise ValueError("Every dataset weight must be positive")
        weights.extend([dataset_weight / len(dataset)] * len(dataset))
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


def concatenate_datasets_edited(
    datasets: list[PiperXValueFramesEdited],
) -> ConcatDataset:
    """Return one index space while preserving each source dataset object."""
    return ConcatDataset(datasets)
