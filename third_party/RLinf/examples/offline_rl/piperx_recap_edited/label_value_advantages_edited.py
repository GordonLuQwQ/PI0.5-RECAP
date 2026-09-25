# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Infer Pi0.5 VLM values and write mixed 0/1 RECAP advantage labels."""

from __future__ import annotations

import argparse
import json
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from checkpoint_edited import load_trained_pi05
from compute_returns_edited import _label_dataset
from data_edited import PiperXValueFramesEdited, collate_value_frames_edited
from pi05_value_critic_edited import Pi05VlmValueCriticEdited
from reward_edited import ReCapRewardSpecEdited


def _reward_spec(config: dict[str, Any]) -> ReCapRewardSpecEdited:
    reward = config["reward"]
    value = config["value_model"]
    return ReCapRewardSpecEdited(
        step_reward=float(reward["step_reward"]),
        success_terminal_reward=float(reward["success_terminal_reward"]),
        failure_terminal_reward=float(reward["failure_terminal_reward"]),
        gamma=float(reward["gamma"]),
        num_bins=int(value["num_bins"]),
        v_min=float(value["v_min"]),
        v_max=float(value["v_max"]),
    )


def _source_name(entry: dict[str, Any]) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(entry["name"])).strip("_")
    if not name:
        raise ValueError("Every data source needs a non-empty name")
    return name


def _selection(entry: dict[str, Any]) -> tuple[int | None, int | None]:
    start = entry.get("episode_start")
    stop = entry.get("episode_stop")
    return (
        None if start is None else int(start),
        None if stop is None else int(stop),
    )


def _make_dataset(
    config: dict[str, Any], entry: dict[str, Any]
) -> PiperXValueFramesEdited:
    data = config["data"]
    start, stop = _selection(entry)
    return PiperXValueFramesEdited(
        entry["path"],
        returns_tag=str(data["returns_tag"]),
        third_person_key=str(data["camera_keys"]["third_person"]),
        wrist_key=str(data["camera_keys"]["wrist"]),
        state_key=str(data["state_key"]),
        episode_start=start,
        episode_stop=stop,
        video_tolerance_s=float(data.get("video_tolerance_s", 0.03)),
        video_cache_size=int(data.get("video_cache_size", 32)),
    )


def _prepare_sources(
    config: dict[str, Any], *, write_returns: bool
) -> tuple[list[tuple[dict[str, Any], PiperXValueFramesEdited]], list[dict[str, Any]]]:
    spec = _reward_spec(config)
    returns_tag = str(config["data"]["returns_tag"])
    prepared = []
    summaries = []
    total_episodes = 0
    for entry in config["data"]["sources"]:
        root = Path(entry["path"]).expanduser().resolve()
        if write_returns:
            _label_dataset(
                root,
                dataset_type=str(entry["returns_type"]),
                tag=returns_tag,
                spec=spec,
            )
        dataset = _make_dataset(config, entry)
        episodes = int(np.unique(dataset.episode_indices).size)
        expected = int(entry["expected_episodes"])
        if episodes != expected:
            raise ValueError(
                f"{entry['name']} selects {episodes} episodes, expected {expected}"
            )
        if np.any(dataset.frame_indices < 0):
            raise ValueError(f"{entry['name']} contains negative frame indices")
        total_episodes += episodes
        summary = {
            "name": _source_name(entry),
            "path": str(root),
            "episodes": episodes,
            "frames": len(dataset),
            "episode_start": entry.get("episode_start"),
            "episode_stop": entry.get("episode_stop"),
            "returns_type": str(entry["returns_type"]),
        }
        prepared.append((entry, dataset))
        summaries.append(summary)
    expected_total = int(config["data"]["expected_total_episodes"])
    if total_episodes != expected_total:
        raise ValueError(
            f"Selected data contains {total_episodes} episodes, expected {expected_total}"
        )
    return prepared, summaries


def _cache_identity(
    entry: dict[str, Any], dataset: PiperXValueFramesEdited, checkpoint: Path
) -> dict[str, Any]:
    stat = checkpoint.stat()
    return {
        "format": "piperx_pi05_value_predictions_edited_v1",
        "dataset": str(dataset.root),
        "source": _source_name(entry),
        "frames": len(dataset),
        "episode_start": entry.get("episode_start"),
        "episode_stop": entry.get("episode_stop"),
        "first_key": [
            int(dataset.episode_indices[0]),
            int(dataset.frame_indices[0]),
        ],
        "last_key": [
            int(dataset.episode_indices[-1]),
            int(dataset.frame_indices[-1]),
        ],
        "value_checkpoint": str(checkpoint),
        "value_checkpoint_size": int(stat.st_size),
        "value_checkpoint_mtime_ns": int(stat.st_mtime_ns),
    }


def _predict_values(
    *,
    config: dict[str, Any],
    entry: dict[str, Any],
    dataset: PiperXValueFramesEdited,
    model: Pi05VlmValueCriticEdited,
    checkpoint: Path,
    overwrite_cache: bool,
) -> np.ndarray:
    inference = config["inference"]
    tag = str(config["data"]["advantage_tag"])
    stem = f"value_predictions_{tag}_{_source_name(entry)}"
    final_path = dataset.root / "meta" / f"{stem}.npy"
    partial_path = dataset.root / "meta" / f".{stem}.partial.npy"
    progress_path = dataset.root / "meta" / f".{stem}.progress.json"
    manifest_path = dataset.root / "meta" / f"{stem}.json"
    identity = _cache_identity(entry, dataset, checkpoint)

    if overwrite_cache:
        for path in (final_path, partial_path, progress_path, manifest_path):
            path.unlink(missing_ok=True)

    if final_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity or manifest.get("complete") is not True:
            raise ValueError(
                f"Stale value cache exists for {_source_name(entry)}; rerun with "
                "--overwrite-cache"
            )
        values = np.load(final_path, mmap_mode="r")
        if values.shape != (len(dataset),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid completed value cache: {final_path}")
        print(json.dumps({"reused_value_cache": str(final_path), "frames": len(values)}))
        return np.asarray(values, dtype=np.float32)

    completed = 0
    if partial_path.is_file() or progress_path.is_file():
        if not partial_path.is_file() or not progress_path.is_file():
            raise ValueError(
                f"Incomplete cache metadata for {_source_name(entry)}; rerun with "
                "--overwrite-cache"
            )
        progress = json.loads(progress_path.read_text())
        if progress.get("identity") != identity:
            raise ValueError(
                f"Partial value cache does not match {_source_name(entry)}; rerun "
                "with --overwrite-cache"
            )
        completed = int(progress["completed_frames"])
        values = np.lib.format.open_memmap(partial_path, mode="r+")
        if values.shape != (len(dataset),) or not 0 <= completed <= len(dataset):
            raise ValueError(f"Invalid partial value cache: {partial_path}")
        print(json.dumps({"resumed_value_cache": str(partial_path), "completed": completed}))
    else:
        values = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float32, shape=(len(dataset),)
        )
        values[:] = np.nan
        values.flush()

    if completed < len(dataset):
        remaining = Subset(dataset, range(completed, len(dataset)))
        workers = int(inference.get("num_workers", 0))
        loader = DataLoader(
            remaining,
            batch_size=int(inference["batch_size"]),
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_value_frames_edited,
            pin_memory=bool(inference.get("pin_memory", True)),
            persistent_workers=workers > 0,
        )
        every = max(1, int(inference.get("checkpoint_every_batches", 10)))
        device = torch.device(str(inference["device"]))
        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"Value VLM: {_source_name(entry)}",
            unit="batch",
        )
        cursor = completed
        with torch.inference_mode():
            for batch_index, batch in enumerate(progress, start=1):
                observation = model.pi05.env_obs_to_observation(
                    {
                        "main_images": batch["main_images"],
                        "wrist_images": batch["wrist_images"],
                        "states": batch["states"],
                        "task_descriptions": batch["task_descriptions"],
                    }
                )
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with autocast:
                    predicted = model(observation).values
                predicted_np = predicted.detach().float().cpu().numpy()
                end = cursor + len(predicted_np)
                values[cursor:end] = predicted_np
                cursor = end
                if batch_index % every == 0 or cursor == len(dataset):
                    values.flush()
                    progress_path.write_text(
                        json.dumps(
                            {"identity": identity, "completed_frames": cursor},
                            indent=2,
                        )
                        + "\n"
                    )
                progress.set_postfix(frames=f"{cursor}/{len(dataset)}")

    values.flush()
    if not np.isfinite(values).all():
        raise ValueError(f"Value inference left NaN or Inf in {partial_path}")
    del values
    os.replace(partial_path, final_path)
    progress_path.unlink(missing_ok=True)
    manifest_path.write_text(
        json.dumps({"identity": identity, "complete": True}, indent=2) + "\n"
    )
    return np.asarray(np.load(final_path, mmap_mode="r"), dtype=np.float32)


def _continuous_advantages(
    dataset: PiperXValueFramesEdited,
    values: np.ndarray,
    *,
    lookahead: int,
    gamma: float,
    discount_next_value: bool,
    global_return_min: float,
) -> dict[str, np.ndarray]:
    if gamma != 1.0:
        raise ValueError("This PiperX scorer currently requires reward.gamma=1.0")
    if lookahead <= 0:
        raise ValueError("advantage.lookahead_steps must be positive")
    if global_return_min >= 0:
        raise ValueError("normalization.return_min must be negative")
    n = len(dataset)
    advantages = np.empty(n, dtype=np.float32)
    value_next = np.empty(n, dtype=np.float32)
    reward_sum_raw = np.empty(n, dtype=np.float32)
    num_valid = np.empty(n, dtype=np.int32)
    starts = np.r_[0, np.flatnonzero(np.diff(dataset.episode_indices)) + 1]
    ends = np.r_[starts[1:], n]
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        length = end - start
        local = np.arange(length, dtype=np.int64)
        valid = np.minimum(lookahead, length - local)
        has_next = local + lookahead < length
        next_values = np.zeros(length, dtype=np.float32)
        next_values[has_next] = values[start + local[has_next] + lookahead]
        raw = np.empty(length, dtype=np.float32)
        raw[has_next] = (
            dataset.raw_returns[start + local[has_next]]
            - dataset.raw_returns[start + local[has_next] + lookahead]
        )
        raw[~has_next] = dataset.raw_returns[start + local[~has_next]]
        normalized_reward = raw / abs(global_return_min)
        gamma_k = np.power(gamma, valid, dtype=np.float64).astype(np.float32)
        if not discount_next_value:
            gamma_k.fill(1.0)
        sl = slice(start, end)
        value_next[sl] = next_values
        reward_sum_raw[sl] = raw
        num_valid[sl] = valid
        advantages[sl] = normalized_reward + gamma_k * next_values - values[sl]
    return {
        "advantage_continuous": advantages,
        "value_current": values.astype(np.float32, copy=False),
        "value_next": value_next,
        "reward_sum_raw": reward_sum_raw,
        "reward_sum": reward_sum_raw / abs(global_return_min),
        "num_valid_rewards": num_valid,
    }


def _write_labels(
    *,
    config: dict[str, Any],
    prepared: list[tuple[dict[str, Any], PiperXValueFramesEdited]],
    scored: list[dict[str, np.ndarray]],
    threshold: float,
) -> list[dict[str, Any]]:
    tag = str(config["data"]["advantage_tag"])
    force_sft = bool(config["advantage"].get("force_sft_positive", False))
    summaries = []
    for (entry, dataset), result in zip(prepared, scored, strict=True):
        labels = result["advantage_continuous"] >= threshold
        if force_sft and str(entry["returns_type"]).lower() == "sft":
            labels[:] = True
        table = pa.table(
            {
                "episode_index": pa.array(dataset.episode_indices),
                "frame_index": pa.array(dataset.frame_indices),
                "advantage_continuous": pa.array(result["advantage_continuous"]),
                "advantage": pa.array(labels),
                "return": pa.array(dataset.raw_returns),
                "value_current": pa.array(result["value_current"]),
                "value_next": pa.array(result["value_next"]),
                "reward_sum": pa.array(result["reward_sum"]),
                "reward_sum_raw": pa.array(result["reward_sum_raw"]),
                "num_valid_rewards": pa.array(result["num_valid_rewards"]),
            }
        )
        output = dataset.root / "meta" / f"advantages_{tag}.parquet"
        pq.write_table(table, output)
        positives = int(labels.sum())
        summary = {
            "name": _source_name(entry),
            "path": str(dataset.root),
            "episodes": int(np.unique(dataset.episode_indices).size),
            "frames": len(dataset),
            "positive_frames": positives,
            "negative_frames": int(len(dataset) - positives),
            "positive_fraction": positives / len(dataset),
            "advantage_min": float(result["advantage_continuous"].min()),
            "advantage_max": float(result["advantage_continuous"].max()),
            "sidecar": str(output),
        }
        summaries.append(summary)
        stats_path = dataset.root / "meta" / f"advantages_{tag}_stats.json"
        stats_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("value_binary_960_edited.yaml"),
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    # Return sidecars are small and deterministic.  Build them during both the
    # dry validation and the real run so the exact episode/outcome selection is
    # checked before loading the 3.39B-parameter value backbone.
    prepared, source_summaries = _prepare_sources(config, write_returns=True)
    check = {
        "status": "passed",
        "expected_total_episodes": int(config["data"]["expected_total_episodes"]),
        "selected_total_episodes": sum(item["episodes"] for item in source_summaries),
        "selected_total_frames": sum(item["frames"] for item in source_summaries),
        "sources": source_summaries,
    }
    if args.check_only:
        print(json.dumps(check, indent=2))
        return

    device = torch.device(str(config["inference"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA value inference was requested, but CUDA is unavailable")
    checkpoint = Path(config["paths"]["value_checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Value checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    configured_base = str(
        Path(config["paths"]["pi05_checkpoint"]).expanduser().resolve()
    )
    saved_base = str(Path(payload["base_pi05_checkpoint"]).expanduser().resolve())
    if saved_base != configured_base:
        raise ValueError(
            f"Value checkpoint base mismatch: saved={saved_base}, configured={configured_base}"
        )

    spec = _reward_spec(config)
    print(f"Loading value VLM from {checkpoint}", flush=True)
    pi05 = load_trained_pi05(config).to(device)
    value_config = config["value_model"]
    model = Pi05VlmValueCriticEdited(
        pi05,
        reward_spec=spec,
        hidden_size=int(value_config["hidden_size"]),
        head_hidden_size=int(value_config["head_hidden_size"]),
        dropout=float(value_config["dropout"]),
        pooling=str(value_config["pooling"]),
        train_pi05_vlm_lora=False,
        train_vision_encoder=False,
        gradient_checkpointing=False,
    ).to(device)
    model.load_value_head(checkpoint)
    model.eval()

    predicted = []
    scored = []
    lookahead = int(config["advantage"]["lookahead_steps"])
    global_return_min = float(config["normalization"]["return_min"])
    for entry, dataset in prepared:
        values = _predict_values(
            config=config,
            entry=entry,
            dataset=dataset,
            model=model,
            checkpoint=checkpoint,
            overwrite_cache=args.overwrite_cache,
        )
        predicted.append(values)
        scored.append(
            _continuous_advantages(
                dataset,
                values,
                lookahead=lookahead,
                gamma=float(config["reward"]["gamma"]),
                discount_next_value=bool(
                    config["advantage"].get("discount_next_value", True)
                ),
                global_return_min=global_return_min,
            )
        )

    all_advantages = np.concatenate(
        [item["advantage_continuous"] for item in scored]
    )
    positive_fraction = float(config["advantage"]["positive_fraction"])
    if not 0.0 < positive_fraction < 1.0:
        raise ValueError("advantage.positive_fraction must be in (0, 1)")
    threshold = float(np.percentile(all_advantages, (1.0 - positive_fraction) * 100.0))
    summaries = _write_labels(
        config=config,
        prepared=prepared,
        scored=scored,
        threshold=threshold,
    )
    total_positive = sum(item["positive_frames"] for item in summaries)
    total_frames = sum(item["frames"] for item in summaries)
    if total_positive == 0 or total_positive == total_frames:
        raise RuntimeError("Value labeling produced only one binary class")
    report = {
        **check,
        "value_checkpoint": str(checkpoint),
        "advantage_tag": str(config["data"]["advantage_tag"]),
        "lookahead_steps": lookahead,
        "global_return_min": global_return_min,
        "positive_fraction_target": positive_fraction,
        "unified_threshold": threshold,
        "positive_frames": total_positive,
        "negative_frames": total_frames - total_positive,
        "actual_positive_fraction": total_positive / total_frames,
        "force_sft_positive": bool(
            config["advantage"].get("force_sft_positive", False)
        ),
        "sources": summaries,
    }
    report_path = Path(config["paths"]["label_report"]).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
