# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Fine-tune PiperX Pi0.5 on multiple value-labeled LeRobot datasets."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler
from rlinf.models.embodiment.openpi_rlinf import get_model
from rlinf.models.embodiment.openpi_rlinf.checkpoint import (
    resolve_native_adapter,
    save_native_adapter,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _concat_parquet(files: list[Path], columns: list[str]) -> pa.Table:
    if not files:
        raise FileNotFoundError("No parquet files were found")
    return pa.concat_tables(
        [pq.read_table(path, columns=columns) for path in files],
        promote_options="default",
    )


def _selection(entry: dict[str, Any], episodes: np.ndarray) -> np.ndarray:
    selected = np.ones(len(episodes), dtype=bool)
    if entry.get("episode_start") is not None:
        selected &= episodes >= int(entry["episode_start"])
    if entry.get("episode_stop") is not None:
        selected &= episodes < int(entry["episode_stop"])
    return selected


def _load_and_validate(
    config_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = yaml.safe_load(config_path.read_text())
    data = config["data"]
    suffix = str(data["prompt_suffix"])
    tag = str(data["advantage_tag"])
    summaries = []
    all_labels = []
    total_episodes = 0
    for entry in data["sources"]:
        root = Path(entry["path"]).expanduser().resolve()
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot info.json not found: {info_path}")
        info = json.loads(info_path.read_text())
        required = {
            "observation.state",
            "observation.images.image",
            "observation.images.image2",
            "action",
        }
        missing = required - set(info["features"])
        if missing:
            raise KeyError(f"{root} is missing features: {sorted(missing)}")

        task_table = pq.read_table(root / "meta" / "tasks.parquet", columns=["task"])
        tasks = [str(task) for task in task_table.column("task").to_pylist()]
        if not tasks or any(suffix in task for task in tasks):
            raise ValueError(
                f"{root} must store original task text without {suffix!r}"
            )

        frame_table = _concat_parquet(
            sorted((root / "data").rglob("*.parquet")),
            ["episode_index", "frame_index"],
        )
        episode = frame_table.column("episode_index").to_numpy().astype(np.int64)
        frame = frame_table.column("frame_index").to_numpy().astype(np.int64)
        selected = _selection(entry, episode)
        episode = episode[selected]
        frame = frame[selected]
        if not len(episode):
            raise ValueError(f"Episode selection is empty for {entry['name']}")
        episodes = int(np.unique(episode).size)
        expected = int(entry["expected_episodes"])
        if episodes != expected:
            raise ValueError(
                f"{entry['name']} selects {episodes} episodes, expected {expected}"
            )

        advantage_path = root / "meta" / f"advantages_{tag}.parquet"
        if not advantage_path.is_file():
            raise FileNotFoundError(
                f"Value-derived labels are missing: {advantage_path}. Run "
                "label_value_advantages_edited.py first."
            )
        advantage = pq.read_table(
            advantage_path,
            columns=["episode_index", "frame_index", "advantage"],
        )
        adv_episode = advantage.column("episode_index").to_numpy().astype(np.int64)
        adv_frame = advantage.column("frame_index").to_numpy().astype(np.int64)
        labels = advantage.column("advantage").to_numpy(zero_copy_only=False).astype(bool)
        data_keys = (episode << 32) | frame
        adv_keys = (adv_episode << 32) | adv_frame
        order = np.argsort(adv_keys)
        positions = np.searchsorted(adv_keys[order], data_keys)
        if (
            np.any(positions >= len(adv_keys))
            or not np.array_equal(adv_keys[order][positions], data_keys)
        ):
            raise KeyError(f"Advantage sidecar does not cover every frame in {entry['name']}")
        aligned_labels = labels[order][positions]
        positives = int(aligned_labels.sum())
        all_labels.append(aligned_labels)
        total_episodes += episodes
        summaries.append(
            {
                "name": str(entry["name"]),
                "path": str(root),
                "episodes": episodes,
                "frames": len(episode),
                "positive_frames": positives,
                "negative_frames": int(len(episode) - positives),
            }
        )

    expected_total = int(data["expected_total_episodes"])
    if total_episodes != expected_total:
        raise ValueError(
            f"Selected {total_episodes} episodes in total, expected {expected_total}"
        )
    labels = np.concatenate(all_labels)
    if labels.all() or not labels.any():
        raise ValueError("Mixed policy training requires both advantage labels 0 and 1")
    conditioning = config["advantage_conditioning"]
    if conditioning.get("mode") != "positive_only":
        raise ValueError("advantage_conditioning.mode must be positive_only")
    probability = float(conditioning["unconditional_probability"])
    if not 0.0 <= probability <= 1.0:
        raise ValueError("unconditional_probability must be in [0, 1]")
    return config, summaries


class _NativeAdvantageDataLoader:
    def __init__(self, torch_loader: Any) -> None:
        self.torch_loader = torch_loader

    def __iter__(self):
        from rlinf.models.embodiment.openpi_rlinf.modules.model import Observation

        for batch in self.torch_loader:
            yield {
                "observation": Observation.from_dict(batch),
                "actions": batch["actions"],
                "advantage": batch["advantage"],
                "positive_tokenized_prompt": batch["tokenized_positive_prompt"],
                "positive_tokenized_prompt_mask": batch[
                    "tokenized_positive_prompt_mask"
                ],
            }


def _make_data_loader(config: dict[str, Any], model_config: Any):
    import openpi.training.data_loader as openpi_data_loader
    from data_edited import (
        AdvantagePreservingDatasetEdited,
        PiperXPolicyFramesEdited,
        TokenizePositiveAdvantageEdited,
    )
    from openpi import transforms as openpi_transforms
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    training = config["training"]
    sources = config["data"]["sources"]
    first_path = str(Path(sources[0]["path"]).expanduser().resolve())
    openpi_config = get_openpi_config(
        model_config.openpi.config_name,
        model_path=model_config.model_path,
        batch_size=int(training["batch_size"]),
        repo_id=first_path,
        data_kwargs=model_config.openpi_data,
    )
    openpi_config = dataclasses.replace(
        openpi_config,
        num_workers=int(config["data"].get("num_workers", 0)),
        seed=int(training["seed"]),
    )
    data_config = openpi_config.data.create(
        openpi_config.assets_dirs,
        openpi_config.model,
    )
    replacements = 0
    inputs = []
    for transform in data_config.model_transforms.inputs:
        if isinstance(transform, openpi_transforms.TokenizePrompt):
            transform = TokenizePositiveAdvantageEdited(
                tokenizer=transform.tokenizer,
                positive_prompt_suffix=str(config["data"]["prompt_suffix"]),
                discrete_state_input=transform.discrete_state_input,
            )
            replacements += 1
        inputs.append(transform)
    if replacements != 1:
        raise RuntimeError(f"Expected one TokenizePrompt transform, got {replacements}")
    data_config = dataclasses.replace(
        data_config,
        model_transforms=dataclasses.replace(
            data_config.model_transforms,
            inputs=tuple(inputs),
        ),
    )

    transformed_sources = []
    tag = str(config["data"]["advantage_tag"])
    for entry in sources:
        root = Path(entry["path"]).expanduser().resolve()
        raw = PiperXPolicyFramesEdited(
            root,
            action_horizon=int(openpi_config.model.action_horizon),
            advantage_path=root / "meta" / f"advantages_{tag}.parquet",
            episode_start=entry.get("episode_start"),
            episode_stop=entry.get("episode_stop"),
            video_tolerance_s=float(config["data"].get("video_tolerance_s", 0.03)),
            video_cache_size=int(config["data"].get("video_cache_size", 32)),
        )
        transformed = openpi_data_loader.transform_dataset(raw, data_config)
        transformed_sources.append(
            AdvantagePreservingDatasetEdited(transformed, raw.advantages)
        )
    dataset = ConcatDataset(transformed_sources)
    torch_loader = openpi_data_loader.TorchDataLoader(
        dataset,
        local_batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=openpi_config.num_workers,
        seed=openpi_config.seed,
        framework="pytorch",
    )
    return _NativeAdvantageDataLoader(torch_loader)


def _parameter_groups(model) -> dict[str, int]:
    groups = {
        "vlm_lora": 0,
        "action_expert_lora": 0,
        "action_projections": 0,
        "other": 0,
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("action_", "time_", "state_proj.")):
            group = "action_projections"
        elif name.startswith("llm.") and any(
            marker in name
            for marker in (
                ".attn.q_proj.0.",
                ".attn.k_proj.0.",
                ".attn.v_proj.0.",
                ".attn.o_proj.0.",
                ".mlps.0.",
            )
        ):
            group = "vlm_lora"
        elif name.startswith("llm.") and any(
            marker in name
            for marker in (
                ".attn.q_proj.1.",
                ".attn.k_proj.1.",
                ".attn.v_proj.1.",
                ".attn.o_proj.1.",
                ".mlps.1.",
            )
        ):
            group = "action_expert_lora"
        else:
            group = "other"
        groups[group] += parameter.numel()
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("value_binary_960_edited.yaml"),
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, sources = _load_and_validate(args.config.expanduser().resolve())
    if args.max_steps is not None:
        if args.max_steps <= 0:
            parser.error("--max-steps must be positive")
        config["training"]["max_steps"] = args.max_steps
    if args.output_dir is not None:
        config["paths"]["output_dir"] = str(args.output_dir.expanduser().resolve())
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "episodes": sum(item["episodes"] for item in sources),
                    "frames": sum(item["frames"] for item in sources),
                    "sources": sources,
                },
                indent=2,
            )
        )
        return

    training = config["training"]
    device = torch.device(str(training["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested, but CUDA is unavailable")
    output_dir = Path(config["paths"]["output_dir"]).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Training output must be a new directory: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )

    seed = int(training["seed"])
    _set_seed(seed)
    torch.set_float32_matmul_precision("high")
    model_config = OmegaConf.create(config["model"])
    initial_checkpoint = Path(
        config["paths"]["initial_policy_checkpoint"]
    ).expanduser().resolve()
    model_config.model_path = str(initial_checkpoint)
    model_config.openpi_data.norm_stats_path = str(
        Path(config["paths"]["norm_stats"]).expanduser().resolve()
    )

    print(f"Loading stage-one Pi0.5 from {initial_checkpoint}", flush=True)
    model = get_model(model_config, torch_dtype=torch.float32)
    model.configure_lora(
        train_vision=bool(model_config.openpi.lora_train_vision),
        train_action_projections=bool(
            model_config.openpi.lora_train_action_projections
        ),
    )
    model.to(device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    parameter_groups = _parameter_groups(model)
    if not parameter_groups["vlm_lora"] or not parameter_groups["action_expert_lora"]:
        raise RuntimeError("Both VLM and action-expert LoRA parameters must be trainable")
    if parameter_groups["other"]:
        raise RuntimeError(f"Unexpected trainable parameters: {parameter_groups['other']}")
    print(json.dumps({"trainable_parameters": parameter_groups}, indent=2), flush=True)

    data_loader = _make_data_loader(config, model_config)
    data_iterator = iter(data_loader)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        betas=(float(training["adam_beta1"]), float(training["adam_beta2"])),
        eps=float(training["adam_eps"]),
        weight_decay=float(training["weight_decay"]),
    )
    max_steps = int(training["max_steps"])
    scheduler = get_lr_scheduler(
        "openpi_cosine",
        optimizer,
        num_warmup_steps=int(training["warmup_steps"]),
        num_training_steps=max_steps,
        min_lr=float(training["min_learning_rate"]),
    )
    writer = SummaryWriter(output_dir / "tensorboard")
    progress = tqdm(range(1, max_steps + 1), desc="Value-conditioned Pi0.5 LoRA")
    running_loss = 0.0
    log_every = int(training["log_every"])
    unconditional_probability = float(
        config["advantage_conditioning"]["unconditional_probability"]
    )
    for step in progress:
        batch = next(data_iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            result = model.sft_forward(
                data=batch,
                advantage_unconditional_probability=unconditional_probability,
            )
            metrics = result if isinstance(result, dict) else {}
            loss = result["loss"] if isinstance(result, dict) else result
            loss = loss.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(training["grad_clip_norm"])
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}: {grad_norm}")
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.detach())
        running_loss += loss_value
        writer.add_scalar("train/loss", loss_value, step)
        writer.add_scalar("train/grad_norm", float(grad_norm), step)
        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], step)
        for name in (
            "advantage_positive_fraction",
            "advantage_conditional_fraction",
            "advantage_positive_unconditional_fraction",
        ):
            if name in metrics:
                writer.add_scalar(f"train/{name}", float(metrics[name]), step)
        if step % log_every == 0:
            progress.set_postfix(loss=f"{running_loss / log_every:.5f}")
            running_loss = 0.0

    loaded_adapter = resolve_native_adapter(initial_checkpoint)
    dense_base = (
        Path(loaded_adapter["base_model_path"])
        if loaded_adapter is not None
        else initial_checkpoint
    )
    checkpoint = output_dir / "checkpoints" / f"global_step_{max_steps}" / "actor"
    save_native_adapter(
        model,
        checkpoint,
        base_model_path=dense_base,
        prompt_suffix=str(config["data"]["prompt_suffix"]),
        metadata={
            "initial_checkpoint": str(initial_checkpoint),
            "value_checkpoint": str(Path(config["paths"]["value_checkpoint"]).resolve()),
            "advantage_tag": str(config["data"]["advantage_tag"]),
            "advantage_conditioning": "positive_only",
            "unconditional_probability": unconditional_probability,
            "sources": sources,
            "steps": max_steps,
            "parameter_groups": parameter_groups,
        },
    )
    summary = {
        "initial_policy_checkpoint": str(initial_checkpoint),
        "dense_base_checkpoint": str(dense_base),
        "adapter_checkpoint": str(checkpoint),
        "value_checkpoint": str(Path(config["paths"]["value_checkpoint"]).resolve()),
        "episodes": sum(item["episodes"] for item in sources),
        "frames": sum(item["frames"] for item in sources),
        "positive_frames": sum(item["positive_frames"] for item in sources),
        "negative_frames": sum(item["negative_frames"] for item in sources),
        "sources": sources,
        "steps": max_steps,
        "batch_size": int(training["batch_size"]),
        "trainable_parameters": parameter_groups,
        "dense_checkpoint_written": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    writer.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
