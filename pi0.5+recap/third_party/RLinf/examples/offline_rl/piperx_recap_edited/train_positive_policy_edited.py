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

"""Advantage-conditioned LoRA SFT for the native PiperX OpenPI_RLinf Pi0.5."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler
from rlinf.models.embodiment.openpi_rlinf import get_model
from rlinf.models.embodiment.openpi_rlinf.checkpoint import save_native_adapter


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_and_validate(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text())
    dataset = Path(config["paths"]["dataset"]).expanduser().resolve()
    info_path = dataset / "meta" / "info.json"
    export_path = dataset / "positive_export.json"
    if not info_path.is_file() or not export_path.is_file():
        raise FileNotFoundError(
            f"Positive LeRobot dataset is incomplete: expected {info_path} and {export_path}"
        )
    info = json.loads(info_path.read_text())
    export = json.loads(export_path.read_text())
    expected_episodes = int(config["data"]["expected_policy_success_episodes"]) + int(
        config["data"]["expected_expert_suffix_episodes"]
    )
    if int(export["total_episodes"]) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} positive episodes, got {export['total_episodes']}"
        )
    if int(info["total_episodes"]) != expected_episodes:
        raise ValueError(
            "LeRobot info.json episode count disagrees with positive_export.json"
        )
    if int(info["total_frames"]) != int(export["total_frames"]):
        raise ValueError(
            "LeRobot info.json frame count disagrees with positive_export.json"
        )

    suffix = str(config["data"]["prompt_suffix"])
    task_table = pq.read_table(dataset / "meta" / "tasks.parquet", columns=["task"])
    tasks = [str(task) for task in task_table.column("task").to_pylist()]
    if not tasks or any(suffix in task for task in tasks):
        raise ValueError(
            "Stored task text must be the original instruction; the advantage "
            f"sidecar adds {suffix!r} during training. Got {tasks}"
        )
    if export.get("training_prompt_suffix") != suffix:
        raise ValueError("Export and policy config disagree about the prompt suffix")

    tag = str(config["data"]["advantage_tag"])
    advantage_path = dataset / "meta" / f"advantages_{tag}.parquet"
    if not advantage_path.is_file():
        raise FileNotFoundError(
            f"Forced-true advantage sidecar is missing: {advantage_path}"
        )
    advantage = pq.read_table(advantage_path, columns=["advantage"])
    if advantage.column("advantage").null_count:
        raise ValueError("Advantage sidecar contains null labels")
    labels = advantage.column("advantage").to_numpy(zero_copy_only=False)
    if len(labels) != int(info["total_frames"]):
        raise ValueError(
            "Advantage sidecar must contain exactly one label per dataset frame"
        )
    if (
        bool(config["data"].get("require_all_true", False))
        and not np.asarray(labels).all()
    ):
        raise ValueError("This stage requires every advantage label to be True")

    conditioning = config.get("advantage_conditioning", {})
    if conditioning.get("mode") != "positive_only":
        raise ValueError("Only advantage_conditioning.mode=positive_only is supported")
    probability = float(conditioning.get("unconditional_probability", -1.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "advantage_conditioning.unconditional_probability must be in [0, 1]"
        )

    required_features = {
        "observation.state",
        "observation.images.image",
        "observation.images.image2",
        "action",
    }
    missing = required_features - set(info["features"])
    if missing:
        raise KeyError(
            f"Positive dataset is missing training features: {sorted(missing)}"
        )
    return config, export


class _NativeAdvantageDataLoader:
    """Yield native observations together with frame-aligned advantage labels."""

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
    openpi_config = get_openpi_config(
        model_config.openpi.config_name,
        model_path=model_config.model_path,
        batch_size=int(training["batch_size"]),
        repo_id=str(Path(config["paths"]["dataset"]).expanduser().resolve()),
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
    prompt_suffix = str(config["data"]["prompt_suffix"])
    model_inputs = []
    tokenizer_replacements = 0
    for transform in data_config.model_transforms.inputs:
        if isinstance(transform, openpi_transforms.TokenizePrompt):
            transform = TokenizePositiveAdvantageEdited(
                tokenizer=transform.tokenizer,
                positive_prompt_suffix=prompt_suffix,
                discrete_state_input=transform.discrete_state_input,
            )
            tokenizer_replacements += 1
        model_inputs.append(transform)
    if tokenizer_replacements != 1:
        raise RuntimeError(
            "Expected exactly one OpenPI TokenizePrompt transform, got "
            f"{tokenizer_replacements}"
        )
    data_config = dataclasses.replace(
        data_config,
        model_transforms=dataclasses.replace(
            data_config.model_transforms,
            inputs=tuple(model_inputs),
        ),
    )

    raw_dataset = PiperXPolicyFramesEdited(
        config["paths"]["dataset"],
        action_horizon=int(openpi_config.model.action_horizon),
        advantage_path=(
            Path(config["paths"]["dataset"])
            / "meta"
            / f"advantages_{config['data']['advantage_tag']}.parquet"
        ),
        video_tolerance_s=float(config["data"].get("video_tolerance_s", 0.03)),
        video_cache_size=int(config["data"].get("video_cache_size", 32)),
    )
    dataset = openpi_data_loader.transform_dataset(raw_dataset, data_config)
    dataset = AdvantagePreservingDatasetEdited(dataset, raw_dataset.advantages)
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
    projection_prefixes = (
        "action_",
        "time_",
        "state_proj.",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(projection_prefixes):
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
        default=Path(__file__).with_name("positive_policy_edited.yaml"),
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, export = _load_and_validate(args.config.expanduser().resolve())
    if args.max_steps is not None:
        if args.max_steps <= 0:
            parser.error("--max-steps must be positive")
        config["training"]["max_steps"] = args.max_steps
    if args.output_dir is not None:
        config["paths"]["output_dir"] = str(args.output_dir.expanduser().resolve())
    if args.check_only:
        print(json.dumps({"status": "passed", "dataset": export}, indent=2))
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
    model_config.model_path = str(
        Path(config["paths"]["base_pi05_checkpoint"]).expanduser().resolve()
    )
    model_config.openpi_data.norm_stats_path = str(
        Path(config["paths"]["norm_stats"]).expanduser().resolve()
    )
    initial_checkpoint = Path(model_config.model_path)

    print(f"Loading native Pi0.5 from {model_config.model_path}", flush=True)
    model = get_model(model_config, torch_dtype=torch.float32)
    model.configure_lora(
        train_vision=bool(model_config.openpi.lora_train_vision),
        train_action_projections=bool(
            model_config.openpi.lora_train_action_projections
        ),
    )
    model.to(device)
    model.train()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    parameter_groups = _parameter_groups(model)
    if not parameter_groups["vlm_lora"] or not parameter_groups["action_expert_lora"]:
        raise RuntimeError(
            "Both VLM LoRA and action-expert LoRA parameters must be trainable"
        )
    if parameter_groups["other"]:
        raise RuntimeError(
            f"Unexpected trainable parameter count: {parameter_groups['other']}"
        )
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
    progress = tqdm(range(1, max_steps + 1), desc="Advantage Pi0.5 LoRA")
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
            raise FloatingPointError(
                f"Non-finite gradient norm at step {step}: {grad_norm}"
            )
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

    checkpoint = output_dir / "checkpoints" / f"global_step_{max_steps}" / "actor"
    save_native_adapter(
        model,
        checkpoint,
        base_model_path=initial_checkpoint,
        prompt_suffix=str(config["data"]["prompt_suffix"]),
        metadata={
            "dataset": str(Path(config["paths"]["dataset"]).resolve()),
            "initial_checkpoint": str(initial_checkpoint),
            "advantage_conditioning": "positive_only",
            "require_all_true": bool(config["data"].get("require_all_true", False)),
            "unconditional_probability": unconditional_probability,
            "advantage_tag": str(config["data"]["advantage_tag"]),
            "steps": max_steps,
            "parameter_groups": parameter_groups,
        },
    )
    summary = {
        "initial_pi05_checkpoint": model_config.model_path,
        "adapter_checkpoint": str(checkpoint),
        "dataset": str(Path(config["paths"]["dataset"]).resolve()),
        "episodes": int(export["total_episodes"]),
        "frames": int(export["total_frames"]),
        "advantage_labels": (
            "all_true"
            if bool(config["data"].get("require_all_true", False))
            else "from_sidecar"
        ),
        "advantage_conditioning": "positive_only",
        "unconditional_probability": unconditional_probability,
        "prompt_suffix": str(config["data"]["prompt_suffix"]),
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
