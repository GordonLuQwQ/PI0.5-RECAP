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

"""Train only the EDITED categorical value head on the trained Pi0.5 VLM."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from checkpoint_edited import load_trained_pi05
from data_edited import (
    build_datasets_edited,
    build_weighted_sampler_edited,
    collate_value_frames_edited,
    concatenate_datasets_edited,
)
from pi05_value_critic_edited import Pi05VlmValueCriticEdited
from reward_edited import ReCapRewardSpecEdited, normalize_return_minus_one_zero
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_TRAINER_CHECKPOINT_FORMAT = "piperx_pi05_recap_value_trainer_edited_v1"


class _OffsetSampler(Sampler[int]):
    """Skip samples already consumed by a resumed deterministic sampler."""

    def __init__(self, sampler: Sampler[int], offset: int) -> None:
        if offset < 0 or offset > len(sampler):
            raise ValueError(
                f"Sampler offset must be in [0, {len(sampler)}], got {offset}"
            )
        self.sampler = sampler
        self.offset = offset

    def __iter__(self):
        return itertools.islice(iter(self.sampler), self.offset, None)

    def __len__(self) -> int:
        return len(self.sampler) - self.offset


def _checkpoint_due(step: int, checkpoint_steps: frozenset[int]) -> bool:
    """Return whether an intermediate checkpoint is due after this step."""
    return step in checkpoint_steps


def _resolve_resume_directory(output_dir: Path, resume_from: str | None) -> Path | None:
    """Resolve an explicit or latest checkpoint directory."""
    if resume_from is None or str(resume_from).strip().lower() in {"", "none", "false"}:
        return None
    if str(resume_from).strip().lower() == "auto":
        latest_path = output_dir / "latest_checkpoint.json"
        if not latest_path.is_file():
            return None
        latest = json.loads(latest_path.read_text())
        checkpoint_dir = output_dir / str(latest["checkpoint_dir"])
    else:
        checkpoint_dir = Path(str(resume_from)).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Value checkpoint directory not found: {checkpoint_dir}"
        )
    for filename in ("pi05_value.pt", "trainer_state.pt", "metadata.json"):
        if not (checkpoint_dir / filename).is_file():
            raise FileNotFoundError(
                f"Incomplete value checkpoint: {checkpoint_dir / filename}"
            )
    return checkpoint_dir


def _save_training_checkpoint(
    *,
    output_dir: Path,
    step: int,
    model: Pi05VlmValueCriticEdited,
    optimizer: torch.optim.Optimizer,
    base_checkpoint: str,
    batch_size: int,
    total_frames: int,
) -> Path:
    """Atomically save a value adapter and resumable optimizer state."""
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoints_dir / f"step_{step:06d}"
    temporary_dir = checkpoints_dir / f".step_{step:06d}.tmp"
    if checkpoint_dir.exists():
        raise FileExistsError(f"Value checkpoint already exists: {checkpoint_dir}")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir()

    model.save_value_head(
        temporary_dir / "pi05_value.pt", base_checkpoint=base_checkpoint
    )
    torch.save(
        {
            "format": _TRAINER_CHECKPOINT_FORMAT,
            "step": step,
            "batch_size": batch_size,
            "total_frames": total_frames,
            "optimizer": optimizer.state_dict(),
        },
        temporary_dir / "trainer_state.pt",
    )
    metadata = {
        "format": _TRAINER_CHECKPOINT_FORMAT,
        "step": step,
        "batch_size": batch_size,
        "total_frames": total_frames,
        "model_checkpoint": "pi05_value.pt",
        "trainer_state": "trainer_state.pt",
    }
    (temporary_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    temporary_dir.rename(checkpoint_dir)

    latest = {
        "format": _TRAINER_CHECKPOINT_FORMAT,
        "step": step,
        "checkpoint_dir": str(checkpoint_dir.relative_to(output_dir)),
    }
    latest_temporary = output_dir / ".latest_checkpoint.json.tmp"
    latest_temporary.write_text(json.dumps(latest, indent=2) + "\n")
    latest_temporary.replace(output_dir / "latest_checkpoint.json")
    print(json.dumps({"saved_value_checkpoint": str(checkpoint_dir), "step": step}))
    return checkpoint_dir


def _restore_training_checkpoint(
    *,
    checkpoint_dir: Path,
    model: Pi05VlmValueCriticEdited,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    total_frames: int,
) -> int:
    """Restore the value adapter and optimizer, returning the completed step."""
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
    state = torch.load(
        checkpoint_dir / "trainer_state.pt",
        map_location=device,
        weights_only=True,
    )
    for payload_name, payload in (("metadata", metadata), ("trainer state", state)):
        if payload.get("format") != _TRAINER_CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unexpected {payload_name} format: {payload.get('format')!r}"
            )
        if int(payload["batch_size"]) != batch_size:
            raise ValueError(
                f"Cannot resume batch size {payload['batch_size']} checkpoint with "
                f"batch size {batch_size}"
            )
        if int(payload["total_frames"]) != total_frames:
            raise ValueError(
                f"Cannot resume checkpoint for {payload['total_frames']} frames with "
                f"dataset containing {total_frames} frames"
            )
    if int(metadata["step"]) != int(state["step"]):
        raise ValueError("Checkpoint metadata and trainer state disagree on step")

    model.load_value_head(checkpoint_dir / "pi05_value.pt")
    optimizer.load_state_dict(state["optimizer"])
    step = int(state["step"])
    print(json.dumps({"resumed_value_checkpoint": str(checkpoint_dir), "step": step}))
    return step


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    rollout_paths = [
        entry.get("path")
        for entry in config["data"]["datasets"]
        if entry.get("type") == "rollout"
    ]
    if config["data"].get("require_rollout_dataset", False) and not any(rollout_paths):
        raise ValueError(
            "Set the rollout dataset path in config_edited.yaml. Training only on "
            "successful SFT trajectories cannot teach the value model to recognize failure."
        )
    return config


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_edited.yaml"),
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    training = config["training"]
    value_config = config["value_model"]
    seed = int(training["seed"])
    _set_seed(seed)

    device_name = str(training.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    spec = _reward_spec(config)

    datasets = build_datasets_edited(config)
    computed_min = min(float(dataset.raw_returns.min()) for dataset in datasets)
    computed_max = max(float(dataset.raw_returns.max()) for dataset in datasets)
    normalization = config["normalization"]
    global_return_min = normalization.get("return_min")
    global_return_min = (
        computed_min if global_return_min is None else float(global_return_min)
    )
    global_return_max = float(normalization.get("return_max", computed_max))
    if global_return_min >= 0 or global_return_max != 0.0:
        raise ValueError(
            "EDITED RECAP normalization requires a negative global minimum and "
            f"return_max=0; got [{global_return_min}, {global_return_max}]"
        )

    batch_size = int(training["batch_size"])
    max_steps = int(training["max_steps"])
    checkpoint_steps = frozenset(
        int(step) for step in training.get("checkpoint_steps", [])
    )
    if any(step <= 0 or step >= max_steps for step in checkpoint_steps):
        raise ValueError(
            "Every training.checkpoint_steps entry must be greater than zero and "
            f"smaller than max_steps={max_steps}"
        )
    total_frames = sum(len(dataset) for dataset in datasets)
    effective_epochs = max_steps * batch_size / total_frames
    print(
        json.dumps(
            {
                "training_schedule": {
                    "frames": total_frames,
                    "batch_size": batch_size,
                    "max_steps": max_steps,
                    "sampled_frames": max_steps * batch_size,
                    "effective_epochs": effective_epochs,
                    "sampling": "weighted_random_with_replacement",
                }
            },
            indent=2,
        )
    )

    output_dir = Path(config["paths"]["output_dir"]).expanduser().resolve()
    resume_directory = _resolve_resume_directory(
        output_dir, training.get("resume_from")
    )

    print(
        f"Loading trained Pi0.5 value backbone from {config['paths']['pi05_checkpoint']}"
    )
    pi05 = load_trained_pi05(config).to(device)
    model = Pi05VlmValueCriticEdited(
        pi05,
        reward_spec=spec,
        hidden_size=int(value_config["hidden_size"]),
        head_hidden_size=int(value_config["head_hidden_size"]),
        dropout=float(value_config["dropout"]),
        pooling=str(value_config["pooling"]),
        train_pi05_vlm_lora=bool(value_config["train_pi05_vlm_lora"]),
        train_vision_encoder=bool(value_config["train_vision_encoder"]),
        gradient_checkpointing=bool(value_config["gradient_checkpointing"]),
    ).to(device)
    model.train()
    head_parameters = list(model.value_head.parameters())
    vlm_parameters = [
        parameter for parameter in model.pi05.parameters() if parameter.requires_grad
    ]
    parameters = head_parameters + vlm_parameters
    parameter_groups = [
        {"params": head_parameters, "lr": float(training["learning_rate"])}
    ]
    if vlm_parameters:
        parameter_groups.append(
            {
                "params": vlm_parameters,
                "lr": float(training["vlm_learning_rate"]),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(training["weight_decay"]),
    )

    start_step = 0
    if resume_directory is not None:
        start_step = _restore_training_checkpoint(
            checkpoint_dir=resume_directory,
            model=model,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            total_frames=total_frames,
        )
    if start_step >= max_steps:
        raise ValueError(
            f"Resume checkpoint step {start_step} must be smaller than max_steps "
            f"{max_steps}"
        )

    sampler: Sampler[int] = build_weighted_sampler_edited(
        datasets,
        config["data"]["datasets"],
        num_samples=max_steps * batch_size,
        seed=seed,
    )
    if start_step:
        sampler = _OffsetSampler(sampler, start_step * batch_size)
    loader = DataLoader(
        concatenate_datasets_edited(datasets),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(training.get("num_workers", 0)),
        collate_fn=collate_value_frames_edited,
        pin_memory=bool(training.get("pin_memory", True)),
        drop_last=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / "tensorboard")
    running_loss = 0.0
    progress = tqdm(
        loader,
        total=max_steps,
        initial=start_step,
        desc="Pi0.5 value head",
    )
    for step, batch in enumerate(progress, start=start_step + 1):
        observation = pi05.env_obs_to_observation(
            {
                "main_images": batch["main_images"],
                "wrist_images": batch["wrist_images"],
                "states": batch["states"],
                "task_descriptions": batch["task_descriptions"],
            }
        )
        targets = normalize_return_minus_one_zero(
            batch["raw_returns"],
            global_return_min=global_return_min,
            clip=bool(normalization.get("clip", False)),
        ).to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(observation, target_values=targets)
        if output.loss is None:
            raise RuntimeError("Value model did not return a training loss")
        if not torch.isfinite(output.loss):
            raise FloatingPointError(
                f"Non-finite value loss at step {step}: {output.loss}"
            )
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(training["grad_clip_norm"])
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite value gradient norm at step {step}: {grad_norm}"
            )
        optimizer.step()

        loss_value = float(output.loss.detach())
        running_loss += loss_value
        writer.add_scalar("train/loss", loss_value, step)
        writer.add_scalar("train/grad_norm", float(grad_norm), step)
        if step % int(training["log_every"]) == 0:
            average = running_loss / int(training["log_every"])
            progress.set_postfix(loss=f"{average:.5f}")
            running_loss = 0.0
        if _checkpoint_due(step, checkpoint_steps):
            writer.flush()
            _save_training_checkpoint(
                output_dir=output_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                base_checkpoint=str(config["paths"]["pi05_checkpoint"]),
                batch_size=batch_size,
                total_frames=total_frames,
            )
        if step >= max_steps:
            break

    checkpoint = output_dir / "pi05_value_final.pt"
    model.save_value_head(
        checkpoint,
        base_checkpoint=str(config["paths"]["pi05_checkpoint"]),
    )
    summary = {
        "base_pi05_checkpoint": str(config["paths"]["pi05_checkpoint"]),
        "value_head_checkpoint": str(checkpoint),
        "global_return_min": global_return_min,
        "global_return_max": global_return_max,
        "steps": max_steps,
        "batch_size": batch_size,
        "sampled_frames": max_steps * batch_size,
        "effective_epochs": effective_epochs,
        "checkpoint_steps": sorted(checkpoint_steps),
        "resumed_from_step": start_step,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_value_head_parameters": sum(
            parameter.numel() for parameter in head_parameters
        ),
        "trainable_pi05_vlm_parameters": sum(
            parameter.numel() for parameter in vlm_parameters
        ),
        "reward_spec": spec.__dict__,
        "datasets": [
            {"path": str(dataset.root), "frames": len(dataset)} for dataset in datasets
        ],
    }
    if device.type == "cuda":
        summary["peak_cuda_allocated_gib"] = round(
            torch.cuda.max_memory_allocated(device) / 1024**3, 3
        )
        summary["peak_cuda_reserved_gib"] = round(
            torch.cuda.max_memory_reserved(device) / 1024**3, 3
        )
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    writer.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
