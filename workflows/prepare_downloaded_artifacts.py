#!/usr/bin/env python3
"""Relocate downloaded PiperX adapters and the value checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.checkpoint_root.expanduser().resolve()
    dense_base = root / "finetune_with_collectedD_ckpt"
    dense_weights = dense_base / "model_state_dict" / "full_weights.pt"
    if not dense_weights.is_file():
        candidates = list(dense_base.glob("**/model_state_dict/full_weights.pt"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one released 600-demo dense checkpoint under {dense_base}, "
                f"found {len(candidates)}. Place it at {dense_weights}."
            )
        actor_dir = candidates[0].parent.parent
        (actor_dir / "model_state_dict").rename(dense_base / "model_state_dict")
        if (actor_dir / "dcp_checkpoint").is_dir():
            (actor_dir / "dcp_checkpoint").rename(dense_base / "dcp_checkpoint")

    adapter_dirs = [
        root
        / "finetune_vla_advantage_TRUE_ckpt"
        / "checkpoints"
        / "global_step_3000"
        / "actor",
        root
        / "final_stage_finetune_ckpt"
        / "checkpoints"
        / "global_step_5000"
        / "actor",
    ]
    updated_manifests: list[str] = []
    for adapter_dir in adapter_dirs:
        manifest_path = adapter_dir / "adapter_config.json"
        weights_path = adapter_dir / "adapter_weights.safetensors"
        if not manifest_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"Incomplete adapter checkpoint: {adapter_dir}")
        manifest = json.loads(manifest_path.read_text())
        manifest["base_model_path"] = str(dense_base)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        updated_manifests.append(str(manifest_path))

    import torch

    value_checkpoint = (
        root
        / "recap_vlm_advantage_indicator_ckpt"
        / "checkpoints"
        / "step_003000"
        / "pi05_value.pt"
    )
    if not value_checkpoint.is_file():
        raise FileNotFoundError(f"Missing value checkpoint: {value_checkpoint}")
    payload = torch.load(value_checkpoint, map_location="cpu", weights_only=True)
    payload["base_pi05_checkpoint"] = str(dense_base)
    temporary = value_checkpoint.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(value_checkpoint)

    print(
        json.dumps(
            {
                "checkpoint_root": str(root),
                "dense_base": str(dense_base),
                "updated_adapter_manifests": updated_manifests,
                "updated_value_checkpoint": str(value_checkpoint),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
