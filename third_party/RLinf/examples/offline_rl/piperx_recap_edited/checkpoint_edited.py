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

"""EDITED loader for the already fine-tuned PiperX Pi0.5 checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def build_pi05_eval_config(config: dict[str, Any]):
    """Build the exact model shape used by the existing PiperX checkpoint."""
    paths = config["paths"]
    checkpoint = Path(paths["pi05_checkpoint"]).expanduser()
    norm_stats = Path(paths["norm_stats"]).expanduser()
    full_weights = checkpoint / "model_state_dict" / "full_weights.pt"
    if not full_weights.is_file():
        raise FileNotFoundError(f"Pi0.5 full weights not found: {full_weights}")
    if not norm_stats.is_file():
        raise FileNotFoundError(f"PiperX norm stats not found: {norm_stats}")

    # These fields mirror piperx_sft_openpi_pi05_rlinf.yaml. In particular,
    # the *_lora variants are required to reconstruct the checkpoint shape.
    return OmegaConf.create(
        {
            "model_type": "openpi_rlinf",
            "model_path": str(checkpoint),
            "precision": "bf16",
            "pi05": True,
            "is_lora": True,
            "lora_rank": 32,
            "use_proprio": True,
            "num_action_chunks": 50,
            "action_dim": 7,
            "num_steps": 10,
            "add_value_head": False,
            "openpi": {
                "task": "eval",
                "config_name": "pi05_piperx",
                "action_horizon": 50,
                "action_chunk": 50,
                "action_env_dim": 7,
                "model_action_dim": 32,
                "max_token_len": 200,
                "discrete_state_input": True,
                "paligemma_variant": "gemma_2b_lora",
                "action_expert_variant": "gemma_300m_lora",
                "lora_train_vision": True,
                "lora_train_action_projections": True,
                "num_images_in_input": 2,
                "rtc_enabled": False,
            },
            "openpi_data": {"norm_stats_path": str(norm_stats)},
        }
    )


def load_trained_pi05(config: dict[str, Any]):
    """Load a second Pi0.5 instance whose VLM will serve as the value backbone."""
    from rlinf.models.embodiment.openpi_rlinf import get_model

    model = get_model(build_pi05_eval_config(config))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
