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

"""EDITED categorical value critic built on the trained PiperX Pi0.5 VLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from reward_edited import (
    ReCapRewardSpecEdited,
    categorical_value_loss,
    expected_value_from_logits,
)
from torch import Tensor, nn


@dataclass
class Pi05ValueOutputEdited:
    """Outputs used for value training and RECAP advantage computation."""

    values: Tensor
    logits: Tensor
    loss: Tensor | None = None


class Pi05VlmValueCriticEdited(nn.Module):
    """Use the trained Pi0.5 prefix VLM and learn a fresh categorical head.

    The wrapped Pi0.5 is a separate model instance loaded from the current
    fine-tuned checkpoint. Its action expert is present in memory because the
    dual-expert checkpoint stores one joint module, but value prediction never
    calls ``run_suffix`` and therefore never executes the action expert.
    """

    def __init__(
        self,
        pi05,
        *,
        reward_spec: ReCapRewardSpecEdited,
        hidden_size: int = 2048,
        head_hidden_size: int = 1024,
        dropout: float = 0.0,
        pooling: str = "mean_token",
        train_pi05_vlm_lora: bool = True,
        train_vision_encoder: bool = False,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if pooling not in {"mean_token", "first_token", "last_token"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        self.pi05 = pi05
        self.reward_spec = reward_spec
        self.pooling = pooling
        self.train_pi05_vlm_lora = bool(train_pi05_vlm_lora)
        self.train_vision_encoder = bool(train_vision_encoder)
        self.train_pi05_vlm = self.train_pi05_vlm_lora or self.train_vision_encoder

        # EDITED: 201 logits over [-1, 0], rather than the scalar PPO head.
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, reward_spec.num_bins),
        )
        self._configure_pi05_parameters(gradient_checkpointing=gradient_checkpointing)

    def _configure_pi05_parameters(self, *, gradient_checkpointing: bool) -> None:
        """Train VLM LoRA/vision only; always freeze the action expert."""
        from rlinf.models.embodiment.openpi_rlinf.modules.lora import (
            is_lora_parameter,
        )

        for parameter in self.pi05.parameters():
            parameter.requires_grad_(False)

        if self.train_pi05_vlm_lora:
            trainable_lora = 0
            for block in self.pi05.llm.layers:
                attention = block.attn
                for projections in (
                    attention.q_proj,
                    attention.k_proj,
                    attention.v_proj,
                    attention.o_proj,
                ):
                    for name, parameter in projections[0].named_parameters():
                        if is_lora_parameter(name):
                            parameter.requires_grad_(True)
                            trainable_lora += parameter.numel()
                for name, parameter in block.mlps[0].named_parameters():
                    if is_lora_parameter(name):
                        parameter.requires_grad_(True)
                        trainable_lora += parameter.numel()
            if trainable_lora == 0:
                raise RuntimeError(
                    "No PaliGemma expert-0 LoRA parameters were found in Pi0.5"
                )

        if self.train_vision_encoder:
            for parameter in self.pi05.img.parameters():
                parameter.requires_grad_(True)

        if self.train_pi05_vlm and gradient_checkpointing:
            self.pi05.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        elif hasattr(self.pi05, "gradient_checkpointing_disable"):
            self.pi05.gradient_checkpointing_disable()

        self.pi05.train(self.training and self.train_pi05_vlm)

    def train(self, mode: bool = True):
        """Train only the configured VLM subset and categorical value head."""
        super().train(mode)
        self.pi05.train(mode and self.train_pi05_vlm)
        return self

    @staticmethod
    def _pool_prefix(
        prefix_output: Tensor,
        prefix_mask: Tensor,
        mode: str,
    ) -> Tensor:
        mask = prefix_mask.bool()
        if mode == "mean_token":
            weights = mask.to(prefix_output.dtype).unsqueeze(-1)
            return (prefix_output * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        if mode == "first_token":
            indices = mask.long().argmax(dim=1)
        else:
            indices = (
                mask.shape[1] - 1 - torch.flip(mask, dims=[1]).long().argmax(dim=1)
            )
        rows = torch.arange(prefix_output.shape[0], device=prefix_output.device)
        return prefix_output[rows, indices]

    def encode_observation(self, observation) -> Tensor:
        """Return one 2048-D trained-Pi0.5 VLM feature per observation."""
        from rlinf.models.embodiment.openpi_rlinf.modules.model import (
            preprocess_observation,
        )

        observation = preprocess_observation(observation, train=False)
        context = torch.enable_grad() if self.train_pi05_vlm else torch.no_grad()
        with context:
            prefix_output, prefix_mask, _ = self.pi05.build_prefix_cache(observation)
            pooled = self._pool_prefix(prefix_output, prefix_mask, self.pooling)
        return pooled.float()

    def forward(
        self,
        observation,
        target_values: Tensor | None = None,
    ) -> Pi05ValueOutputEdited:
        """Predict a categorical value distribution and optional training loss."""
        features = self.encode_observation(observation)
        head_parameter = next(self.value_head.parameters())
        logits = self.value_head(
            features.to(device=head_parameter.device, dtype=head_parameter.dtype)
        )
        values = expected_value_from_logits(logits, spec=self.reward_spec)
        loss = None
        if target_values is not None:
            loss = categorical_value_loss(
                logits,
                target_values.to(logits.device),
                spec=self.reward_spec,
            )
        return Pi05ValueOutputEdited(values=values, logits=logits, loss=loss)

    def save_value_head(self, output_file: str | Path, *, base_checkpoint: str) -> None:
        """Save the value head and tuned VLM adapter without dense Pi0.5 weights."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pi05_adapter = {
            name: parameter.detach().cpu()
            for name, parameter in self.pi05.named_parameters()
            if parameter.requires_grad
        }
        torch.save(
            {
                "format": "piperx_pi05_recap_value_edited_v2",
                "base_pi05_checkpoint": base_checkpoint,
                "pooling": self.pooling,
                "reward_spec": self.reward_spec.__dict__,
                "pi05_value_adapter": pi05_adapter,
                "value_head": self.value_head.state_dict(),
            },
            output_file,
        )

    def load_value_head(self, checkpoint_file: str | Path) -> None:
        """Load a value head and reject a mismatched categorical support."""
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        if payload.get("reward_spec") != self.reward_spec.__dict__:
            raise ValueError(
                "Value-head reward/bin settings do not match the current config: "
                f"saved={payload.get('reward_spec')}, current={self.reward_spec.__dict__}"
            )
        if payload.get("pooling") != self.pooling:
            raise ValueError(
                f"Value-head pooling mismatch: saved={payload.get('pooling')}, "
                f"current={self.pooling}"
            )
        named_parameters = dict(self.pi05.named_parameters())
        with torch.no_grad():
            for name, tensor in payload.get("pi05_value_adapter", {}).items():
                if name not in named_parameters:
                    raise KeyError(f"Saved Pi0.5 adapter parameter is absent: {name}")
                parameter = named_parameters[name]
                if parameter.shape != tensor.shape:
                    raise ValueError(
                        f"Pi0.5 adapter shape mismatch for {name}: "
                        f"saved={tuple(tensor.shape)}, current={tuple(parameter.shape)}"
                    )
                parameter.copy_(tensor.to(parameter.device, parameter.dtype))
        self.value_head.load_state_dict(payload["value_head"], strict=True)
