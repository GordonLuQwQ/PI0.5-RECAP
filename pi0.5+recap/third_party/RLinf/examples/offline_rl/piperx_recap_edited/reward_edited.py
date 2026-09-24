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

"""EDITED RECAP reward, return normalization, and categorical targets.

The constants intentionally match RLinf's RECAP example:

* ordinary step: ``-1``
* successful terminal step: ``0``
* failed terminal step: ``-300``
* discount: ``1``
* normalized support: ``[-1, 0]`` with 201 atoms
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


@dataclass(frozen=True)
class ReCapRewardSpecEdited:
    """Reward and distribution settings copied from the RECAP example."""

    step_reward: float = -1.0
    success_terminal_reward: float = 0.0
    failure_terminal_reward: float = -300.0
    gamma: float = 1.0
    num_bins: int = 201
    v_min: float = -1.0
    v_max: float = 0.0

    def __post_init__(self) -> None:
        if self.num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if self.v_max <= self.v_min:
            raise ValueError("v_max must be greater than v_min")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")

    @property
    def bin_width(self) -> float:
        """Distance between adjacent categorical atoms (0.005 by default)."""
        return (self.v_max - self.v_min) / (self.num_bins - 1)


def episode_rewards(
    num_frames: int,
    *,
    success: bool,
    spec: ReCapRewardSpecEdited,
) -> np.ndarray:
    """Build per-frame rewards for one episode."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    rewards = np.full(num_frames, spec.step_reward, dtype=np.float32)
    rewards[-1] = (
        spec.success_terminal_reward if success else spec.failure_terminal_reward
    )
    return rewards


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute ``G_t = r_t + gamma * G_(t+1)`` from the episode end."""
    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.ndim != 1 or rewards.size == 0:
        raise ValueError("rewards must be a non-empty 1-D array")
    returns = np.empty_like(rewards)
    running = np.float32(0.0)
    for index in range(rewards.size - 1, -1, -1):
        running = rewards[index] + np.float32(gamma) * running
        returns[index] = running
    return returns


def normalize_return_minus_one_zero(
    raw_return: float | np.ndarray | torch.Tensor,
    *,
    global_return_min: float,
    clip: bool = False,
):
    """Match RLinf ``ReturnNormalizer(normalize_to_minus_one_zero=True)``.

    With ``return_max == 0``, RLinf computes ``raw / abs(global_return_min)``.
    The minimum return becomes -1 and terminal success return 0 stays 0.
    """
    if global_return_min >= 0:
        raise ValueError("global_return_min must be negative")
    value = raw_return / abs(global_return_min)
    if not clip:
        return value
    if isinstance(value, torch.Tensor):
        return value.clamp(-1.0, 0.0)
    return np.clip(value, -1.0, 0.0)


def categorical_target_distribution(
    target_values: torch.Tensor,
    *,
    spec: ReCapRewardSpecEdited,
) -> torch.Tensor:
    """Linearly project scalar targets onto the two nearest value atoms."""
    values = target_values.float().view(-1).clamp(spec.v_min, spec.v_max)
    positions = (values - spec.v_min) / spec.bin_width
    lower = positions.floor().long().clamp(0, spec.num_bins - 1)
    upper = positions.ceil().long().clamp(0, spec.num_bins - 1)

    upper_weight = positions - lower.float()
    lower_weight = 1.0 - upper_weight
    same_atom = lower == upper
    lower_weight = torch.where(same_atom, torch.ones_like(lower_weight), lower_weight)
    upper_weight = torch.where(same_atom, torch.zeros_like(upper_weight), upper_weight)

    distribution = torch.zeros(
        values.shape[0], spec.num_bins, dtype=torch.float32, device=values.device
    )
    distribution.scatter_add_(1, lower[:, None], lower_weight[:, None])
    distribution.scatter_add_(1, upper[:, None], upper_weight[:, None])
    return distribution


def categorical_value_loss(
    logits: torch.Tensor,
    target_values: torch.Tensor,
    *,
    spec: ReCapRewardSpecEdited,
) -> torch.Tensor:
    """Cross entropy against the projected categorical target distribution."""
    if logits.ndim != 2 or logits.shape[1] != spec.num_bins:
        raise ValueError(
            f"logits must have shape [B, {spec.num_bins}], got {tuple(logits.shape)}"
        )
    targets = categorical_target_distribution(target_values, spec=spec)
    return -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def expected_value_from_logits(
    logits: torch.Tensor,
    *,
    spec: ReCapRewardSpecEdited,
) -> torch.Tensor:
    """Convert the 201-bin distribution to its expected scalar value."""
    atoms = torch.linspace(
        spec.v_min,
        spec.v_max,
        spec.num_bins,
        dtype=torch.float32,
        device=logits.device,
    )
    return (torch.softmax(logits.float(), dim=-1) * atoms).sum(dim=-1)
