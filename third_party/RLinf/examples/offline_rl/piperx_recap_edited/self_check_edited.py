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

"""EDITED lightweight checks for reward and categorical-value math."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from reward_edited import (
    ReCapRewardSpecEdited,
    categorical_target_distribution,
    discounted_returns,
    episode_rewards,
    expected_value_from_logits,
    normalize_return_minus_one_zero,
)
from train_value_edited import (
    _checkpoint_due,
    _OffsetSampler,
    _restore_training_checkpoint,
    _save_training_checkpoint,
)


class _TinyValueModel(torch.nn.Module):
    """Small stand-in that exercises the value trainer checkpoint contract."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def save_value_head(self, output_file: str | Path, *, base_checkpoint: str) -> None:
        torch.save(
            {"weight": self.weight.detach().cpu(), "base": base_checkpoint},
            output_file,
        )

    def load_value_head(self, checkpoint_file: str | Path) -> None:
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        with torch.no_grad():
            self.weight.copy_(payload["weight"])


def _check_training_checkpoints() -> None:
    checkpoint_steps = frozenset({3000, 6000})
    assert not _checkpoint_due(2999, checkpoint_steps)
    assert _checkpoint_due(3000, checkpoint_steps)
    assert _checkpoint_due(6000, checkpoint_steps)
    assert not _checkpoint_due(9000, checkpoint_steps)
    assert not _checkpoint_due(9356, checkpoint_steps)

    sampler = _OffsetSampler(torch.utils.data.SequentialSampler(range(10)), 3)
    assert len(sampler) == 7
    assert list(sampler) == list(range(3, 10))

    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        model = _TinyValueModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.weight.square().sum().backward()
        optimizer.step()
        expected_weight = model.weight.detach().clone()
        checkpoint_dir = _save_training_checkpoint(
            output_dir=output_dir,
            step=3000,
            model=model,
            optimizer=optimizer,
            base_checkpoint="tiny-base",
            batch_size=112,
            total_frames=374226,
        )

        resumed_model = _TinyValueModel()
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
        step = _restore_training_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=resumed_model,
            optimizer=resumed_optimizer,
            device=torch.device("cpu"),
            batch_size=112,
            total_frames=374226,
        )
        assert step == 3000
        torch.testing.assert_close(resumed_model.weight, expected_weight)
        assert resumed_optimizer.state_dict()["state"]


def main() -> None:
    spec = ReCapRewardSpecEdited()
    success = discounted_returns(
        episode_rewards(600, success=True, spec=spec), spec.gamma
    )
    failure = discounted_returns(
        episode_rewards(600, success=False, spec=spec), spec.gamma
    )
    assert success[0] == -599.0 and success[-1] == 0.0
    assert failure[0] == -899.0 and failure[-1] == -300.0

    normalized = normalize_return_minus_one_zero(
        np.asarray([-899.0, -599.0, 0.0]), global_return_min=-899.0
    )
    np.testing.assert_allclose(
        normalized, np.asarray([-1.0, -599.0 / 899.0, 0.0]), atol=1e-7
    )

    targets = torch.tensor([-1.0, -0.9975, 0.0])
    projected = categorical_target_distribution(targets, spec=spec)
    torch.testing.assert_close(projected.sum(-1), torch.ones(3))
    torch.testing.assert_close(projected[1, :2], torch.tensor([0.5, 0.5]))

    logits = torch.full((3, spec.num_bins), -100.0)
    logits[0, 0] = 100.0
    logits[1, 100] = 100.0
    logits[2, -1] = 100.0
    values = expected_value_from_logits(logits, spec=spec)
    torch.testing.assert_close(values, torch.tensor([-1.0, -0.5, 0.0]))

    _check_training_checkpoints()

    print("EDITED RECAP math check passed")
    print(f"success G_0={success[0]:.0f}, failure G_0={failure[0]:.0f}")
    print(
        f"bins={spec.num_bins}, bin_width={spec.bin_width:.3f}, support=[{spec.v_min}, {spec.v_max}]"
    )
    print(
        "checkpoint interval, atomic save, optimizer restore, and sampler offset passed"
    )


if __name__ == "__main__":
    main()
