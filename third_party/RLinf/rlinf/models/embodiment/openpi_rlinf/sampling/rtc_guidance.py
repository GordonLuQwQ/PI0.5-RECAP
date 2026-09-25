# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RTC overlap guidance for the vendored Pi0 eval sampler."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rlinf.models.embodiment.openpi_rlinf.modules.model import preprocess_observation
from rlinf.models.embodiment.openpi_rlinf.sampling import rl_sampler


@dataclass
class RTCGuidanceContext:

    prev_model_actions: torch.Tensor
    executed_horizon: int
    delay_steps: int

    def get_prev_remaining(self) -> torch.Tensor:
        """Return the previous chunk after actions already executed."""
        return self.prev_model_actions[:, self.executed_horizon :, :]


def build_rtc_target_and_mask(
    prev_remaining: torch.Tensor,
    horizon: int,
    action_dim: int,
    delay_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the previous chunk and build the RTC overlap mask."""
    batch_size = prev_remaining.shape[0]
    target = torch.zeros((batch_size, horizon, action_dim), device=device, dtype=dtype)
    mask = torch.zeros((batch_size, horizon, 1), device=device, dtype=dtype)

    overlap = min(prev_remaining.shape[1], horizon)
    target[:, :overlap] = prev_remaining[:, :overlap].to(device=device, dtype=dtype)
    hard_end = min(max(delay_steps, 0), overlap)

    mask[:, :hard_end, 0] = 1.0

    if hard_end < overlap:
        indices = torch.arange(hard_end, overlap, device=device, dtype=dtype)
        denominator = float(overlap - hard_end + 1)
        c_i = (overlap - indices) / denominator
        soft_weights = c_i * torch.expm1(c_i) / (math.e - 1.0)
        mask[:, hard_end:overlap, 0] = soft_weights

    return target, mask


def exact_guidance_weight(
    paper_tau: torch.Tensor,
    guidance_clip: float,
) -> torch.Tensor:
    """Compute the clipped guidance coefficient from the RTC equations."""
    one_minus_tau = 1.0 - paper_tau
    denominator = (paper_tau * one_minus_tau).clamp_min(1e-6)
    numerator = paper_tau.square() + one_minus_tau.square()
    return (numerator / denominator).clamp(max=float(guidance_clip))


def exact_rtc_velocity(
    pi0_model,
    observation,
    x_t: torch.Tensor,
    model_t: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kv_cache: tuple,
    prefix_mask: torch.Tensor,
    guidance_clip: float,
) -> torch.Tensor:
    """Return the Pi0-time velocity after exact RTC guidance."""
    batch_size = x_t.shape[0]
    paper_tau = 1.0 - model_t
    time_batch = model_t.expand(batch_size).to(dtype=torch.float32)

    with torch.enable_grad():
        x_t_grad = x_t.detach().to(torch.float32).requires_grad_(True)
        suffix_out = pi0_model.run_suffix(
            observation,
            x_t_grad,
            time_batch,
            kv_cache,
            prefix_mask,
        )
        raw_velocity = pi0_model.velocity_from_suffix(suffix_out).to(torch.float32)


        action_hat = x_t_grad - model_t * raw_velocity
        error = ((target - action_hat) * mask).detach()

        pinv_correction = torch.autograd.grad(
            outputs=action_hat,
            inputs=x_t_grad,
            grad_outputs=error,
            create_graph=False,
            retain_graph=False,
        )[0]

        guidance_weight = exact_guidance_weight(paper_tau, guidance_clip)
        guided_velocity = raw_velocity - guidance_weight * pinv_correction

    return guided_velocity.detach()


@torch.no_grad()
def sample_actions_with_rtc_guidance(
    pi0_model,
    observation,
    rtc_context: RTCGuidanceContext,
    *,
    num_steps: int,
    noise: torch.Tensor | None = None,
    rng: torch.Generator | None = None,
    guidance_clip: float = 3.0,
) -> torch.Tensor:
    """Euler ODE sampling with overlap guidance against the previous chunk."""
    observation = preprocess_observation(observation, train=False)
    B = observation.state.shape[0]
    device = observation.state.device
    if noise is None:
        noise = torch.randn(
            B,
            pi0_model.action_horizon,
            pi0_model.action_dim,
            device=device,
            dtype=torch.float32,
            generator=rng,
        )
    else:
        noise = noise.to(device=device, dtype=torch.float32)

    prefix_out, prefix_mask, kv_cache = pi0_model.build_prefix_cache(observation)
    del prefix_out

    prev_remaining = rtc_context.get_prev_remaining()
    target, mask = build_rtc_target_and_mask(
        prev_remaining=prev_remaining,
        horizon=pi0_model.action_horizon,
        action_dim=pi0_model.action_dim,
        delay_steps=rtc_context.delay_steps,
        device=device,
        dtype=noise.dtype,
    )

    x_t = noise
    timesteps = rl_sampler.get_timesteps(num_steps, device).to(dtype=torch.float32)
    step_indices = torch.empty(B, device=device, dtype=torch.long)

    for index in range(num_steps):
        model_t = timesteps[index]
        step_indices.fill_(index)

        velocity = exact_rtc_velocity(
            pi0_model=pi0_model,
            observation=observation,
            x_t=x_t,
            model_t=model_t,
            target=target,
            mask=mask,
            kv_cache=kv_cache,
            prefix_mask=prefix_mask,
            guidance_clip=guidance_clip,
        )

        x_t_mean, _ = rl_sampler.sample_mean_var(
            x_t.to(torch.float32),
            velocity,
            step_indices,
            noise_method="flow_ode",
            noise_level=0.0,
            num_steps=num_steps,
        )
        x_t = x_t_mean.detach()

    return x_t
