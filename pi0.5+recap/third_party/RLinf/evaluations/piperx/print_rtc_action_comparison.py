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

"""Print matched RTC and unguided action chunks from a paired PiperX eval."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rlinf.models.embodiment.openpi_rlinf.sampling.rtc_guidance import (
    RTCGuidanceContext,
    build_rtc_target_and_mask,
    exact_guidance_weight,
)

ACTION_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_closure",
)


@dataclass(frozen=True)
class ActionComparison:
    """Arrays and metadata needed to explain one RTC replan."""

    call_index: int
    guidance_mode: str
    executed_horizon: int
    delay_steps: int
    overlap: int
    mask: np.ndarray
    target_actions: np.ndarray
    no_rtc_actions: np.ndarray
    rtc_actions: np.ndarray
    final_delta: np.ndarray
    guidance_scales: np.ndarray
    guidance_multipliers: np.ndarray
    no_rtc_model_actions: np.ndarray
    rtc_model_actions: np.ndarray
    target_model_actions: np.ndarray


def _read_json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _load_call(base: Path, call_index: int) -> dict[str, np.ndarray]:
    path = base / "model_calls" / f"{call_index:05d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def _unnormalize_actions(model_actions: np.ndarray, stats: dict) -> np.ndarray:
    """Apply the π0.5 quantile output transform to the seven PiperX actions."""
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    action_dim = len(q01)
    normalized = np.asarray(model_actions, dtype=np.float32)[..., :action_dim]
    return (normalized + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def _guidance_scales(
    num_steps: int, guidance_clip: float, guidance_mode: str
) -> np.ndarray:
    """Return the configured guidance coefficient at each denoising step."""
    paper_tau = torch.tensor(0.0, dtype=torch.float32)
    dt = torch.tensor(1.0 / num_steps, dtype=torch.float32)
    scales = []
    for _ in range(num_steps):
        if guidance_mode == "exact":
            scales.append(exact_guidance_weight(paper_tau, guidance_clip))
        elif guidance_mode == "approx":
            scales.append(
                torch.clamp(
                    (1.0 - paper_tau) / torch.clamp(paper_tau + 1e-4, min=1e-4),
                    max=guidance_clip,
                )
            )
        else:
            raise ValueError(f"Unsupported RTC guidance mode: {guidance_mode!r}")
        paper_tau = paper_tau + dt
    return torch.stack(scales).numpy()


def build_comparison(root: Path, call_index: int) -> ActionComparison:
    """Load one matched replan and reproduce its official RTC mask."""
    if call_index < 1:
        raise ValueError("call_index must be at least 1; bootstrap has no RTC guidance")

    plain_base = root / "no_rtc" / "standard"
    rtc_base = root / "rtc" / "standard"
    plain_meta = _read_json_lines(plain_base / "model_calls.jsonl")
    rtc_meta = _read_json_lines(rtc_base / "model_calls.jsonl")
    if call_index >= len(plain_meta) or call_index >= len(rtc_meta):
        raise IndexError(
            f"call {call_index} is unavailable: no_rtc={len(plain_meta)}, rtc={len(rtc_meta)}"
        )

    plain_row, rtc_row = plain_meta[call_index], rtc_meta[call_index]
    if not rtc_row.get("rtc_context", False):
        raise ValueError(f"RTC call {call_index} does not contain guidance context")
    matched_fields = {
        "noise seed": plain_row.get("noise_seed") == rtc_row.get("noise_seed"),
        "camera images": plain_row.get("image_sha256") == rtc_row.get("image_sha256"),
    }
    plain_call = _load_call(plain_base, call_index)
    rtc_call = _load_call(rtc_base, call_index)
    matched_fields["measured state"] = np.array_equal(
        plain_call["measured_state"], rtc_call["measured_state"]
    )
    failures = [name for name, matches in matched_fields.items() if not matches]
    if failures:
        raise ValueError(
            "Cannot attribute the difference to RTC because these inputs differ: "
            + ", ".join(failures)
        )

    previous_call = _load_call(rtc_base, call_index - 1)
    previous_model = torch.from_numpy(previous_call["model_actions"])
    executed_horizon = int(rtc_row["executed_horizon"])
    delay_steps = int(rtc_row["predicted_delay_steps"])
    context = RTCGuidanceContext(
        prev_model_actions=previous_model,
        executed_horizon=executed_horizon,
        delay_steps=delay_steps,
    )
    remaining = context.get_prev_remaining()
    if remaining is None:
        raise ValueError("The previous action chunk has no remaining actions")
    horizon = rtc_call["model_actions"].shape[1]
    action_dim = rtc_call["model_actions"].shape[2]
    target, mask = build_rtc_target_and_mask(
        prev_remaining=remaining,
        horizon=horizon,
        action_dim=action_dim,
        delay_steps=delay_steps,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    config = json.loads((rtc_base / "env_config.json").read_text())
    model_cfg = config["piperx"]["model"]
    guidance_mode = model_cfg["openpi"].get("rtc_guidance_mode", "approx")
    norm_path = Path(model_cfg["openpi_data"]["norm_stats_path"])
    norm_stats = json.loads(norm_path.read_text())["norm_stats"]["actions"]
    no_rtc_model = plain_call["model_actions"].astype(np.float32)
    rtc_model = rtc_call["model_actions"].astype(np.float32)
    target_model = target.numpy()
    no_rtc_actions = _unnormalize_actions(no_rtc_model, norm_stats)
    rtc_actions = _unnormalize_actions(rtc_model, norm_stats)
    target_actions = np.full_like(rtc_actions, np.nan)
    overlap = remaining.shape[1]
    target_actions[:, :overlap] = _unnormalize_actions(remaining.numpy(), norm_stats)

    # The paired-eval recorder also saved decoded prefixes. Check this script is
    # applying exactly the same output transform before presenting the values.
    np.testing.assert_allclose(
        no_rtc_actions[:, : plain_call["decoded_actions"].shape[1]],
        plain_call["decoded_actions"],
        rtol=0,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        rtc_actions[:, : rtc_call["decoded_actions"].shape[1]],
        rtc_call["decoded_actions"],
        rtol=0,
        atol=2e-6,
    )

    guidance_scales = _guidance_scales(
        int(model_cfg["openpi"]["num_steps"]),
        float(model_cfg["openpi"]["rtc_guidance_clip"]),
        guidance_mode,
    )
    mask_array = mask.numpy()
    return ActionComparison(
        call_index=call_index,
        guidance_mode=guidance_mode,
        executed_horizon=executed_horizon,
        delay_steps=delay_steps,
        overlap=overlap,
        mask=mask_array,
        target_actions=target_actions,
        no_rtc_actions=no_rtc_actions,
        rtc_actions=rtc_actions,
        final_delta=rtc_actions - no_rtc_actions,
        guidance_scales=guidance_scales,
        guidance_multipliers=mask_array[..., None] * guidance_scales,
        no_rtc_model_actions=no_rtc_model,
        rtc_model_actions=rtc_model,
        target_model_actions=target_model,
    )


def _phase(index: int, comparison: ActionComparison) -> str:
    if index < comparison.delay_steps:
        return "hard"
    if index < comparison.overlap:
        return "soft"
    return "free"


def format_comparison(comparison: ActionComparison) -> str:
    """Format all 50 env-space action rows for terminal inspection."""
    first_scale = float(comparison.guidance_scales[0])
    if comparison.guidance_mode == "exact":
        correction_description = (
            "  pi0_velocity_rtc = raw_velocity - guidance_scale * "
            "VJP((old_target - action_hat) * mask)"
        )
        multiplier_description = (
            "mask_weight * guidance_scale is an input coefficient to the VJP; "
            "the Jacobian determines the actual velocity correction."
        )
    else:
        correction_description = (
            "  correction = guidance_scale * mask_weight * "
            "(old_target - denoising_mean)"
        )
        multiplier_description = (
            "The listed multiplier directly scales the approximate residual."
        )
    lines = [
        "RTC action comparison",
        (
            f"call={comparison.call_index} "
            f"guidance_mode={comparison.guidance_mode} "
            f"executed_horizon={comparison.executed_horizon} "
            f"predicted_delay_steps={comparison.delay_steps} "
            f"overlap={comparison.overlap}"
        ),
        "The RTC correction is iterative, not one fixed addition to the final action:",
        correction_description,
        multiplier_description,
        "guidance_scale by denoising step: "
        + np.array2string(comparison.guidance_scales, precision=6, separator=", "),
        (
            "The delta printed below is the observed final net effect: "
            "rtc_action - no_rtc_action."
        ),
        "Action order: " + ", ".join(ACTION_NAMES),
        "",
    ]
    for index in range(comparison.rtc_actions.shape[1]):
        mask_weight = float(comparison.mask[0, index, 0])
        lines.extend(
            [
                (
                    f"[{index:02d}] phase={_phase(index, comparison):4s} "
                    f"mask_weight={mask_weight:.8f} "
                    f"first_denoise_multiplier={mask_weight * first_scale:.8f}"
                ),
                "  denoise_multipliers = "
                + np.array2string(
                    comparison.guidance_multipliers[0, index, 0],
                    precision=7,
                    separator=", ",
                    suppress_small=False,
                ),
                "  old_target = "
                + np.array2string(
                    comparison.target_actions[0, index],
                    precision=7,
                    separator=", ",
                    suppress_small=False,
                ),
                "  no_rtc     = "
                + np.array2string(
                    comparison.no_rtc_actions[0, index],
                    precision=7,
                    separator=", ",
                    suppress_small=False,
                ),
                "  rtc        = "
                + np.array2string(
                    comparison.rtc_actions[0, index],
                    precision=7,
                    separator=", ",
                    suppress_small=False,
                ),
                "  rtc-no_rtc = "
                + np.array2string(
                    comparison.final_delta[0, index],
                    precision=7,
                    separator=", ",
                    suppress_small=False,
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def save_comparison(
    comparison: ActionComparison, output_dir: Path
) -> tuple[Path, Path, Path]:
    """Save a readable report, flat CSV, and full model-space arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rtc_action_comparison_call_{comparison.call_index:05d}"
    text_path = output_dir / f"{stem}.txt"
    csv_path = output_dir / f"{stem}.csv"
    npz_path = output_dir / f"{stem}.npz"
    text_path.write_text(format_comparison(comparison))

    fixed_columns = [
        "action_index",
        "phase",
        "mask_weight",
        "first_denoise_guidance_scale",
        "first_denoise_multiplier",
    ]
    multiplier_columns = [
        f"denoise_{index:02d}_multiplier"
        for index in range(comparison.guidance_scales.shape[0])
    ]
    value_columns = [
        f"{kind}_{name}"
        for kind in ("old_target", "no_rtc", "rtc", "rtc_minus_no_rtc")
        for name in ACTION_NAMES
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fixed_columns + multiplier_columns + value_columns
        )
        writer.writeheader()
        first_scale = float(comparison.guidance_scales[0])
        for index in range(comparison.rtc_actions.shape[1]):
            weight = float(comparison.mask[0, index, 0])
            row = {
                "action_index": index,
                "phase": _phase(index, comparison),
                "mask_weight": weight,
                "first_denoise_guidance_scale": first_scale,
                "first_denoise_multiplier": weight * first_scale,
            }
            row.update(
                {
                    column: float(value)
                    for column, value in zip(
                        multiplier_columns,
                        comparison.guidance_multipliers[0, index, 0],
                        strict=True,
                    )
                }
            )
            values = {
                "old_target": comparison.target_actions[0, index],
                "no_rtc": comparison.no_rtc_actions[0, index],
                "rtc": comparison.rtc_actions[0, index],
                "rtc_minus_no_rtc": comparison.final_delta[0, index],
            }
            for kind, action in values.items():
                row.update(
                    {
                        f"{kind}_{name}": float(value)
                        for name, value in zip(ACTION_NAMES, action, strict=True)
                    }
                )
            writer.writerow(row)

    np.savez_compressed(
        npz_path,
        mask=comparison.mask,
        guidance_scales=comparison.guidance_scales,
        guidance_multipliers=comparison.guidance_multipliers,
        target_actions=comparison.target_actions,
        no_rtc_actions=comparison.no_rtc_actions,
        rtc_actions=comparison.rtc_actions,
        final_delta=comparison.final_delta,
        target_model_actions=comparison.target_model_actions,
        no_rtc_model_actions=comparison.no_rtc_model_actions,
        rtc_model_actions=comparison.rtc_model_actions,
    )
    return text_path, csv_path, npz_path


def main() -> None:
    """Validate, print, and save one matched RTC/no-RTC action comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "comparison_root",
        type=Path,
        help="Paired-eval root containing no_rtc/standard and rtc/standard",
    )
    parser.add_argument(
        "--call",
        type=int,
        default=1,
        help="Matched model-call index; 1 is the first guided replan",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <comparison_root>/action_sequence_comparison",
    )
    args = parser.parse_args()
    comparison = build_comparison(args.comparison_root.resolve(), args.call)
    output_dir = args.output_dir or (
        args.comparison_root.resolve() / "action_sequence_comparison"
    )
    paths = save_comparison(comparison, output_dir)
    print(format_comparison(comparison), end="")
    print("Saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
