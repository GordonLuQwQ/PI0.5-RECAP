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

"""Read the resolved environment protocol passed by RLinf to the scene process."""

import json
import os
from pathlib import Path

ENV_CONFIG = json.loads(Path(os.environ["PIPERX_EVAL_CONFIG"]).read_text())
PIPERX = ENV_CONFIG["piperx"]
OUTPUT = Path(PIPERX["output_dir"])
SUITE = PIPERX["suite"]
PAIRS = tuple(tuple(pair) for pair in PIPERX["pairs"])
SEEDS = PIPERX["seeds"]
TOTAL_TRIALS = len(PAIRS) * len(SEEDS)
MAX_CONTROL_STEPS = ENV_CONFIG["max_episode_steps"]
CALIBRATION = Path(PIPERX["calibration"])
CAMERAS = tuple(PIPERX["cameras"])
FAST_CHUNK_STEP = bool(PIPERX.get("fast_chunk_step", False))
VIDEO_STRIDE = int(PIPERX.get("video_stride", 1))
VIDEO_CAMERAS = tuple(PIPERX.get("video_cameras", CAMERAS))
VIDEO_FAILURE_ONLY = bool(PIPERX.get("video_failure_only", False))
PARALLEL_ENVS = int(PIPERX.get("parallel_envs", 1))
WORKER_SLOT = int(os.environ.get("PIPERX_WORKER_SLOT", "-1"))
CHECKPOINT = Path(PIPERX["model"]["model_path"])
EXCLUDED_TASK = PIPERX["held_out_task"]
CYLINDER_X_RANGE = None
PLACEMENT_RULE = PIPERX["placement_rule"]
RTC_CONFIG = {
    **PIPERX["rtc"],
    "implementation": "rlinf_official_entrypoint",
    "guidance_mode": PIPERX["model"]["openpi"]["rtc_guidance_mode"],
    "guidance_clip": PIPERX["model"]["openpi"]["rtc_guidance_clip"],
    "action_horizon": PIPERX["model"]["openpi"]["action_horizon"],
    "action_chunk": PIPERX["model"]["openpi"]["action_chunk"],
    "num_steps": PIPERX["model"]["openpi"]["num_steps"],
}


def trial_batches() -> list[list[dict | None]]:
    """Return resumable fixed-width batches of pending trials."""
    from ledger import completed_trials, ensure_manifest, slug

    signature = ensure_manifest()
    done = {row["directory"] for row in completed_trials(OUTPUT, signature)}
    trials = [
        {"pair": list(pair), "seed": seed, "directory": slug(pair, seed)}
        for seed in SEEDS
        for pair in PAIRS
    ]
    batches = []
    for start in range(0, len(trials), PARALLEL_ENVS):
        batch = trials[start : start + PARALLEL_ENVS]
        batch.extend([None] * (PARALLEL_ENVS - len(batch)))
        pending = [
            trial if trial is not None and trial["directory"] not in done else None
            for trial in batch
        ]
        if any(trial is not None for trial in pending):
            batches.append(pending)
    return batches


def trial_plan() -> list[dict | None]:
    """Return this simulator slot's trial from every pending batch."""
    batches = trial_batches()
    if WORKER_SLOT >= 0:
        if WORKER_SLOT >= PARALLEL_ENVS:
            raise ValueError(
                f"PIPERX_WORKER_SLOT={WORKER_SLOT} exceeds parallel_envs={PARALLEL_ENVS}"
            )
        return [batch[WORKER_SLOT] for batch in batches]
    return [trial for batch in batches for trial in batch if trial is not None]
