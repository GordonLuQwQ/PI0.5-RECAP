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

"""Compose the official Hydra config and count pending evaluation episodes."""

import argparse
import json
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main() -> None:
    """Print the pending count for run.sh without creating a model or simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    with initialize_config_dir(
        version_base="1.1", config_dir=str(Path(__file__).parent)
    ):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)
    from rlinf.config import validate_only_eval_rollout_model

    OmegaConf.resolve(cfg)
    validate_only_eval_rollout_model(cfg.rollout.model)
    if cfg.rollout.model.openpi.rtc_enabled != cfg.runner.rtc.enabled:
        raise ValueError("The runner and model RTC switches must agree")
    for path in (
        cfg.rollout.model.model_path,
        cfg.rollout.model.openpi_data.norm_stats_path,
        cfg.env.eval.piperx.calibration,
        cfg.env.eval.piperx.python_path,
        cfg.env.eval.piperx.bridge_script,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    workers = (
        "RTCEnvWorker + RTCMultiStepRolloutWorker"
        if cfg.runner.rtc.enabled
        else "EnvWorker + MultiStepRolloutWorker"
    )
    model = cfg.rollout.model.openpi
    print(
        f"{cfg.env.eval.piperx.suite}: {workers}; "
        f"RTC={cfg.runner.rtc.enabled}; guidance={model.rtc_enabled}; "
        f"guidance_mode={model.rtc_guidance_mode}; "
        f"horizon={model.action_horizon}; chunk={model.action_chunk}; "
        f"rtc_history={model.get('rtc_history_horizon', model.action_horizon)}; "
        f"min_exec_horizon={cfg.runner.rtc.min_exec_horizon}; "
        f"denoise_steps={model.num_steps}",
        file=sys.stderr,
        flush=True,
    )
    output = Path(cfg.env.eval.piperx.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "env_config.json"
    config_path.write_text(
        json.dumps(OmegaConf.to_container(cfg.env.eval, resolve=True), indent=2)
    )
    os.environ["PIPERX_EVAL_CONFIG"] = str(config_path)
    from ledger import ensure_manifest, update_summary
    from settings import PARALLEL_ENVS, trial_batches

    signature = ensure_manifest()
    batches = trial_batches()
    pending_trials = sum(
        trial is not None for batch in batches for trial in batch
    )
    update_summary(output, signature, status="ready" if batches else "completed")
    cfg.env.eval.rollout_epoch = len(batches)
    OmegaConf.save(cfg, output / "eval_config.yaml", resolve=True)
    print(
        f"parallel_envs={PARALLEL_ENVS}; pending_trials={pending_trials}; "
        f"pending_batches={len(batches)}",
        file=sys.stderr,
        flush=True,
    )
    print(len(batches))


if __name__ == "__main__":
    main()
