# Third-party dependencies

This directory contains the modified RLinf source used by the PiperX Pi0.5 and RECAP experiments. The dependency environment follows the official RLinf installer, while the checked-in source includes the project-specific PiperX, Genesis, LoRA, evaluation, and offline RECAP integrations described below.

## RLinf source and attribution

The bundled [`RLinf/`](RLinf/) tree is based on [RLinf](https://github.com/RLinf/RLinf) v0.4.0 at upstream commit:

```text
f26caaa02597113bb533d459dd2c8a4f7b19946b
```

It is a modified source distribution rather than an unmodified upstream checkout. Keep the upstream [`RLinf/LICENSE`](RLinf/LICENSE) file when redistributing this directory. The original project documentation remains in [`RLinf/README.md`](RLinf/README.md) and [`RLinf/README.zh-CN.md`](RLinf/README.zh-CN.md).

## Installation

Install from the bundled source so that the PiperX additions are available. The command below uses RLinf's official UV installer:

```bash
cd third_party/RLinf

bash requirements/install.sh embodied \
  --model openpi \
  --env libero

source .venv/bin/activate
```

Verify the environment after installation:

```bash
python -c "import rlinf, torch, ray; print(rlinf.__file__); print(torch.__version__)"
ray --version
```

The `libero` environment target supplies the OpenPI and LeRobot dependencies used by the policy and dataset loaders. Genesis is intentionally not installed in the RLinf virtual environment. The RLinf policy process runs under Python 3.11, while the Genesis scene runs in its own environment and exchanges observations and actions with RLinf through the bundled local socket bridge.


The upstream installation reference is available in [`RLinf/docs/source-en/rst_source/start/installation.rst`](RLinf/docs/source-en/rst_source/start/installation.rst).

## Project-specific RLinf changes

The bundled fork adds the following functionality on top of the pinned upstream revision:

1. A PiperX LeRobot data configuration with `third_person` and `wrist` RGB inputs, a 7-dimensional robot state, and 7-dimensional absolute joint/gripper actions.
2. A PiperX Genesis environment adapter and local socket bridge between the RLinf Python 3.11 process and the separate Genesis process.
3. PiperX Pi0.5 SFT configuration, two-camera input mapping, and task-specific normalization-statistics loading.
4. Native Pi0.5 LoRA training and adapter checkpoint overlay for the VLM branch, action expert, and action/state/time projection layers.
5. Binary advantage routing between the original task prompt and the task prompt with the `Advantage: positive` suffix.
6. Standard, reversal, and RTC PiperX evaluation with persisted scenes, videos, per-pair success rates, and a failure taxonomy.
7. An offline RECAP workflow for return generation, Pi0.5 value-VLM training, binary-advantage labeling, and value-conditioned Pi0.5 fine-tuning.

The main integration points are:

- [`RLinf/rlinf/envs/sim/genesis/piperx_env.py`](RLinf/rlinf/envs/sim/genesis/piperx_env.py): RLinf environment wrapper for the external Genesis process.
- [`RLinf/rlinf/models/embodiment/openpi/dataconfig/piperx_dataconfig.py`](RLinf/rlinf/models/embodiment/openpi/dataconfig/piperx_dataconfig.py): PiperX observation, state, action, and prompt mapping.
- [`RLinf/examples/sft/config/piperx_sft_openpi_pi05_rlinf.yaml`](RLinf/examples/sft/config/piperx_sft_openpi_pi05_rlinf.yaml): PiperX Pi0.5 supervised fine-tuning configuration.
- [`RLinf/evaluations/piperx/`](RLinf/evaluations/piperx/): Genesis bridge, evaluation protocol, RTC execution, reports, and videos.
- [`RLinf/examples/offline_rl/piperx_recap_edited/`](RLinf/examples/offline_rl/piperx_recap_edited/): custom value-VLM and advantage-conditioned VLA pipeline.

Installing a fresh, unmodified RLinf checkout does not provide these integrations. Use the bundled fork, or reproduce the same changes against the pinned upstream commit.

## Models, statistics, and datasets

Model weights and generated data are intentionally excluded from `third_party/`. They must be downloaded or generated separately:

- the RLinf-compatible Pi0.5 base checkpoint;
- PiperX state/action `norm_stats.json`;
- LeRobot training datasets;
- value-VLM and policy checkpoints produced by training;
- Genesis/PiperX assets supplied by the data-engine package.

Several experiment YAML files preserve the original machine's absolute `/home/ajifang/...` paths. Replace those paths with local checkpoint, dataset, normalization-statistics, Genesis project, and output locations before running on another machine.

The separate `Physical-Intelligence/openpi` checkout used during early experiments is not required by the final runtime. The official RLinf installer installs its supported OpenPI package into `RLinf/.venv`; the final pipeline imports that installed package together with the modified in-tree `openpi_rlinf` implementation.



This separation avoids forcing Genesis and RLinf/OpenPI into one dependency environment. Run RLinf commands from `third_party/RLinf` with its `.venv`; run data-engine and Genesis scene commands with the environment documented by the data-engine package.

For the experiment-specific workflow, see the bundled [PiperX RECAP documentation](RLinf/examples/offline_rl/piperx_recap_edited/README.md) and [PiperX evaluation documentation](RLinf/evaluations/piperx/README.md).
