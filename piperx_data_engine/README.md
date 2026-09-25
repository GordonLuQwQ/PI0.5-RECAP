# PiperX Genesis Data Engine

This package owns every Genesis-side data operation for the PiperX stacking workflow. It collects scripted demonstrations, exports current-policy rollouts, replays failed rollouts, adds online IK corrections, and converts each result to LeRobot v3 format. It includes the Genesis-ready PiperX URDF and only the meshes referenced by that URDF.

Pi0.5 training, RTC inference, value-model training, and advantage-conditioned VLA updates live in the bundled `../third_party/RLinf` tree. The numbered entrypoints in `../workflows` connect the data engine to RLinf without duplicating either implementation.

## Task

The scene contains three objects:

- red cube
- red cylinder
- blue cube

An instruction has the form `put the {source} on the {destination}`. Because source and destination must differ, there are six ordered tasks:

1. `put the red cube on the red cylinder`
2. `put the red cube on the blue cube`
3. `put the red cylinder on the red cube`
4. `put the red cylinder on the blue cube`
5. `put the blue cube on the red cube`
6. `put the blue cube on the red cylinder`

`--task-set benchmark` collects tasks 2–6 and holds task 1 out completely for compositional generalization. The held-out task is red-cube-on-red-cylinder because the other five tasks still expose every object as both a source and a destination, while task 6 supplies a seen cube-on-cylinder geometry.

`--task-set all` collects all six ordered tasks. This is useful for the 600-episode all-task dataset, but it does not provide an unseen instruction-pair test.


The teacher uses object poses and contacts only to construct demonstrations and evaluate success. These privileged values are not included in policy observations. At each control step, the collector stores observation `o_t` before applying action `a_t`.

## Included files

| File | Role |
|---|---|
| `protocol.py` | Objects, ordered task pairs, seed partitions, randomization ranges, and success tolerances |
| `env.py` | Genesis scene, cameras, PiperX controller, observations, contacts, and reset logic |
| `planning.py` | Cartesian interpolation, batched IK, joint-path retiming, and dry-run collision checking |
| `teacher.py` | Scripted grasp-to-stack state machine and feedback gates |
| `success.py` | Stateful 2-second stacking success criterion and failure labels |
| `collect.py` | Balanced batched collection and raw episode writing |
| `export_lerobot.py` | Raw-to-LeRobot v3 conversion with H.264 camera streams |
| `export_policy_rollouts_lerobot.py` | Full policy-rollout export with terminal success labels |
| `intervention.py` | First-observable-failure watchdog used during online correction |
| `recover_policy_failures.py` | Replays a failed policy prefix and hands control to the IK teacher |
| `export_recovery_lerobot.py` | Exports only the contiguous IK expert suffix from each correction |
| `export_recovery_preview.py` | Creates correction videos and an English HTML index |
| `export_policy_failures_lerobot.py` | Replays and exports failed policy trajectories for value training |
| `export_positive_recap_lerobot.py` | Builds the all-positive rollout-success plus IK-suffix dataset |
| `runtime.py` | Genesis initialization for CPU or CUDA |
| `assets/piper_x_description/` | Bundled Genesis-ready PiperX URDF, visual meshes, and collision meshes |
| `install.sh` | Creates the Conda environment and installs Genesis, LeRobot, and this package |

## Installation

Run the included installer. It creates or reuses a Conda environment named `piperx_data` with Python 3.12, then installs Genesis, LeRobot dataset support, and this data engine. Use the Genesis and LeRobot clone commands in the [root README](../README.md#installation). The installer applies `patches/genesis_neutral_collision.patch`, the existing convex-overlap fix used during collection, to the Genesis source tree. Supplying the two source-tree paths explicitly makes the installation independent of where this repository was cloned:

```bash
cd /path/to/pi0.5-recap/piperx_data_engine
GENESIS_ROOT=/path/to/genesis-world \
LEROBOT_ROOT=/path/to/lerobot \
bash install.sh
conda activate piperx_data
```

To use a different environment name:

```bash
GENESIS_ROOT=/path/to/genesis-world \
LEROBOT_ROOT=/path/to/lerobot \
ENV_NAME=my_piperx_env \
bash install.sh
```

The bundled URDF is used automatically, so no asset argument is required. To use a different PiperX asset after installation, either set `PIPERX_URDF` or pass `--urdf`:

```bash
export PIPERX_URDF=/absolute/path/to/another/piper_x_description.urdf
```

## Collect from zero

Run a one-episode-per-task pilot first. A new output directory is required for every run.

```bash
export GENESIS_ROOT=/path/to/genesis-world
cd "${GENESIS_ROOT}"
RUN="data/piperx_benchmark_pilot_$(date +%Y%m%d_%H%M%S)"

PYOPENGL_PLATFORM=egl piperx-collect \
  --backend gpu \
  --num-envs 5 \
  --task-set benchmark \
  --episodes-per-pair 1 \
  --max-attempts-per-pair 20 \
  --seed-start 1000 \
  --cameras third_person wrist \
  --output "${RUN}_raw"
```

Collect the challenge split with 100 successful episodes for each of the five training pairs:

```bash
cd "${GENESIS_ROOT}"
RUN="data/piperx_benchmark_5x100_$(date +%Y%m%d_%H%M%S)"

PYOPENGL_PLATFORM=egl piperx-collect \
  --backend gpu \
  --num-envs 5 \
  --task-set benchmark \
  --episodes-per-pair 100 \
  --max-attempts-per-pair 1000 \
  --seed-start 1000 \
  --cameras third_person wrist \
  --output "${RUN}_raw"
```

Collect 100 successful episodes for every ordered pair, for 600 episodes total:

```bash
cd "${GENESIS_ROOT}"
RUN="data/piperx_all6_100_$(date +%Y%m%d_%H%M%S)"

PYOPENGL_PLATFORM=egl piperx-collect \
  --backend gpu \
  --num-envs 6 \
  --task-set all \
  --episodes-per-pair 100 \
  --max-attempts-per-pair 1000 \
  --seed-start 1000 \
  --cameras third_person wrist \
  --output "${RUN}_raw"
```

The collector balances saved successes across instructions. Failed teacher attempts are written to `attempts.jsonl`; they are not exported as imitation-learning episodes. Fixed-size Genesis batches may contain padding rows near the end of collection, and those rows are never saved as training data.

## Export to LeRobot

Run the exporter after collection finishes. It can run in a separate LeRobot environment because it does not import Genesis.

```bash
piperx-export-lerobot \
  --input "${RUN}_raw" \
  --output "${RUN}_lerobot" \
  --repo-id local/piperx_stacking \
  --cameras third_person wrist \
  --encoder-threads 4
```

The result uses LeRobot v3 Parquet metadata and H.264/YUV420P videos at 20 FPS. Camera fields are mapped as follows:

| Raw camera | LeRobot feature |
|---|---|
| `third_person` | `observation.images.image` |
| `wrist` | `observation.images.image2` |

## Action space

The action is a 7-dimensional continuous vector applied at 20 Hz:

```text
a_t = [q1, q2, q3, q4, q5, q6, gripper_closure]
```

| Component | Meaning | Raw range |
|---|---|---|
| `q1` | Absolute joint-1 target | `[-2.618, 2.618]` rad |
| `q2` | Absolute joint-2 target | `[0, 3.14]` rad |
| `q3` | Absolute joint-3 target | `[-2.9671, 0]` rad |
| `q4` | Absolute joint-4 target | `[-1.57, 1.57]` rad |
| `q5` | Absolute joint-5 target | `[-1.57, 1.57]` rad |
| `q6` | Absolute joint-6 target | `[-3.14, 3.14]` rad |
| `gripper_closure` | Symmetric finger command | `0.0` fully open, `1.0` fully closed |

Each 20 Hz command is linearly interpolated over ten 0.005-second physics steps. The observation state has the same seven channels, but contains measured joint positions and measured gripper closure rather than the previous command.

Absolute joint targets were chosen because they can be replayed through the same PD controller during collection, training evaluation, and learned-policy rollout. They also avoid integrating prediction error across a long demonstration.

## Observation setup

Each recorded observation contains measured 7D robot state and the selected RGB views:

| View | Resolution | Mount |
|---|---:|---|
| `third_person` | 640 × 480 | fixed third-person view; position `(1.50, 0.00, 0.42)` m, look-at `(0.30, 0.00, 0.22)` m |
| `wrist` | 224 × 224 | rigidly attached to `Link6` |

Every default collection records exactly `third_person` and `wrist`.

## Object randomization

All values are in the robot/world coordinate frame. Sampling depends only on the scene seed and is independent of the language instruction.

| Quantity | Distribution |
|---|---|
| Center X | independently uniform in `[0.24, 0.40]` m |
| Center Y | independently uniform in `[-0.16, 0.16]` m |
| Center Z | fixed at `0.041` m (`0.02` m tabletop + `0.02` m half-height + `0.001` m clearance) |
| Roll, pitch | fixed at `0` rad |
| Yaw | independently uniform in `[-π, π]` rad |
| Pairwise center distance | rejection sampled until every XY distance is at least `0.10` m |

Each object is 0.04 m tall. Cubes are 0.04 × 0.04 × 0.04 m; the cylinder has a 0.02 m radius. The sampling protocol does not randomize object size, mass, color, camera pose, robot home pose, lighting, or table geometry.

The XY region stays inside the PiperX grasp workspace while covering both sides of the arm. The 0.10 m separation prevents initial overlap and leaves enough clearance for a side grasp. Full-yaw sampling prevents the cube demonstrations from containing one privileged face orientation.

Scene seeds are partitioned as follows:

| Purpose | Seeds |
|---|---|
| Training collection | 1000–1999 |
| Validation | 8000–8019 |
| Final evaluation | 10000–10049 |
| Fixed-scene reversal test | 20000–20019 |

The same seed may be used for different instructions. This deliberately creates identical object layouts for language-reversal comparisons.

## Scripted teacher

For each environment, the teacher:

1. generates two orthogonal grasp candidates;
2. solves the complete approach and descent paths with batched IK;
3. retimes joint targets to 1.25 rad/s and 5.0 rad/s² command limits;
4. dry-runs paths in a separate Genesis scene with at most 0.04 rad between collision samples;
5. approaches, descends, closes the gripper, and verifies two-finger contact;
6. lifts and verifies at least 0.025 m source-object lift;
7. transfers while preserving the measured TCP-to-object offset;
8. lowers onto the destination, checks support contact, releases, retreats, and returns home;
9. holds the scene while the success monitor checks stability.

Transitions depend on measured TCP pose, object pose, contact, and lift evidence. A failed attempt does not count toward the balanced per-task quota, so the next unused training seed is attempted for that instruction.

## Success definition

Success is latched only when all conditions remain true for 2.0 seconds after release and retreat:

- source/destination XY center error ≤ 0.008 m;
- stack-height error ≤ 0.003 m;
- source and destination tilt ≤ 10°;
- destination displacement from its initial pose ≤ 0.015 m;
- third-object translation ≤ 0.005 m and rotation ≤ 5°;
- every object linear speed ≤ 0.015 m/s and angular speed ≤ 0.20 rad/s;
- TCP is at least 0.12 m from both stacked objects;
- the robot has no object contact;
- Genesis reports contact between source and destination.

## Raw output

Each successful episode produces:

- `episode_NNNNNN.npz`: state, selected RGB streams, action, teacher phase, and final state;
- `episode_NNNNNN.json`: instruction, seed, task pair, initial poses, success details, and teacher events.

Each run also contains `collection.json`, `protocol.json`, `provenance.json`, `cameras.json`, `attempts.jsonl`, and `summary.json`. These files record the exact collection arguments, seed pool, camera calibration, attempt outcomes, and progress needed to reproduce the dataset.

## Policy rollouts and IK correction

The data engine exposes the Genesis-side commands needed after the first Pi0.5 checkpoint has been trained:

```text
piperx-export-policy-rollouts   complete rollout set with is_success labels
piperx-recover-policy-failures  policy prefix followed by online IK recovery
piperx-export-recovery          contiguous expert suffixes only
piperx-export-recovery-preview  MP4 previews and watch.html
piperx-export-policy-failures   failed trajectories for the value VLM
piperx-export-positive-recap    rollout successes plus corrected expert suffixes
```

Use the numbered scripts in `../workflows` for the complete sequence and all required arguments. The repository-level `../README.md` documents the eight stages from fresh collection through value-conditioned VLA fine-tuning.
