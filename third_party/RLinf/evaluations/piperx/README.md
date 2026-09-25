# PiperX 的 RLinf 评估

这组配置通过 RLinf 的 `evaluations/run_eval.sh` 和 `eval_embodied_agent.py`，评估本机训练好的 π0.5 checkpoint。普通 eval 使用 `EnvWorker` 和 `MultiStepRolloutWorker`；RTC 使用 `RTCEnvWorker`、`RTCMultiStepRolloutWorker` 和 `openpi_rlinf` guidance sampler。两种方式都连接现有的 PiperX／Genesis 场景。

RTC 调度使用 [RLinf 官方指南](../../docs/source-zh/rst_source/guides/rtc.rst)中的 env worker 和 rollout worker，`openpi_rlinf` sampler 按官方 JAX 实现用 VJP 修正速度场。`runner.rtc.enabled` 控制 worker 选择，`actor.model.openpi.rtc_enabled` 通过 `${runner.rtc.enabled}` 同步控制模型的动作衔接；`rollout.model` 引用同一份模型配置。配置默认开启 RTC，模型使用 `openpi.task: eval`。启动时会打印 worker、RTC 开关、guidance 模式、预测长度、请求下一段的门槛和去噪步数。

当前普通 eval 用以下命令启动，先执行 6 个方向任务各 10 次，共 60 条；随后对红圆柱与蓝方块这一对执行两个方向，各 1 次，共 2 条。

```bash
PIPERX_RESULT_ROOT="/path/to/rlinf-experiments/pi05_vlm_action_lora_2views_10k_plus5k_20260921/evaluation_official_standard_h50_chunk10_results" \
bash /path/to/RLinf/evaluations/piperx/run.sh \
  'runner.rtc.enabled=False' 'actor.model.num_action_chunks=10'
```

上述普通 eval 命令每次预测 50 步，只取前 10 步执行，然后重新观察。`action_horizon=50` 是模型预测长度，命令中的 `num_action_chunks=10` 使 `action_chunk` 同步为 10，即传给环境的动作长度；`num_steps=10` 是模型去噪次数。反归一化后，每个动作有 7 维。RTC 保留完整的 50 步动作缓冲和模型空间历史，并使用 `min_exec_horizon=10`，启动命令为：

```bash
PIPERX_RESULT_ROOT="/path/to/rlinf-experiments/pi05_vlm_action_lora_2views_10k_plus5k_20260921/evaluation_official_rtc_exact_h50_exec10_delay14_results" \
bash /path/to/RLinf/evaluations/piperx/run.sh \
  'runner.rtc.enabled=True'
```

每条常规任务使用场景种子 `10000–10009`，reversal 使用 `20000`。每个种子的完整 Genesis 状态只初始化一次，后续方向任务恢复同一状态。指令改变源物体和目标物体。每回合最多执行 600 个物理控制步，成功后停止物理推进；普通 worker 完成剩余的固定通信轮次后提交结果。

启动器设置两个已有 Python 环境的路径，然后调用官方命令：

```bash
bash evaluations/run_eval.sh piperx piperx_eval_pi05_RTC
bash evaluations/run_eval.sh piperx piperx_reversal_eval_pi05_RTC
```

实际调用还会传入未完成回合数和日志目录。重复运行 `run.sh` 会跳过已经完整保存的回合，未完成回合先归档再重跑。不能将其他参数的结果累加到同一目录；需要另设 `PIPERX_RESULT_ROOT`。

模型加载 `global_step_5000/actor` 中的原生 LoRA 权重，使用训练时的 normalization stats、两路相机、离散状态提示词及 50 步模型 horizon。模型内部输出 `B × 50 × 32`；反归一化并选取 7 个动作维度后，RTC worker 收到 `B × 50 × 7` 动作缓冲，分别为 6 个绝对关节角和连续夹爪闭合度。上面的普通 eval 命令则将环境输出截取为 `B × 10 × 7`。

`Pi0Eval` 将完整的 `B × 50 × 32` 模型空间动作返回给 rollout worker。下一次请求通过 `executed_horizon` 去掉已经经过的动作位置，将剩余部分作为 RTC 的旧轨迹目标。

例如，模型预测 `a_0…a_49` 后，当前 chunk 在执行到索引 10 时请求下一段。RTC 使用 `a_10…a_49` 共 40 步作为目标；若预测推理延迟为 5 步，新 chunk 的前 5 个位置以 `a_10…a_14` 为目标且 mask 为 1，接下来 35 个位置参考 `a_15…a_49` 且 mask 逐渐减小，最后 10 个位置没有旧预测重叠且 mask 为 0。exact sampler 先由当前去噪变量估计干净动作，再通过 VJP 将加权误差转换成速度场修正，最后继续 Euler 积分。若实际延迟也是 5 步，环境在等待期间继续执行旧动作，收到新 chunk 后从其索引 5 开始取动作。

当前 RTC 配置使用 `min_exec_horizon=10`：当前 chunk 的时间索引达到 10 且没有待完成请求时，官方 worker 发起下一次推理。等待期间继续从 50 步动作缓冲取动作，收到结果后按实际延迟跳过新 chunk 中已经过去的位置。`action_horizon=50`、`action_chunk=50`，并且 `rtc_history_horizon=null`，因此环境动作缓冲和下一轮 guidance 历史都保留 50 步。后续请求的时间索引包含收到该 chunk 前已经经过的延迟，因此不能将门槛解释为每次收到新 chunk 后固定再执行 10 步。

RTC 的初始延迟为 `14` 步，延迟窗口为 `8`，guidance 使用 `exact`，系数上限为 `5.0`。`rtc_enabled` 引用 `runner.rtc.enabled`，两处开关保持同步。仿真按官网说明使用 `chunk_pause_seconds` 模拟动作执行时间；本配置设为 `0.05` 秒，对应 20 Hz 的动作步间隔下限，实际墙钟速度取决于仿真和推理耗时。模型 Euler 去噪保持 `10` 步。

执行流程为：首次推理生成动作 chunk → `PiperXEnv.step()` 逐步执行 → 达到 `min_exec_horizon=10` 后发送最新观测请求下一段 → rollout worker 将上一段模型空间动作、已执行步数和预测延迟传给 `Pi0Eval` → guidance sampler 生成下一段动作。等待推理期间继续执行旧动作；收到结果后，官方 env worker 根据实际经过的步数对齐新 chunk。每回合结束会完成待处理请求，整组评估结束后发送 `stop`。

RLinf 使用 Python 3.11，现有 Genesis 场景使用 Python 3.13，因此 `PiperXEnv` 通过本地 socket 传递 NumPy 图像、实测状态和动作。[bridge.py](bridge.py) 提供数据包编解码、观测检查和评估轨迹保存。子进程负责场景、单步动作、任务判定和视频记录，RTC 调度由 RLinf worker 负责。新接入在 `get_env_cls()` 中选择该环境，并在 OpenPI data config 中注册 `pi05_piperx`；官方评估入口和两个 RTC worker 保持不变，`openpi_rlinf` sampler 使用 exact 速度场 guidance。

结果保存在 `PIPERX_RESULT_ROOT` 下。上述普通 eval 命令使用 `evaluation_official_standard_h50_chunk10_results/`，当前 RTC 配置使用 `evaluation_official_rtc_exact_h50_exec10_delay14_results/`，两者内部结构相同。原有 RTC 目录保留旧参数的运行记录供对比。

```text
/path/to/rlinf-experiments/pi05_vlm_action_lora_2views_10k_plus5k_20260921/evaluation_official_rtc_exact_h50_exec10_delay14_results/
  watch.html
  standard/summary.json
  standard/watch.html
  reversal/summary.json
  reversal/watch.html
```

每个任务子目录保存两个视角的完整视频、动作轨迹、初始状态、相机标定和 `report.json`；日志位于各套任务的 `logs/` 和 `genesis.log`。Ray 临时目录使用 `/dev/shm/piperx-$UID`，视频解码禁用交互输入并设有超时。

成功判定沿用本实验的 `released_on_top_v1`：水平偏差不超过 2 cm、高度误差不超过 5 mm、源物体接触目标物体且已脱离机器人、夹爪闭合度不超过 0.5。该判定无持续稳定时间要求，不是 RTC 指南定义的任务指标。

要在完全相同的场景中比较 RTC 和普通 10 步 action chunk，可运行以下配对评测。它使用一个 scene seed 覆盖全部 6 个方向任务，共执行 6 条 no-RTC 和 6 条 RTC。两种模式顺序运行，避免两个 eval 共享同一个 Ray cluster；no-RTC 生成的 Genesis reset snapshot 会原样复制给 RTC。普通模式预测 50 步并执行前 10 步，RTC 模式在第 10 步发起异步重规划并在推理期间继续执行旧 chunk。

```bash
PIPERX_SCENE_SEED=10000 \
bash evaluations/piperx/run_paired_joint_comparison.sh
```

结果位于脚本打印的 `joint_comparison/`。每个任务包含一张 PNG 和一份 CSV：PNG 分别对比 6 个关节的实测角度、请求的关节目标和相邻控制步的目标变化；`summary.json` 保存两种模式每个关节变化量的最大值与 95 百分位。绘图前脚本会验证两次运行的初始机器人状态和完整场景真值一致。

要逐位置查看 RTC 对动作序列的影响，可对同观测、同随机噪声的成对评估记录运行：

```bash
python evaluations/piperx/print_rtc_action_comparison.py \
  /path/to/rlinf-experiments/pi05_vlm_action_lora_2views_10k_plus5k_20260921/evaluation_rtc_vs_no_rtc_20260922_043905 \
  --call 1
```

脚本先验证两次推理的图像、实测状态和随机噪声一致，再打印每个动作位置的旧 chunk 目标、hard／soft／free 阶段、mask 权重、10 次去噪各自的 guidance 系数、无 RTC 动作、RTC 动作及最终差值。它会从保存的配置识别 `exact` 或旧的 `approx` 模式。exact 模式中的 `guidance_scale × mask_weight` 是进入 VJP 的系数，实际速度修正还取决于 Jacobian；输出中的 `rtc-no_rtc` 是完整去噪后可观测到的净变化。完整结果同时保存为文本、CSV 和 NPZ。环境一旦因先前动作产生分叉，脚本会拒绝比较对应 call，避免把不同观测造成的变化归因于 RTC。

训练入口、各文件职责以及 RLinf 到 Genesis 的完整调用关系见 [RECAP 实验目录说明](../../examples/offline_rl/piperx_recap_edited/README.md)。RTC 参数含义见 [RTC 指南](../../docs/source-zh/rst_source/guides/rtc.rst)。
