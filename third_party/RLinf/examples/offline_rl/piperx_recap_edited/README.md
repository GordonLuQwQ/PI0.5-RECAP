# PiperX RECAP value VLM（EDITED）

## 当前的 all-true 两阶段流程

`run_all_true_pipeline_edited.sh` 会按顺序完成当前实验，不需要在阶段之间手工换命令：

1. 恢复第二批 300 次 policy rollout 的持久化初态并重放全部 73 条失败轨迹，重新渲染第三视角和腕部视角，转换为标准 LeRobot v3，并根据原评测报告逐帧写入 `is_success=false`。
2. 把最初的 600 条成功示教转换为相同的双视角 schema、逐帧写入 `is_success=true`，再与 73 条失败轨迹物理合并为一个 673 条 LeRobot v3 数据集。原始 600 条目录保持不变。
3. 检查已经保存的 step-3000 value checkpoint 并保持它不变；这个 all-true VLA 阶段不会继续训练 value VLM。
4. 恢复同一批 rollout 的初态并重放 227 条成功轨迹，重新渲染第三视角和腕部视角；成功标签来自原评测报告。
5. 加入当前 26 条纠正轨迹中 `is_expert == 1` 的 IK 后缀。失败 policy 前缀不会被当作正样本重新学习。
6. 为合计 253 条正轨迹的每一帧写入 `advantage=true`。数据加载器同时编码原任务指令与 `原任务指令 + Advantage: positive`，并把布尔 advantage 保留到模型 `forward`。
7. 从当前 step-5000 PiperX π0.5 actor checkpoint 继续训练 VLA 3000 step，同时更新 actor 的 VLM LoRA、action-expert LoRA 和 action/state/time projection。第一阶段所有样本都使用 positive prompt，不做 unconditional dropout；只保存最终 adapter，不复制 dense 权重。

在前台执行这一条命令：

```bash
cd /path/to/RLinf
bash examples/offline_rl/piperx_recap_edited/run_all_true_pipeline_edited.sh
```

脚本会检查 600 条原始数据、300 条 rollout、全部 73 条失败轨迹、26 条已保存纠正轨迹、磁盘空间和每帧 advantage。已经完整结束的阶段会跳过；如果发现只有一部分的输出目录，则停止并报告该目录，防止把残缺数据接着训练。

最终输出是：

```text
/path/to/genesis-world/vla/stacking/data/pi05_policy_failures_73_lerobot
/path/to/genesis-world/vla/stacking/data/pi05_value_600success_73failure_lerobot
/path/to/rlinf-experiments/piperx_recap_value_stage1_600plus73_edited/checkpoints/step_003000/pi05_value.pt
/path/to/genesis-world/vla/stacking/data/pi05_positive_227plus26_lerobot
/path/to/rlinf-experiments/piperx_advantage_stage1_all_true_edited/checkpoints/global_step_3000/actor
```

最后一个目录是可直接填入 `model_path` 的 native OpenPI_RLinf adapter。加载时会先读取原 step-5000 dense checkpoint，再覆盖这次训练的 adapter，并在正 indicator 推理指令后加入 advantage 条件。

这里的 indicator 是真正进入 policy `forward` 的布尔路由信号，而不是事先永久改写数据集任务文本。`indicator=1` 选择 positive prompt tokens，`indicator=0` 选择原任务 prompt tokens。第一阶段要求 sidecar 全 True。后续阶段会使用单独的 continuation 配置载入这个 stage-one checkpoint，把 `data.require_all_true` 设为 `false`，并将 `data.advantage_tag` 指向 Value VLM 产生的逐帧 0/1 sidecar；本次改动只完成第一阶段，不读取此前的 positive adapter。

Value VLM 会从同一个 673 条数据集看到 600 条成功示教与全部 73 条完整失败 rollout，因此能够学习任务进度以及成功/失败回报差异。该数据集共有 374226 帧。当前训练器按 `max_steps` 而不是 `epochs` 停止，并采用有放回随机采样；batch size 112、6683 step 共抽样 748496 帧，相当于约 2.0001 个有效 epoch。当前 stage-one 实验仍按要求把所有 label 设为 `true`，所以 value 预测尚未参与标签计算。下一阶段应使用训练后的 value 计算 `A_t > 0` 并生成同 schema 的布尔 sidecar；训练器已经允许 True/False 混合标签。

第二批 rollout 的 seed 已参与训练，最终成功率必须使用新的未见 seed 重新评测。

这套代码把你当前已经训练过的 PiperX π0.5 当作 value VLM，而不是重新使用官方 RECAP 示例中的 SigLIP2 + Gemma3 critic。加载的 checkpoint 是：

```text
/path/to/RLinf/logs/20260921-20:49:58-piperx_sft_openpi_pi05_rlinf/piperx_mixed600_official_base/checkpoints/global_step_5000/actor
```

它与当前 policy 使用同一套 `SigLIP + PaliGemma` 权重、两路相机顺序、PiperX state 编码、tokenizer 和 normalization。value forward 只运行 π0.5 的 observation prefix，然后把有效 image/language token 的 hidden state 做 masked mean pooling，再交给新的 201-bin value head。action expert 虽然属于同一个 checkpoint、会被加载到显存，但 value forward 不调用 `run_suffix()`，因此不执行 flow matching action expert。

它会另外建立一个 π0.5 副本，因此不会覆盖或漂移现有 policy checkpoint。默认从你训练好的权重继续微调 **PaliGemma expert-0 的 LoRA** 与新的 value head；action expert 始终冻结且不执行。SigLIP 视觉编码器默认冻结，若之后确实需要全量视觉微调，可把 `train_vision_encoder` 改成 `true`。最终 checkpoint 只保存 value head 和 value 副本中更新过的 VLM adapter，不再复制 13.5 GB 的 dense 权重。

## 文件与职责

- `run_all_true_pipeline_edited.sh`：当前实验的一键前台入口，串联失败轨迹格式转换、673 条物理合并、value 训练、成功轨迹重放、all-true 校验和 policy LoRA SFT。
- `config_stage1_600plus73_edited.yaml`：读取单一 673 条 LeRobot 数据集训练 value VLM 的配置。
- `positive_policy_edited.yaml`：227 条成功 rollout 加 26 条 IK expert 后缀的 policy 微调配置。
- `train_positive_policy_edited.py`：读取 LeRobot v3 双视角数据，保留逐帧 advantage 到 native π0.5 `forward`，训练 VLM/action-expert LoRA，并保存紧凑 native adapter。
- `/path/to/genesis-world/vla/stacking/export_positive_recap_lerobot.py`：重放成功 rollout、追加 expert 后缀并生成 all-true sidecar。
- `/path/to/genesis-world/vla/stacking/export_policy_failures_lerobot.py`：重放全部 73 条失败 rollout、生成双视角 LeRobot value 数据。
- `/path/to/genesis-world/vla/stacking/merge_value_lerobot.py`：给 600 条成功数据补 outcome 标签、移除未使用的第三路相机，并与 73 条失败数据物理合并成 673 条。
- `config_edited.yaml`：保留失败前缀的 failure-aware value 实验配置；当前 all-true 流程不读取它。
- `checkpoint_edited.py`：按现有 checkpoint 的真实结构重建 `gemma_2b_lora + gemma_300m_lora` π0.5，并加载 `global_step_5000` 权重。
- `pi05_value_critic_edited.py`：执行 π0.5 VLM prefix，聚合 2048 维 hidden state，接 `2048 → 1024 → 201` value head；只打开 PaliGemma expert-0 LoRA 的梯度。
- `reward_edited.py`：奖励、discounted return、`[-1, 0]` normalization、201-bin 投影和 categorical cross entropy。
- `compute_returns_edited.py`：读取 SFT/rollout parquet，生成 `return`、`reward`、`prompt` sidecar；不改写原始动作、图像、`info.json` 或 `stats.json`。
- `export_failure_prefixes_edited.py`：把现有 IK 纠正数据中 `is_expert == 0` 的双视角 policy 前缀导出为失败 rollout。
- `data_edited.py`：直接读取当前 LeRobot v3 parquet 与 MP4，并将第三视角、腕部视角、7 维 state 和 prompt 送进 π0.5。它不依赖 RLinf 环境里较旧的 LeRobot reader。
- `train_value_edited.py`：混合 SFT 与 rollout 数据，使用不同学习率优化 VLM LoRA 和 value head，写 TensorBoard，在固定间隔原子保存模型与 optimizer，并可从 `latest_checkpoint.json` 自动恢复。
- `predict_value_edited.py`：加载训练结果，对 LeRobot 数据中的指定帧输出 normalized value 与换算后的 raw return。
- `self_check_edited.py`：不加载大模型，检查奖励、normalization、bin 投影和期望值数学。
- `compute_returns_edited.yaml`：奖励设置的简表，便于与你参考的 RECAP 配置逐项对照；运行脚本读取的是 `config_edited.yaml`。

它还会调用这些现有 RLinf/OpenPI 文件：

- `rlinf/models/embodiment/openpi_rlinf/__init__.py`：建立并加载 π0.5。
- `rlinf/models/embodiment/openpi_rlinf/pi0.py`：`build_prefix_cache()` 运行视觉与语言 prefix。
- `rlinf/models/embodiment/openpi_rlinf/modules/model.py`：把图像 resize/pad，并建立 model observation。
- `rlinf/models/embodiment/openpi/dataconfig/piperx_dataconfig.py`：两路相机、7 维 state 与 prompt 的 PiperX 映射。
- `/path/to/models/pi05_base_official_openpi_rlinf/piperx_five_tasks_2views/norm_stats.json`：与当前 π0.5 训练和评测相同的 state/action normalization。

## 奖励和 normalization

数值与 RLinf RECAP 样例相同：

```text
普通步                   r_t = -1
成功轨迹最后一步         r_T = 0
失败轨迹最后一步         r_T = -300
gamma                         = 1
G_t = r_t + gamma * G_(t+1)
V_target = G_t / abs(global_return_min)
value support                 = [-1, 0]
value bins                    = 201
相邻 bin 间距                = 0.005
```

以 600 帧轨迹为例，成功轨迹首帧的 raw return 是 `-599`，失败轨迹首帧是 `-899`。若混合数据的 `global_return_min=-899`，两者的 value target 分别约为 `-0.6663` 与 `-1.0`。训练时先把连续 target 线性投影到相邻的两个 bin，再计算 categorical cross entropy；推理值是 201 个 bin 概率的期望。

## 当前使用的 rollout 负样本

第二批 300 次 policy 评测包含 227 次成功和 73 次失败；原目录只有稀疏第三视角视频，因此 `export_policy_failures_lerobot.py` 会从持久化 Genesis 初始状态逐动作重放全部 73 条完整失败轨迹，重新生成全帧第三视角、腕部视角和状态，并依据原始 `report.json` 将每帧标成 `is_success=false`。GPU 仿真重放不是逐位确定的，因此关节状态漂移会写入审计摘要，但不会覆盖原评测时已经确定的 outcome。随后 `merge_value_lerobot.py` 给 600 条成功示教补上 `is_success=true`，只保留与 π0.5 一致的两路训练相机，并生成单一 673 条训练集。26 条 IK 纠正轨迹的 policy 前缀来自这些失败任务的一部分，不再重复加入 value 数据。

导出后的 LeRobot 数据每帧包含：

```text
observation.images.image      第三视角
observation.images.image2     腕部视角
observation.state             7 维
episode_index
frame_index
timestamp
task_index（或 task）
is_success                    轨迹最终成功标记
```

`compute_returns_edited.py` 与原 RECAP 一样读取每条 rollout 最后一帧的 `is_success`。73 条失败轨迹使用完整的 policy episode，而不是首次失败检测处截断的前缀。

## 旧的 26-prefix 对照实验命令

下面的手工命令保留给旧的失败前缀对照实验；当前一键脚本不会调用它们。

先检查奖励与 bin 数学：

```bash
cd /path/to/RLinf/examples/offline_rl/piperx_recap_edited
/path/to/RLinf/.venv/bin/python self_check_edited.py
```

先把现有26条双视角失败前缀导出成 LeRobot rollout：

```bash
cd /path/to/RLinf/examples/offline_rl/piperx_recap_edited
/path/to/miniconda3/envs/lerobot_pi05/bin/python -u \
  export_failure_prefixes_edited.py \
  --input /path/to/genesis-world/vla/stacking/data/pi05_ik_corrections_35_fixed_20260923_204117_raw \
  --output /path/to/genesis-world/vla/stacking/data/pi05_recap_failure_prefixes_26_lerobot \
  --repo-id local/piperx_recap_failure_prefixes_26
```

然后为成功示教和失败 rollout 写 return sidecar：

```bash
cd /path/to/RLinf/examples/offline_rl/piperx_recap_edited
/path/to/RLinf/.venv/bin/python -u compute_returns_edited.py \
  --config config_edited.yaml
```

随后在前台训练独立的 value VLM 副本：

```bash
cd /path/to/RLinf/examples/offline_rl/piperx_recap_edited
CUDA_VISIBLE_DEVICES=0 \
/path/to/RLinf/.venv/bin/python -u train_value_edited.py \
  --config config_edited.yaml
```

训练输出位于：

```text
/path/to/rlinf-experiments/piperx_recap_value_edited/
├── checkpoints/
│   ├── step_003000/
│   └── step_006000/
├── latest_checkpoint.json
├── pi05_value_final.pt
├── training_summary.json
└── tensorboard/
```

查看 TensorBoard：

```bash
/path/to/RLinf/.venv/bin/tensorboard \
  --logdir /path/to/rlinf-experiments/piperx_recap_value_edited/tensorboard \
  --host 0.0.0.0 \
  --port 6006
```

训练完成后检查某一帧的 value 预测：

```bash
cd /path/to/RLinf/examples/offline_rl/piperx_recap_edited
CUDA_VISIBLE_DEVICES=0 \
/path/to/RLinf/.venv/bin/python -u predict_value_edited.py \
  --config config_edited.yaml \
  --frame-index 10000
```

## 为什么不能只替换官方 RECAP 的 `model_path`

官方 `recap_value_model.yaml` 建立的是 SigLIP2 + Gemma3-270M + critic expert。当前 π0.5 是 SigLIP + PaliGemma + action expert，checkpoint key、tokenizer、hidden path 和 forward contract 都不同。只把 `model_path` 改成 π0.5 会得到大量 missing/unexpected keys，也没有真正调用你训练过的 π0.5 VLM。EDITED 版本显式通过 `openpi_rlinf.get_model()` 加载当前 checkpoint，并从它真实的 `build_prefix_cache()` 输出训练 value head。
