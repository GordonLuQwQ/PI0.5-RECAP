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

"""Read PiperX LeRobot v3 demonstrations through the OpenPI SFT interface."""

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler

from examples.offline_rl.piperx_recap_edited.data_edited import PiperXPolicyFramesEdited
from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root


def build_piperx_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Load one collection or the released five-task plus supplemental mixture."""
    import openpi.training.data_loader as openpi_data_loader

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    paths = [data_paths] if isinstance(data_paths, (str, Path)) else list(data_paths)
    specs = cfg.data.get("piper_specs")
    if specs is not None and len(specs) != len(paths):
        raise ValueError("piper_specs must have one entry per training dataset")
    batch_size = (
        int(cfg.actor.get("eval_batch_size", cfg.actor.micro_batch_size))
        if eval_dataset
        else int(cfg.actor.micro_batch_size)
    )
    model = cfg.actor.model
    config = get_openpi_config(
        model.openpi.config_name,
        model_path=model.model_path,
        batch_size=batch_size,
        repo_id=str(paths[0]),
        data_kwargs=model.openpi_data,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    sources = []
    for index, path in enumerate(paths):
        raw = PiperXPolicyFramesEdited(
            resolve_lerobot_dataset_root(str(path)),
            action_horizon=int(model.openpi.action_horizon),
        )
        selected = raw
        excluded = (
            list(specs[index].get("exclude_tasks", [])) if specs is not None else []
        )
        if excluded:
            indices = np.flatnonzero(~np.isin(raw.prompts, excluded))
            selected = Subset(raw, indices.tolist())
        sources.append(openpi_data_loader.transform_dataset(selected, data_config))
    dataset = sources[0] if len(sources) == 1 else ConcatDataset(sources)
    sampler = (
        DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=not eval_dataset
        )
        if world_size > 1
        else None
    )
    torch_loader = openpi_data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        shuffle=not eval_dataset and sampler is None,
        sampler=sampler,
        num_workers=int(cfg.data.get("num_workers", 0)),
        seed=int(cfg.actor.seed),
        framework="pytorch",
    )
    return openpi_data_loader.DataLoaderImpl(data_config, torch_loader), data_config


def calculate_piperx_norm_stats(dataset_root: Path) -> Path:
    """Calculate 7D state/action statistics without decoding camera videos."""
    import openpi.shared.normalize as normalize
    import pyarrow.parquet as pq

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    columns = {"state": "observation.state", "actions": "action"}
    for path in sorted((dataset_root / "data").rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=list(columns.values())):
            for key, column in columns.items():
                values = np.asarray(batch.column(column).to_pylist(), dtype=np.float32)
                stats[key].update(values)
    normalize.save(
        dataset_root, {key: stat.get_statistics() for key, stat in stats.items()}
    )
    return dataset_root / "norm_stats.json"
