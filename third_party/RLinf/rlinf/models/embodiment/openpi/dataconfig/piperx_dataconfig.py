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

"""Two-view PiperX observations and absolute joint/closure actions."""

import dataclasses
import pathlib
from typing import Any

import numpy as np
from openpi import transforms
from openpi.models.model import BaseModelConfig
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory


@dataclasses.dataclass(frozen=True)
class PiperXInputs(transforms.DataTransformFn):
    """Map the RLinf camera keys to the camera slots used during PiperX SFT."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        main_image = np.asarray(data["observation/image"])
        wrist_image = np.asarray(data["observation/wrist_image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (7,):
            raise ValueError(
                f"PiperX requires q1..q6 and measured closure; got {state.shape}"
            )
        result = {
            "state": state,
            "image": {
                "base_0_rgb": main_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(main_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "prompt": str(data["prompt"]),
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32)
        return result


@dataclasses.dataclass(frozen=True)
class PiperXOutputs(transforms.DataTransformFn):
    """Keep six absolute joint targets in radians and continuous closure."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "actions": np.asarray(data["actions"])[..., :7]}


@dataclasses.dataclass(frozen=True)
class PiperXDataConfig(DataConfigFactory):
    """Use the training normalization, camera ordering and discrete state prompt."""

    def create(
        self, assets_dirs: pathlib.Path, model_config: BaseModelConfig
    ) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=transforms.Group(
                inputs=[PiperXInputs()], outputs=[PiperXOutputs()]
            ),
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("actions",),
        )
