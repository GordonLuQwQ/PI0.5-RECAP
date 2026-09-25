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

"""Checkpoint helpers for OpenPI_RLinf."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from rlinf.utils.logging import get_logger

logger = get_logger()

FULL_WEIGHTS_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
)

NATIVE_ADAPTER_MANIFEST = "adapter_config.json"
NATIVE_ADAPTER_WEIGHTS = "adapter_weights.safetensors"
NATIVE_ADAPTER_FORMAT = "openpi_rlinf_native_adapter_v1"

_FSDP_WRAPPER_PREFIXES = (
    "_fsdp_wrapped_module.",
    "_orig_mod.",
    "module.",
)

_BARE_PI0_PREFIXES = (
    "llm.",
    "img.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
    "state_proj.",
    "action_time_mlp_in.",
    "action_time_mlp_out.",
    "pointnet.",
)

_OLD_OPENPI_PREFIX = "paligemma_with_expert."
_OLD_WRAPPER_MODEL_PREFIX = "model."


def _missing_pi0_keys(missing_keys) -> list[str]:
    """Return missing keys that belong to the Pi0 backbone, not extra heads."""
    return [key for key in missing_keys if key.startswith(_BARE_PI0_PREFIXES)]


def resolve_model_safetensors(model_path: Any) -> pathlib.Path | None:
    """Resolve a base ``model.safetensors`` checkpoint path."""
    path = pathlib.Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".safetensors"):
        return path
    weights_path = path / "model.safetensors"
    return weights_path if weights_path.exists() else None


def resolve_full_weights(model_path: Any) -> pathlib.Path | None:
    """Resolve an RLinf FSDP ``full_weights.pt`` checkpoint path."""
    path = pathlib.Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".pt"):
        return path
    for rel_path in FULL_WEIGHTS_CANDIDATES:
        candidate = path / rel_path
        if candidate.exists():
            return candidate
    return None


def resolve_native_adapter(model_path: Any) -> dict[str, Any] | None:
    """Resolve a compact native adapter checkpoint and validate its base."""
    path = pathlib.Path(model_path).expanduser()
    if not path.is_dir():
        return None
    manifest_path = path / NATIVE_ADAPTER_MANIFEST
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != NATIVE_ADAPTER_FORMAT:
        raise ValueError(
            f"Unsupported OpenPI_RLinf adapter format in {manifest_path}: "
            f"{manifest.get('format')!r}"
        )
    base_value = manifest.get("base_model_path")
    if not isinstance(base_value, str) or not base_value:
        raise ValueError(f"Adapter manifest has no base_model_path: {manifest_path}")
    base_path = pathlib.Path(base_value).expanduser()
    if not base_path.is_absolute():
        base_path = (path / base_path).resolve()
    weights_name = manifest.get("weights", NATIVE_ADAPTER_WEIGHTS)
    if pathlib.Path(weights_name).name != weights_name:
        raise ValueError(f"Adapter weights must be a filename: {weights_name!r}")
    weights_path = path / weights_name
    if not weights_path.is_file():
        raise FileNotFoundError(f"Adapter weights not found: {weights_path}")
    if resolve_native_adapter(base_path) is not None:
        raise ValueError("Nested native adapter checkpoints are not supported")
    if (
        resolve_model_safetensors(base_path) is None
        and resolve_full_weights(base_path) is None
    ):
        raise FileNotFoundError(f"Adapter base checkpoint has no weights: {base_path}")
    return {
        **manifest,
        "manifest_path": manifest_path,
        "adapter_dir": path,
        "base_model_path": base_path,
        "weights_path": weights_path,
    }


def load_native_adapter(model: Any, adapter: dict[str, Any]) -> None:
    """Overlay a validated compact adapter onto an initialized native Pi0."""
    import safetensors.torch
    import torch

    state_dict = safetensors.torch.load_file(str(adapter["weights_path"]), device="cpu")
    named_parameters = dict(model.named_parameters())
    declared_names = adapter.get("parameter_names")
    if declared_names is not None and set(declared_names) != set(state_dict):
        raise ValueError("Adapter manifest parameter_names do not match its weights")
    with torch.no_grad():
        for name, tensor in state_dict.items():
            parameter = named_parameters.get(name)
            if parameter is None:
                raise KeyError(
                    f"Adapter parameter is absent from this Pi0 shape: {name}"
                )
            if parameter.shape != tensor.shape:
                raise ValueError(
                    f"Adapter parameter shape mismatch for {name}: "
                    f"saved={tuple(tensor.shape)} current={tuple(parameter.shape)}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    logger.info(
        "openpi_rlinf: overlaid %d native adapter tensors from %s",
        len(state_dict),
        adapter["weights_path"],
    )


def save_native_adapter(
    model: Any,
    output_dir: str | pathlib.Path,
    *,
    base_model_path: str | pathlib.Path,
    prompt_suffix: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Save trainable Pi0 parameters without duplicating dense base weights."""
    import safetensors.torch

    output = pathlib.Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Adapter output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state_dict:
        raise ValueError("No trainable parameters are available for adapter saving")
    weights_path = output / NATIVE_ADAPTER_WEIGHTS
    safetensors.torch.save_file(state_dict, str(weights_path))
    manifest = {
        "format": NATIVE_ADAPTER_FORMAT,
        "weights": weights_path.name,
        "base_model_path": str(pathlib.Path(base_model_path).expanduser().resolve()),
        "prompt_suffix": prompt_suffix,
        "parameter_names": sorted(state_dict),
        "parameter_count": sum(tensor.numel() for tensor in state_dict.values()),
        "metadata": metadata or {},
    }
    temporary = output / (NATIVE_ADAPTER_MANIFEST + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / NATIVE_ADAPTER_MANIFEST)
    return output


def _normalize_key(key: str) -> str:
    while True:
        for prefix in _FSDP_WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        else:
            return key


def _normalize_state_dict(state_dict):
    """Map checkpoint keys onto the inherited-Pi0 layout.

    New checkpoints store bare Pi0 keys (``llm.*``). Older wrapper checkpoints
    stored the same tensors under ``model.llm.*``; strip that prefix. RLT and
    extra heads (``rlt_module.*``, ``value_head.*``, …) stay as-is.
    """
    normalized = {}
    for key, tensor in state_dict.items():
        key = _normalize_key(key)
        if key.startswith(_OLD_WRAPPER_MODEL_PREFIX):
            rest = key[len(_OLD_WRAPPER_MODEL_PREFIX) :]
            # Only strip when the remainder is a Pi0 module (or nested FSDP
            # leftover). Algorithm heads never lived under ``model.``.
            if rest.startswith(_BARE_PI0_PREFIXES) or rest.startswith(
                _FSDP_WRAPPER_PREFIXES
            ):
                key = rest
        if key in normalized:
            raise ValueError(
                f"Duplicate checkpoint key after prefix normalization: {key!r}."
            )
        normalized[key] = tensor
    return normalized


def _convert_official_openpi_keys(state_dict, source) -> dict:
    """Rewrite ``paligemma_with_expert.*`` keys onto the inherited-Pi0 layout.

    Extra heads (``value_head.*``, ``rlt_module.*``, …) are kept as-is.
    """
    if not any(key.startswith(_OLD_OPENPI_PREFIX) for key in state_dict):
        return state_dict

    from rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi_rlinf import (
        old_to_new_state_dict,
    )

    converted = old_to_new_state_dict(state_dict)
    for key, tensor in state_dict.items():
        if key.startswith(_OLD_OPENPI_PREFIX) or key in converted:
            continue
        converted[key] = tensor
    logger.info(
        "openpi_rlinf: converted OpenPI PyTorch checkpoint keys from %s in memory",
        source,
    )
    return converted


def load_full_weights(model, weights_path, *, expect_rlt: bool) -> None:
    """Load an RLinf ``full_weights.pt`` checkpoint into a Pi0 (or subclass)."""
    import torch

    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

    loaded = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = _convert_official_openpi_keys(
        _normalize_state_dict(as_state_dict(loaded)),
        weights_path,
    )
    if expect_rlt and not any(key.startswith("rlt_module.") for key in state_dict):
        raise ValueError(
            "openpi_rlinf RLT checkpoint has no rlt_module.* weights. "
            "Stage2 must consume a Stage1 checkpoint trained with openpi.use_rlt=True."
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    matched = len(state_dict) - len(unexpected)
    if matched <= 0:
        raise RuntimeError(
            f"No tensors from {weights_path} matched the openpi_rlinf model. "
            "This usually means the checkpoint is still in the legacy official "
            "OpenPI PyTorch key layout."
        )
    missing_pi0 = _missing_pi0_keys(missing)
    if missing_pi0:
        raise RuntimeError(
            f"Pi0 tensors missing from {weights_path}: {missing_pi0[:8]}"
        )
    if expect_rlt and any(key.startswith("rlt_module.") for key in missing):
        raise RuntimeError(
            f"RLT checkpoint {weights_path} did not load all rlt_module weights; "
            f"missing={missing[:8]}"
        )

    if missing or unexpected:
        logger.warning(
            "openpi_rlinf: loaded checkpoint %s with strict=False "
            "(matched=%d missing=%d unexpected=%d)",
            weights_path,
            matched,
            len(missing),
            len(unexpected),
        )
    else:
        logger.info("openpi_rlinf: loaded full checkpoint from %s", weights_path)


def load_base_safetensors(model, safetensors_path) -> None:
    """Load a base checkpoint, accepting new and legacy OpenPI layouts."""
    import safetensors.torch

    from rlinf.models.embodiment.openpi_rlinf.modules.lora import is_lora_parameter

    state_dict = _convert_official_openpi_keys(
        safetensors.torch.load_file(str(safetensors_path), device="cpu"),
        safetensors_path,
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys loading {safetensors_path}: {unexpected[:8]}"
        )
    # A dense base intentionally has no freshly initialized adapter factors.
    # If an adapter checkpoint is supplied, require every adapter tensor.
    has_adapters = any(is_lora_parameter(key) for key in state_dict)
    missing_pi0 = _missing_pi0_keys(
        [
            key
            for key in incompatible.missing_keys
            if has_adapters or not is_lora_parameter(key)
        ]
    )
    if missing_pi0:
        raise RuntimeError(
            f"Pi0 tensors missing from {safetensors_path}: {missing_pi0[:8]}"
        )
    if incompatible.missing_keys:
        logger.info(
            "openpi_rlinf: loaded base safetensors %s; leaving %d extra module "
            "tensors randomly initialized (%s...)",
            safetensors_path,
            len(incompatible.missing_keys),
            incompatible.missing_keys[:4],
        )
