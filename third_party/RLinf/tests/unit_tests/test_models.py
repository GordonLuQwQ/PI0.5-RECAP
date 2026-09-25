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

"""Model registration, embeddings, and the reward-model helpers."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models import get_model, register_model
from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTTokenTransformer,
)
from rlinf.scheduler import Worker
from rlinf.utils.env_helpers import HistoryManager
from rlinf.utils.env_helpers.delay_sampler import (
    ConstantDelaySampler,
    DelaySampler,
    ExponentialDelaySampler,
    GaussianDelaySampler,
    UniformDelaySampler,
)


@pytest.fixture
def native_pi05_modules(monkeypatch):
    """Run the native model at small sizes, keeping its production forwards."""
    import dataclasses

    from rlinf.models.embodiment.openpi_rlinf.modules import gemma, siglip

    original = gemma.get_config

    def small_config(variant):
        cfg = original(variant)
        width = 16 if "300m" in variant else 32
        return dataclasses.replace(
            cfg,
            width=width,
            depth=2,
            mlp_dim=width * 2,
            num_heads=4,
            num_kv_heads=1,
            head_dim=8,
        )

    monkeypatch.setattr(gemma, "get_config", small_config)
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 128)
    monkeypatch.setattr(
        siglip,
        "_decode_variant",
        lambda _: {
            "width": 32,
            "depth": 1,
            "num_heads": 4,
            "mlp_dim": 64,
            "patch_size": (56, 56),
        },
    )
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old_threads)


def _native_pi05_base():
    from rlinf.models.embodiment.openpi_rlinf.pi0 import Pi0
    from rlinf.models.embodiment.openpi_rlinf.pi0_config import Pi0Config

    model = Pi0(
        Pi0Config(
            pi05=True, dtype="float32", action_dim=8, action_horizon=3, max_token_len=8
        )
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "ada_modulation" in name or name.startswith("img.head."):
                parameter.normal_(std=0.1)
    return model


def _native_pi05_batch():
    from rlinf.models.embodiment.openpi_rlinf.modules.model import Observation

    images = {
        name: torch.rand(1, 224, 224, 3) * 2 - 1
        for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    obs = Observation(
        images=images,
        image_masks={
            name: torch.tensor([name != "right_wrist_0_rgb"]) for name in images
        },
        state=torch.zeros(1, 8),
        tokenized_prompt=torch.tensor([[1, 2, 3, 4]]),
        tokenized_prompt_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    return obs, torch.randn(1, 3, 8)


def _native_pi05_lora_from_base(tmp_path, base):
    import safetensors.torch

    safetensors.torch.save_file(base.state_dict(), str(tmp_path / "model.safetensors"))
    cfg = OmegaConf.create(
        {
            "model_type": "openpi_rlinf",
            "model_path": str(tmp_path),
            "precision": "fp32",
            "pi05": True,
            "is_lora": True,
            "lora_rank": 32,
            "load_to_device": False,
            "action_dim": 7,
            "num_action_chunks": 3,
            "num_steps": 2,
            "openpi": {
                "task": "sft",
                "action_horizon": 3,
                "model_action_dim": 8,
                "max_token_len": 8,
                "paligemma_variant": "gemma_2b_lora",
                "action_expert_variant": "gemma_300m_lora",
                "lora_train_vision": True,
            },
        }
    )
    return get_model(cfg)


def test_native_pi05_lora_zero_init_and_both_expert_gradients(
    native_pi05_modules, tmp_path
):
    from rlinf.models.embodiment.openpi_rlinf.modules.lora import is_lora_parameter

    torch.manual_seed(11)
    base = _native_pi05_base()
    adapted = _native_pi05_lora_from_base(tmp_path, base)
    obs, actions = _native_pi05_batch()
    noise, time = torch.randn_like(actions), torch.tensor([0.4])
    expected = base.compute_loss(obs, actions, noise=noise, time=time)
    actual = adapted.compute_loss(obs, actions, noise=noise, time=time)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.mean().backward()
    for expert in (0, 1):
        for part in (f"attn.k_proj.{expert}.", f"mlps.{expert}."):
            grads = [
                p.grad.abs().sum()
                for n, p in adapted.named_parameters()
                if part in n and is_lora_parameter(n) and p.grad is not None
            ]
            assert grads and sum(grads) > 0
    assert adapted.img.stem.weight.grad.abs().sum() > 0
    for name, parameter in adapted.llm.named_parameters():
        if not is_lora_parameter(name):
            assert not parameter.requires_grad and parameter.grad is None
    with torch.no_grad():
        adapted.llm.layers[0].attn.v_proj[1].lora_b.normal_(std=0.1)
    changed = adapted.compute_loss(obs, actions, noise=noise, time=time)
    assert not torch.allclose(actual, changed)
    restored = _native_pi05_lora_from_base(tmp_path, base)
    restored.load_state_dict(adapted.state_dict(), strict=True)
    torch.testing.assert_close(
        restored.compute_loss(obs, actions, noise=noise, time=time), changed
    )


def test_native_pi05_lora_sft_backward(native_pi05_modules, tmp_path):
    from rlinf.models.embodiment.base_policy import ForwardType

    model = _native_pi05_lora_from_base(tmp_path, _native_pi05_base())
    loss = model(forward_type=ForwardType.SFT, data=_native_pi05_batch())
    loss.backward()
    assert torch.isfinite(loss)
    for expert in (0, 1):
        grad = model.llm.layers[0].attn.v_proj[expert].lora_b.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_native_pi05_positive_advantage_prompt_routing():
    from rlinf.models.embodiment.openpi_rlinf.modules.model import Observation
    from rlinf.models.embodiment.openpi_rlinf.pi0 import (
        route_positive_advantage_prompt,
    )

    base_tokens = torch.tensor([[1, 2], [3, 4], [5, 6]])
    base_masks = torch.tensor([[True, True], [True, True], [True, False]])
    observation = Observation(
        images={},
        image_masks={},
        state=torch.zeros(3, 2),
        tokenized_prompt=base_tokens,
        tokenized_prompt_mask=base_masks,
    )
    positive_tokens = base_tokens + 10
    positive_masks = torch.ones_like(base_masks)
    routed, routing = route_positive_advantage_prompt(
        observation,
        positive_tokenized_prompt=positive_tokens,
        positive_tokenized_prompt_mask=positive_masks,
        advantage=torch.tensor([True, True, False]),
        unconditional_probability=0.1,
        random_values=torch.tensor([0.05, 0.5, 0.9]),
    )

    assert routing["conditional"].tolist() == [False, True, False]
    torch.testing.assert_close(
        routed.tokenized_prompt,
        torch.stack([base_tokens[0], positive_tokens[1], base_tokens[2]]),
    )
    torch.testing.assert_close(
        routed.tokenized_prompt_mask,
        torch.stack([base_masks[0], positive_masks[1], base_masks[2]]),
    )
    torch.testing.assert_close(observation.tokenized_prompt, base_tokens)


def test_native_pi05_advantage_sft_contract(native_pi05_modules):
    model = _native_pi05_base()
    observation, actions = _native_pi05_batch()
    data = {
        "observation": observation,
        "actions": actions,
        "advantage": torch.tensor([True]),
        "positive_tokenized_prompt": observation.tokenized_prompt + 10,
        "positive_tokenized_prompt_mask": observation.tokenized_prompt_mask,
    }

    with pytest.raises(ValueError, match="unconditional_probability is required"):
        model.sft_forward(data=data)

    result = model.sft_forward(
        data=data,
        advantage_unconditional_probability=0.1,
        advantage_random_values=torch.tensor([0.5]),
    )
    assert torch.isfinite(result["loss"])
    assert result["advantage_positive_fraction"].item() == 1.0
    assert result["advantage_conditional_fraction"].item() == 1.0
    assert result["advantage_positive_unconditional_fraction"].item() == 0.0


def test_native_pi05_base_requires_all_dense_weights(native_pi05_modules, tmp_path):
    import safetensors.torch

    from rlinf.models.embodiment.openpi_rlinf.checkpoint import load_base_safetensors

    base = _native_pi05_base()
    model = _native_pi05_lora_from_base(tmp_path, base)
    state = base.state_dict()
    del state["llm.layers.0.attn.q_proj.1.weight"]
    path = tmp_path / "incomplete.safetensors"
    safetensors.torch.save_file(state, str(path))
    with pytest.raises(RuntimeError, match="Pi0 tensors missing"):
        load_base_safetensors(model, path)
    state = model.state_dict()
    del state["llm.layers.0.attn.q_proj.1.lora_b"]
    safetensors.torch.save_file(state, str(path))
    with pytest.raises(RuntimeError, match="Pi0 tensors missing"):
        load_base_safetensors(model, path)


def test_native_pi05_adapter_round_trip(tmp_path):
    from rlinf.models.embodiment.openpi_rlinf.checkpoint import (
        load_native_adapter,
        resolve_native_adapter,
        save_native_adapter,
    )

    base = tmp_path / "base"
    base.mkdir()
    (base / "full_weights.pt").touch()
    model = torch.nn.Linear(3, 2)
    expected = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    output = save_native_adapter(
        model,
        tmp_path / "adapter",
        base_model_path=base,
        prompt_suffix="\nAdvantage: positive",
    )

    adapter = resolve_native_adapter(output)
    assert adapter is not None
    assert adapter["base_model_path"] == base
    assert adapter["prompt_suffix"] == "\nAdvantage: positive"
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    load_native_adapter(model, adapter)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)


def test_native_pi05_adapter_prompt_suffix_is_applied(monkeypatch):
    from rlinf.models.embodiment.openpi_rlinf import env_io

    model = env_io.EnvIO()
    model._input_transform_fn = object()
    model._output_transform_fn = object()
    model.config_name = "pi05_piperx"
    model.state_indices = ()
    model.prompt_suffix = "\nAdvantage: positive"
    monkeypatch.setattr(
        env_io,
        "repack_env_obs",
        lambda *args, **kwargs: {"prompt": np.asarray(["perform task"])},
    )
    monkeypatch.setattr(
        model, "input_transform", lambda observation, transpose: observation
    )
    monkeypatch.setattr(
        model, "_observation_dict_to_device", lambda observation: observation
    )

    observation = model.env_obs_to_observation({})

    assert observation["prompt"] == ["perform task\nAdvantage: positive"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP1 needs a GPU")
def test_native_pi05_lora_fsdp_checkpoint(native_pi05_modules, tmp_path):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
    from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
    from rlinf.models.embodiment.base_policy import ForwardType

    module = _native_pi05_lora_from_base(tmp_path, _native_pi05_base()).cuda()
    expected = {key: value.cpu().clone() for key, value in module.state_dict().items()}
    config = OmegaConf.create(
        {"wrap_policy": {"transformer_layer_cls_to_wrap": ["Block", "Encoder1DBlock"]}}
    )
    policy = get_fsdp_wrap_policy(
        module, config, is_lora=True, model_type="openpi_rlinf"
    )
    torch.distributed.init_process_group(
        "nccl", rank=0, world_size=1, init_method=f"file://{tmp_path}/rendezvous"
    )
    try:
        model = FSDP(
            module,
            auto_wrap_policy=policy,
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            device_id=0,
        )
        optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
        # Initialize moment buffers without updating the model.
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                optimizer.state[parameter] = {
                    "step": torch.tensor(0.0),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(forward_type=ForwardType.SFT, data=_native_pi05_batch())
        loss.backward()
        checkpoint = str(tmp_path / "checkpoint")
        FSDPStrategy.save_checkpoint(model, optimizer, scheduler, checkpoint)
        model.zero_grad(set_to_none=True)
        FSDPStrategy.load_checkpoint(model, optimizer, scheduler, checkpoint)
        actual = torch.load(
            tmp_path / "checkpoint/model_state_dict/full_weights.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert actual.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            torch.testing.assert_close(
                module.state_dict()[key].cpu(), expected[key], rtol=0, atol=0
            )
    finally:
        torch.distributed.destroy_process_group()


class _DummyModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class _DummyFSDPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _DummyBlock()
        self.head = torch.nn.Linear(4, 2)
        self.head._fsdp_wrap_name = "custom_head"


def test_custom_model_registration_smoke():
    model_type = f"custom_model_smoke_{int(time.time() * 1000)}"
    received = {"torch_dtype": None}

    def _builder(cfg, torch_dtype):
        received["torch_dtype"] = torch_dtype
        return _DummyModel()

    register_model(model_type, _builder, category="embodied")

    supported_model = SupportedModel(model_type)
    assert supported_model.value == model_type

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    model = get_model(cfg)

    assert isinstance(model, _DummyModel)
    assert received["torch_dtype"] == torch.float32


def test_custom_model_registration_with_fsdp_wrap_policy():
    model_type = f"custom_model_fsdp_{int(time.time() * 1000)}"

    def _builder(cfg, torch_dtype):
        return _DummyFSDPModel()

    register_model(
        model_type,
        _builder,
        category="embodied",
    )

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    fsdp_cfg = OmegaConf.create(
        {
            "wrap_policy": {
                "transformer_layer_cls_to_wrap": ["_DummyBlock"],
                "module_classes_to_wrap": ["_DummyBlock"],
                "no_split_names": ["custom_head"],
            },
            "use_orig_params": True,
        }
    )
    model = get_model(cfg)
    wrap_policy = get_fsdp_wrap_policy(
        module=model,
        config=fsdp_cfg,
        is_lora=False,
        model_type=model_type,
    )

    assert wrap_policy is not None
    assert wrap_policy(module=model.block, recurse=False, nonwrapped_numel=0)
    assert wrap_policy(module=model.head, recurse=False, nonwrapped_numel=0)


def _make_model(*, prefix_seq_len: int = 5) -> RLTTokenTransformer:
    torch.manual_seed(0)
    return RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        prefix_seq_len=prefix_seq_len,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
    )


def test_decoder_causal_mask_blocks_future_teacher_targets():
    model = _make_model()
    model.eval()
    rl_tokens = torch.randn(1, 1, model.embed_dim)
    targets = torch.randn(1, model.prefix_seq_len, model.input_dim)

    changed_targets = targets.clone()
    changed_targets[:, 2:] += 100.0

    original_output = model.decode(rl_tokens, targets)
    changed_output = model.decode(rl_tokens, changed_targets)

    # target[2:] enters decoder positions 3+, so positions 0..2 must not
    # change when causal attention prevents access to future positions.
    torch.testing.assert_close(
        original_output[:, :3],
        changed_output[:, :3],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(original_output[:, 3:], changed_output[:, 3:])


def test_loss_masks_trailing_padding():
    model = _make_model(prefix_seq_len=4)
    model.eval()
    prefix_embs = torch.randn(2, 4, model.input_dim)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    loss, _ = model.loss(prefix_embs, mask)
    reconstructed, _ = model.reconstruct(prefix_embs, mask)
    valid = mask.unsqueeze(-1).to(dtype=torch.float32)
    expected_loss = (
        torch.square(reconstructed.float() - prefix_embs.float()) * valid
    ).sum() / (valid.sum() * model.input_dim)
    torch.testing.assert_close(loss, expected_loss)

    changed_padding = prefix_embs.clone()
    changed_padding[~mask] += 1000.0
    changed_loss, _ = model.loss(changed_padding, mask)
    torch.testing.assert_close(loss, changed_loss, rtol=1e-5, atol=1e-5)


def test_reconstruct_output_shape_matches_prefix_embeddings():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(3, 4, model.input_dim)

    reconstructed, _ = model.reconstruct(prefix_embs)

    assert reconstructed.shape == prefix_embs.shape


def test_reconstruct_detaches_targets_but_trains_encoder_and_decoder():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(2, 4, model.input_dim, requires_grad=True)

    loss, _ = model.loss(prefix_embs)
    loss.backward()

    assert prefix_embs.grad is None
    encoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    )
    decoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.decoder.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad_norm > 0
    assert decoder_grad_norm > 0


class _FakeValueExpert:
    def __init__(self, image_emb, lang_emb):
        self.image_emb = image_emb
        self.lang_emb = lang_emb

    def embed_image(self, image):
        return self.image_emb.to(device=image.device)

    def embed_language_tokens(self, tokens):
        return self.lang_emb.to(device=tokens.device)


def _load_value_critic_model(monkeypatch):
    value_model_dir = (
        Path(__file__).resolve().parents[2]
        / "rlinf/models/embodiment/value_model/recap"
    )
    package_name = "value_model_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(value_model_dir)]
    monkeypatch.setitem(sys.modules, package_name, package)

    module_name = f"{package_name}.modeling_critic"
    spec = importlib.util.spec_from_file_location(
        module_name,
        value_model_dir / "modeling_critic.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module.ValueCriticModel


def test_value_model_does_not_rescale_gemma3_language_embeddings(monkeypatch):
    """Gemma3 embed_tokens already applies sqrt(hidden_size) internally."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.Gemma3ForCausalLM")

    ValueCriticModel = _load_value_critic_model(monkeypatch)

    hidden_size = 4
    image_emb = torch.zeros(1, 2, hidden_size)
    lang_emb = torch.arange(12, dtype=torch.float32).reshape(1, 3, hidden_size)

    model = SimpleNamespace(
        gradient_checkpointing_enabled=False,
        training=False,
        value_expert=_FakeValueExpert(image_emb=image_emb, lang_emb=lang_emb),
        _apply_checkpoint=lambda func, *args: func(*args),
    )

    prefix_embs, prefix_pad_masks = ValueCriticModel.embed_prefix(
        model,
        images=[torch.empty(1, 3, 8, 8)],
        img_masks=[torch.tensor([True])],
        lang_tokens=torch.tensor([[1, 2, 3]]),
        lang_masks=torch.tensor([[True, True, False]]),
    )

    torch.testing.assert_close(prefix_embs[:, 2:], lang_emb)
    torch.testing.assert_close(
        prefix_pad_masks,
        torch.tensor([[True, True, True, True, False]]),
    )


_STARVLA_UTILS_DIR = (
    Path(__file__).resolve().parents[2] / "rlinf/models/embodiment/starvla/utils"
)
_FRANKA_ACTION_STATS = {
    "q01": [-0.5] * 7,
    "q99": [0.5] * 7,
    "min": [-1.0] * 7,
    "max": [1.0] * 7,
    "mask": [True] * 6 + [False],
}


def _load_starvla_util(name: str) -> ModuleType:
    # The starvla package __init__ imports starVLA, which only its venv has.
    spec = importlib.util.spec_from_file_location(
        f"starvla_{name}_under_test", _STARVLA_UTILS_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("source", "bound"), [("q01q99", 0.5), ("minmax", 1.0)])
def test_starvla_action_stats_follow_the_configured_source(source, bound):
    action_space = _load_starvla_util("action_space")
    model = SimpleNamespace(norm_stats={"franka": {"action": _FRANKA_ACTION_STATS}})

    stats = action_space.resolve_action_norm_stats(
        model, "franka", action_dim=7, action_stats_source=source
    )

    np.testing.assert_array_equal(stats["q99"], [bound] * 7)
    np.testing.assert_array_equal(stats["q01"], [-bound] * 7)
    np.testing.assert_array_equal(stats["mask"], [True] * 6 + [False])


def test_starvla_action_stats_name_the_available_keys_for_an_unknown_key():
    action_space = _load_starvla_util("action_space")
    model = SimpleNamespace(norm_stats={"franka": {"action": _FRANKA_ACTION_STATS}})

    with pytest.raises(RuntimeError, match=r"available keys: \['franka'\]"):
        action_space.resolve_action_norm_stats(model, "libero_spatial", action_dim=7)


def test_starvla_action_stats_require_a_norm_stats_mapping():
    action_space = _load_starvla_util("action_space")

    with pytest.raises(RuntimeError, match="no usable 'norm_stats' mapping"):
        action_space.resolve_action_norm_stats(
            SimpleNamespace(norm_stats=None), "franka", action_dim=7
        )


def test_starvla_env_actions_keep_their_shape_and_map_the_libero_gripper(monkeypatch):
    action_space = _load_starvla_util("action_space")
    received_shapes = []

    def unnormalize_actions(actions, action_norm_stats):
        received_shapes.append(actions.shape)
        return actions

    tools = ModuleType("starVLA.model.tools")
    tools.FrameworkTools = SimpleNamespace(unnormalize_actions=unnormalize_actions)
    monkeypatch.setitem(sys.modules, "starVLA.model.tools", tools)

    normalized = np.zeros((2, 3, 7), dtype=np.float32)
    normalized[..., 0] = 0.25
    normalized[0, :, 6] = 1.0
    stats = {"q99": np.ones(7), "q01": -np.ones(7), "mask": np.ones(7, dtype=bool)}

    env_actions = action_space.unnormalize_actions_for_env(
        normalized, stats, policy_setup="libero"
    )

    # starVLA unnormalizes [T, action_dim]; the chunk layout comes back intact.
    assert received_shapes == [(6, 7)]
    assert env_actions.shape == (2, 3, 7)
    np.testing.assert_array_equal(env_actions[..., 0], 0.25)
    # LIBERO wants the 0/1 gripper as -1 (open) / +1 (closed).
    np.testing.assert_array_equal(env_actions[0, :, 6], -1.0)
    np.testing.assert_array_equal(env_actions[1, :, 6], 1.0)


def test_starvla_autocast_targets_the_worker_accelerator(monkeypatch):
    accelerator = _load_starvla_util("accelerator")
    # CPU stands in for a non-CUDA accelerator such as an Ascend NPU.
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")

    with accelerator.accelerator_autocast(torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16


def test_starvla_autocast_is_a_noop_without_an_accelerator(monkeypatch):
    accelerator = _load_starvla_util("accelerator")
    monkeypatch.setattr(Worker, "torch_device_type", None)

    with accelerator.accelerator_autocast(torch.bfloat16):
        assert not torch.is_autocast_enabled("cpu")
        assert not torch.is_autocast_enabled("cuda")


def test_starvla_gaussian_is_float32_and_keeps_the_gradient_path():
    accelerator = _load_starvla_util("accelerator")
    mean = torch.zeros(2, 3, dtype=torch.bfloat16, requires_grad=True)
    log_std = torch.nn.Parameter(torch.zeros(3))

    dist = accelerator.build_gaussian(mean, log_std.exp())
    sample = dist.rsample()

    assert dist.loc.dtype == dist.scale.dtype == sample.dtype == torch.float32
    dist.log_prob(sample.detach()).sum().backward()
    assert mean.grad is not None and mean.grad.dtype == torch.bfloat16
    assert log_std.grad is not None


class _QwenVisionPatchEmbed(torch.nn.Module):
    """Shape contract of Qwen2.5-VL PatchEmbed: Conv3d kernel == stride."""

    def __init__(self, in_channels=3, temporal=2, patch=4, embed_dim=8):
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal
        self.patch_size = patch
        kernel = (temporal, patch, patch)
        self.proj = torch.nn.Conv3d(
            in_channels, embed_dim, kernel_size=kernel, stride=kernel, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(self.proj.weight.dtype))
        return hidden_states.view(-1, self.proj.out_channels)


def test_qwen_vl_linear_patch_embed_matches_conv3d_and_backprops():
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        _linear_patch_embed_forward,
    )

    torch.manual_seed(0)
    module = _QwenVisionPatchEmbed()
    patches = torch.randn(5, 3 * 2 * 4 * 4, requires_grad=True)

    conv_out = module(patches)
    linear_out = _linear_patch_embed_forward(module, patches)
    torch.testing.assert_close(linear_out, conv_out, rtol=1e-5, atol=1e-5)

    linear_out.sum().backward()
    assert module.proj.weight.grad is not None
    assert patches.grad is not None


def test_qwen_vl_linear_patch_embed_is_rebound_on_npu(monkeypatch):
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        _linear_patch_embed_forward,
        patch_vision_patch_embed,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    model = torch.nn.Sequential(_QwenVisionPatchEmbed())
    original_forward = model[0].forward

    assert patch_vision_patch_embed(model) == 1
    assert model[0].forward.__func__ is _linear_patch_embed_forward
    assert original_forward.__func__ is not _linear_patch_embed_forward


def test_qwen_vl_linear_patch_embed_is_left_alone_on_nvidia(monkeypatch):
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        patch_vision_patch_embed,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    model = torch.nn.Sequential(_QwenVisionPatchEmbed())

    assert patch_vision_patch_embed(model) == 0
    assert model[0].forward.__func__ is _QwenVisionPatchEmbed.forward


_WAN_NPU_PATCHES = "rlinf.envs.sim.world_model.backend.npu_patches"


@pytest.fixture
def wan_dit(monkeypatch):
    """diffsynth's Wan DiT module, whose operators the NPU patches rebind."""
    from rlinf.utils.patcher import Patcher

    def flash_attention(q, k, v, num_heads, compatibility_mode=False):
        return q

    def rope_apply(x, freqs, num_heads):
        return x

    class RMSNorm(torch.nn.Module):
        def forward(self, x):
            return x

    dit = ModuleType("diffsynth.models.wan_video_dit")
    dit.flash_attention, dit.rope_apply, dit.RMSNorm = (
        flash_attention,
        rope_apply,
        RMSNorm,
    )
    for name in ("diffsynth", "diffsynth.models"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, dit.__name__, dit)
    yield dit
    Patcher.clear()


def _import_wan_npu_patches(monkeypatch, *, mindiesd: bool) -> ModuleType:
    """Import the Wan NPU patches on an Ascend stack with or without MindIE-SD."""
    vendor = {"torch_npu": ModuleType("torch_npu"), "mindiesd": None}
    if mindiesd:
        names = (
            "mindiesd",
            "mindiesd.layers",
            "mindiesd.layers.flash_attn",
            "mindiesd.layers.flash_attn.attention_forward",
        )
        vendor.update({name: ModuleType(name) for name in names})
        vendor["mindiesd"].rotary_position_embedding = lambda *args, **kwargs: None
        vendor[names[-1]].attention_forward = lambda *args, **kwargs: None
    for name, module in vendor.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.find_spec(_WAN_NPU_PATCHES)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _WAN_NPU_PATCHES, module)
    spec.loader.exec_module(module)
    return module


def _patch_like_wan_backend(npu_patches: ModuleType) -> None:
    """Run the patch sequence ``WanBackend._build_pipeline`` runs."""
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    npu_patches.apply_npu_patches(Patcher)
    Patcher.apply()


def _wan_operators(dit: ModuleType) -> tuple:
    return dit.flash_attention, dit.rope_apply, dit.RMSNorm.forward


def test_wan_npu_patches_leave_diffsynth_alone_on_nvidia(wan_dit, monkeypatch):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=True)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    operators = _wan_operators(wan_dit)

    _patch_like_wan_backend(npu_patches)
    assert _wan_operators(wan_dit) == operators


def test_wan_npu_patches_log_why_mindiesd_is_unavailable(wan_dit, monkeypatch, caplog):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=False)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    operators = _wan_operators(wan_dit)

    _patch_like_wan_backend(npu_patches)
    assert _wan_operators(wan_dit) == operators
    assert "import of mindiesd halted" in caplog.text


def test_wan_npu_patches_rebind_the_dit_operators_on_every_build(wan_dit, monkeypatch):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=True)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    kernels = (
        npu_patches.npu_flash_attention,
        npu_patches.npu_rope_apply,
        npu_patches.npu_rmsnorm_forward,
    )

    # Every WanBackend in the process repeats the sequence on the patched module.
    for _ in range(2):
        _patch_like_wan_backend(npu_patches)
        assert _wan_operators(wan_dit) == kernels


def _history_cfg():
    return OmegaConf.create(
        {
            "model": {
                "history_buffers": {
                    "main": {
                        "history_size": 2,
                        "min_history_size": 1,
                        "input_interval": 3,
                        "history_keys": ["main_images"],
                        "input_on_done": True,
                    }
                }
            }
        }
    )


def _append_step(manager: HistoryManager, value: int) -> None:
    manager.append_to_history_entries(
        {"main_images": torch.tensor([[value], [value + 10]])}
    )


def test_build_history_input_skips_between_interval_ticks():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_input == {}
    assert history_length == {}
    assert manager.history_counts == [2, 2]


def test_build_history_input_emits_on_interval_tick():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)
    _append_step(manager, 3)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_length == {"main": [2, 2]}
    assert history_input["main"]["main_images"][0] == [
        torch.tensor([2]),
        torch.tensor([3]),
    ]
    assert history_input["main"]["main_images"][1] == [
        torch.tensor([12]),
        torch.tensor([13]),
    ]


def _success_potential_state_machine():
    from rlinf.models.embodiment.reward.vlm_reward_model import (
        ShapedVLMRewardModel,
    )

    model = ShapedVLMRewardModel.__new__(ShapedVLMRewardModel)
    model.potential_gamma = 1.0
    model.potential_scale = 1.0
    model.potential_ema_alpha = 0.5
    model.potential_clip = 0.0
    model.success_threshold = 0.5
    model.success_bonus = 1.0
    model.success_confirmation_windows = 1
    model.gt_success_bonus = 0.0
    model.infer_micro_batch_size = 0
    model._previous_potentials = None
    model._success_fired = None
    model._success_streak = None
    return model


def test_empty_history_input_still_resets_shaping_state_on_done():
    model = _success_potential_state_machine()
    model._previous_potentials = torch.tensor([0.4, 0.8])
    model._success_fired = torch.tensor([True, True])
    model._success_streak = torch.tensor([3, 1], dtype=torch.int32)

    rewards = model.compute_reward(
        {
            "history_input": {},
            "dones": torch.tensor([True, False]),
        }
    )

    assert rewards.tolist() == pytest.approx([0.0, 0.0])
    assert torch.isnan(model._previous_potentials[0])
    assert float(model._previous_potentials[1]) == pytest.approx(0.8)
    assert model._success_fired.tolist() == [False, True]
    assert model._success_streak.tolist() == [0, 1]


VALUE_CLIP = 0.2
HUBER_DELTA = 10.0


def _critic_metrics(values, prev_values, returns, loss_mask=None):
    _, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=loss_mask,
    )
    return metrics


def test_value_clip_ratio_is_zero_when_no_update_is_clipped():
    prev_values = torch.zeros(4, 8)
    values = torch.full((4, 8), VALUE_CLIP / 2)
    returns = torch.zeros(4, 8)

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_clip_ratio_reports_the_fraction_of_clipped_updates():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    # Half of the entries move outside the trust region, half stay inside.
    values = torch.full((4, 8), VALUE_CLIP / 2)
    values[:, :4] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_grows_with_the_size_of_the_value_update():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)

    ratios = [
        float(
            _critic_metrics(torch.full((4, 8), scale), prev_values, returns)[
                "critic/value_clip_ratio"
            ]
        )
        for scale in (0.5 * VALUE_CLIP, 2 * VALUE_CLIP)
    ]

    assert ratios == [pytest.approx(0.0), pytest.approx(1.0)]


def test_value_clip_ratio_ignores_masked_out_entries():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    loss_mask[:, :2] = True

    # Every valid entry is clipped; every padded entry is not.
    values = torch.zeros(4, 8)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(1.0)


def test_value_clip_ratio_broadcasts_a_narrower_loss_mask():
    prev_values = torch.zeros(4, 8, 3)
    returns = torch.zeros(4, 8, 3)
    loss_mask = torch.zeros(4, 8, 1, dtype=torch.bool)
    loss_mask[:, :4] = True

    values = torch.zeros(4, 8, 3)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    # 2 of the 4 unmasked steps are clipped.
    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_is_zero_when_every_entry_is_masked_out():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    values = torch.full((4, 8), 10 * VALUE_CLIP)

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_loss_is_unchanged_by_the_metric_computation():
    torch.manual_seed(0)
    prev_values = torch.randn(4, 8)
    values = torch.randn(4, 8, requires_grad=True)
    returns = torch.randn(4, 8)

    loss, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=None,
    )

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -VALUE_CLIP, VALUE_CLIP
    )
    expected = torch.max(
        torch.nn.functional.huber_loss(
            values, returns, delta=HUBER_DELTA, reduction="none"
        ),
        torch.nn.functional.huber_loss(
            value_pred_clipped, returns, delta=HUBER_DELTA, reduction="none"
        ),
    ).mean()

    assert float(loss.detach()) == pytest.approx(float(expected.detach()), abs=1e-6)
    assert loss.requires_grad
    assert not metrics["critic/value_clip_ratio"].requires_grad


def test_create_builds_expected_sampler_types():
    constant = DelaySampler.create(
        OmegaConf.create({"type": "constant", "delay": 0.12})
    )
    uniform = DelaySampler.create(
        OmegaConf.create({"type": "uniform", "min_delay": 0.03, "max_delay": 0.08})
    )
    exponential = DelaySampler.create(
        OmegaConf.create({"type": "exponential", "rate": 0.5})
    )
    gaussian = DelaySampler.create(
        OmegaConf.create({"type": "gaussian", "mean": 0.20, "stddev": 0.03})
    )

    assert isinstance(constant, ConstantDelaySampler)
    assert isinstance(uniform, UniformDelaySampler)
    assert isinstance(exponential, ExponentialDelaySampler)
    assert isinstance(gaussian, GaussianDelaySampler)


def test_create_accepts_none():
    assert DelaySampler.create(None) is None


def test_same_seed_produces_same_sequence_per_sampler():
    first = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)
    second = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)

    assert first.sample(8) == second.sample(8)


def test_constant_sampler_uses_seconds_helpers():
    sampler = ConstantDelaySampler(delay=0.25)

    assert sampler.sample(3) == [0.25, 0.25, 0.25]
    assert sampler.sample_one() == 0.25


def test_gaussian_sampler_never_returns_negative_seconds():
    sampler = GaussianDelaySampler(mean=0, stddev=0.1, seed=0)

    assert all(delay >= 0 for delay in sampler.sample(100))


def test_invalid_ranges_raise_clear_errors():
    with pytest.raises(ValueError, match="min_delay must be <="):
        UniformDelaySampler(min_delay=0.2, max_delay=0.1)

    with pytest.raises(ValueError, match="rate must be > 0"):
        ExponentialDelaySampler(rate=0)


def test_num_samples_must_be_non_negative_int():
    sampler = ConstantDelaySampler(delay=1)

    with pytest.raises(TypeError, match="num_samples must be an int"):
        sampler.sample(1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_samples must be >= 0"):
        sampler.sample(-1)


class _FakeEnv:
    """Minimal non-gym env exposing the chunk_step/reset surface."""

    def chunk_step(self, *args, **kwargs):
        return "stepped"

    def reset(self, *args, **kwargs):
        return "obs", {}


# Mock gymnasium and its transitive imports for unit-test environments that
# do not install the embodied extras. A minimal gym.Wrapper shim is enough
# because InsertDelay only delegates to self.env.


class _FakeGymEnv:
    pass


class _FakeGymWrapper:
    def __init__(self, env):
        self.env = env


_fake_gym = MagicMock()
_fake_gym.Env = _FakeGymEnv
_fake_gym.Wrapper = _FakeGymWrapper

if "gymnasium" not in sys.modules:
    sys.modules["gymnasium"] = _fake_gym
if "imageio" not in sys.modules:
    sys.modules["imageio"] = MagicMock()


def _delayed_env(delay: float):
    from rlinf.envs.wrappers import InsertDelay

    return InsertDelay(
        _FakeEnv(), OmegaConf.create({"type": "constant", "delay": delay})
    )


def test_chunk_step_does_not_block_the_caller():
    env = _delayed_env(0.5)

    start = time.monotonic()
    assert env.chunk_step() == "stepped"
    elapsed = time.monotonic() - start

    # The delay is sampled, not slept: blocking here would stall the event loop.
    assert elapsed < 0.05


def test_wait_delay_waits_out_the_accumulated_delay():
    env = _delayed_env(0.05)
    env.chunk_step()
    env.chunk_step()

    start = time.monotonic()
    asyncio.run(env.wait_delay())
    elapsed = time.monotonic() - start

    # Both sampled delays are paid, never dropped.
    assert elapsed == pytest.approx(0.1, abs=0.03)


def test_wait_delay_yields_to_other_coroutines():
    env = _delayed_env(0.2)
    env.chunk_step()
    progressed = []

    async def main():
        async def ticker():
            for _ in range(4):
                await asyncio.sleep(0.01)
                progressed.append(1)

        await asyncio.gather(env.wait_delay(), ticker())

    asyncio.run(main())
    # A blocking sleep would have starved the ticker entirely.
    assert len(progressed) == 4


def test_wait_delay_is_a_noop_when_nothing_is_pending():
    env = _delayed_env(0.5)

    start = time.monotonic()
    asyncio.run(env.wait_delay())

    assert time.monotonic() - start < 0.05


def test_delay_metrics_report_every_sample():
    env = _delayed_env(0.03)
    env.chunk_step()
    env.reset()

    metrics = env.insert_delay_metrics()

    assert metrics.tolist() == pytest.approx([0.03, 0.03])
    assert env.insert_delay_metrics().numel() == 0


def test_piperx_inputs_preserve_training_camera_order_and_measured_state():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.dataconfig.piperx_dataconfig import PiperXInputs

    main_image = np.full((16, 16, 3), 37, dtype=np.uint8)
    wrist_image = np.full((16, 16, 3), 183, dtype=np.uint8)
    state = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.25], dtype=np.float32)
    result = PiperXInputs()(
        {
            "observation/image": main_image,
            "observation/wrist_image": wrist_image,
            "observation/state": state,
            "prompt": "put the blue cube on the red cylinder",
        }
    )
    np.testing.assert_array_equal(result["image"]["base_0_rgb"], main_image)
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], wrist_image)
    np.testing.assert_array_equal(result["state"], state)
    assert result["image_mask"] == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": False,
    }
    assert not result["image"]["right_wrist_0_rgb"].any()


def test_piperx_actions_keep_absolute_joint_targets_and_continuous_closure():
    pytest.importorskip("openpi")
    from rlinf.envs.action_utils import prepare_actions
    from rlinf.models.embodiment.openpi.dataconfig.piperx_dataconfig import (
        PiperXOutputs,
    )

    padded = np.zeros((1, 50, 32), dtype=np.float32)
    targets = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -1.8, 0.25], dtype=np.float32)
    padded[..., :7] = targets
    decoded = PiperXOutputs()({"actions": padded})["actions"]
    env_actions = prepare_actions(
        decoded,
        env_type="genesis",
        model_type="openpi_rlinf",
        num_action_chunks=50,
        action_dim=7,
    )
    np.testing.assert_array_equal(
        env_actions.numpy(), np.broadcast_to(targets, (1, 50, 7))
    )


def test_piperx_registration_uses_the_checkpoint_observation_contract():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    config = get_openpi_config("pi05_piperx")
    assert config.model.pi05
    assert config.model.discrete_state_input
    assert config.model.max_token_len == 200
    assert config.model.action_horizon == 50
    assert config.model.action_dim == 32


def test_piperx_env_dispatch_does_not_require_genesis_in_the_model_interpreter():
    from rlinf.envs import get_env_cls
    from rlinf.envs.sim.genesis.piperx_env import PiperXEnv

    cfg = OmegaConf.create({"init_params": {"task_name": "piperx_stacking"}})
    assert get_env_cls("genesis", cfg) is PiperXEnv


def test_piperx_fast_chunk_step_batches_one_request_per_scene():
    from rlinf.envs.sim.genesis.piperx_env import PiperXEnv

    env = object.__new__(PiperXEnv)
    env.cfg = OmegaConf.create({"piperx": {"fast_chunk_step": True}})
    env.num_envs = 4
    env._done = np.zeros(4, dtype=bool)
    env._success = np.zeros(4, dtype=bool)
    env._steps = np.zeros(4, dtype=np.int64)
    env._instructions = [f"task {index}" for index in range(4)]
    env._last_arrays = [None] * 4
    env._last_step = None
    env.logger = MagicMock()
    requests = []

    def rpc_all(operation, arrays_by_slot, slots):
        requests.append((operation, arrays_by_slot, slots))
        return {
            slot: (
                {},
                {
                    "instruction": f"task {slot}",
                    "success": True,
                    "done": True,
                    "steps": 2,
                    "step_successes": [False, True, True],
                    "step_dones": [False, True, True],
                },
            )
            for slot in slots
        }

    env._rpc_all = rpc_all
    env._batched_observation = lambda: {
        "states": torch.zeros(4, 7),
        "task_descriptions": env._instructions,
    }

    observations, rewards, terminations, truncations, infos = env.chunk_step(
        torch.zeros(4, 3, 7)
    )

    assert len(requests) == 1
    assert requests[0][0] == "chunk_step"
    assert requests[0][2] == [0, 1, 2, 3]
    assert all(
        request["actions"].shape == (1, 3, 7) for request in requests[0][1].values()
    )
    assert len(observations) == 3
    torch.testing.assert_close(rewards, torch.tensor([[0.0, 1.0, 1.0]]).expand(4, -1))
    torch.testing.assert_close(
        terminations, torch.tensor([[False, True, True]]).expand(4, -1)
    )
    assert not truncations.any()
    torch.testing.assert_close(
        infos[-1]["episode"]["episode_len"], torch.full((4,), 2.0)
    )


def test_piperx_bridge_accepts_batched_observations():
    from evaluations.piperx.bridge import validate_observation

    arrays = {
        "state": np.zeros((4, 7), dtype=np.float32),
        "third_person": np.zeros((4, 8, 8, 3), dtype=np.uint8),
        "wrist": np.zeros((4, 8, 8, 3), dtype=np.uint8),
    }
    validate_observation(arrays, ("third_person", "wrist"))


def test_eval_episode_done_requires_every_env_to_finish_once_in_chunk():
    from rlinf.workers.env.env_worker import EnvWorker

    assert EnvWorker._all_eval_envs_done(
        torch.tensor([[False, True, True], [True, True, True]])
    )
    assert not EnvWorker._all_eval_envs_done(
        torch.tensor([[False, True, True], [False, False, False]])
    )
    assert not EnvWorker._all_eval_envs_done(None)


def test_rollout_observation_merge_preserves_episode_done_marker():
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    worker = object.__new__(MultiStepRolloutWorker)
    merged = worker._merge_obs_batches(
        [
            {
                "obs": {
                    "states": torch.zeros(1, 7),
                    "task_descriptions": ["put the red cube on the red cylinder"],
                },
                "final_obs": None,
                "eval_episode_done": torch.tensor([True]),
            }
        ]
    )

    torch.testing.assert_close(merged["eval_episode_done"], torch.tensor([True]))


@pytest.mark.parametrize(
    "control_steps,video_stride,expected_frames",
    [(0, 10, 1), (1, 10, 2), (10, 10, 2), (11, 10, 3), (1200, 10, 121)],
)
def test_piperx_video_frame_count_includes_initial_and_final_frames(
    control_steps, video_stride, expected_frames
):
    from evaluations.piperx.bridge import expected_video_frame_count

    assert expected_video_frame_count(control_steps, video_stride) == expected_frames


def test_success_video_frames_are_discarded_without_creating_a_file(
    tmp_path, monkeypatch
):
    settings = ModuleType("settings")
    settings.CALIBRATION = tmp_path / "unused.json"
    settings.CAMERAS = ("third_person", "wrist")
    settings.VIDEO_CAMERAS = ("third_person",)
    settings.VIDEO_FAILURE_ONLY = True
    settings.VIDEO_STRIDE = 10
    monkeypatch.setitem(sys.modules, "settings", settings)

    module_path = Path(__file__).parents[2] / "evaluations/piperx/common.py"
    spec = importlib.util.spec_from_file_location("piperx_common_test", module_path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)

    class FakeEnv:
        protocol = SimpleNamespace(control_hz=20)

        @staticmethod
        def observe(camera_names):
            return {
                name: np.zeros((1, 8, 8, 3), dtype=np.uint8) for name in camera_names
            }

        @staticmethod
        def step(*args, **kwargs):
            return None

    recorded = common.RecordedEnv(FakeEnv(), tmp_path)
    for _ in range(10):
        recorded.step(np.zeros((1, 7), dtype=np.float32))

    assert recorded.finish(keep_video=False) == {}
    assert not (tmp_path / "third_person.mp4").exists()


@pytest.mark.parametrize("history_horizon,executed", [(50, 7), (50, 10)])
def test_rtc_remaining_actions_keep_model_dimensions_and_time_alignment(
    history_horizon, executed
):
    from rlinf.models.embodiment.openpi_rlinf.sampling.rtc_guidance import (
        RTCGuidanceContext,
        build_rtc_target_and_mask,
    )

    previous = torch.arange(2 * history_horizon * 32, dtype=torch.float32).reshape(
        2, history_horizon, 32
    )
    context = RTCGuidanceContext(previous, executed_horizon=executed, delay_steps=5)
    remaining = context.get_prev_remaining()
    torch.testing.assert_close(remaining, previous[:, executed:])
    target, mask = build_rtc_target_and_mask(
        remaining, 50, 32, 5, previous.device, previous.dtype
    )
    assert target.shape == (2, 50, 32) and mask.shape == (2, 50, 1)
    overlap = history_horizon - executed
    torch.testing.assert_close(target[:, :overlap], previous[:, executed:])
    assert torch.all(mask[:, :5] == 1)
    assert torch.all((mask[:, 5:overlap] > 0) & (mask[:, 5:overlap] < 1))
    assert torch.all(torch.diff(mask[:, 5:overlap], dim=1) < 0)
    assert not mask[:, overlap:].any()
    assert not target[:, overlap:].any()


def test_native_rtc_guides_the_pi0_velocity_with_vjp():
    from rlinf.models.embodiment.openpi_rlinf.sampling.rtc_guidance import (
        exact_guidance_weight,
        exact_rtc_velocity,
    )

    class ToyModel:
        @staticmethod
        def run_suffix(observation, x_t, time_batch, kv_cache, prefix_mask):
            return x_t.square()

        @staticmethod
        def velocity_from_suffix(suffix):
            return suffix

    x_t = torch.full((1, 1, 1), 0.5)
    with torch.no_grad():
        guided_velocity = exact_rtc_velocity(
            pi0_model=ToyModel(),
            observation=None,
            x_t=x_t,
            model_t=torch.tensor(0.5),
            target=torch.ones_like(x_t),
            mask=torch.ones((1, 1, 1)),
            kv_cache=(),
            prefix_mask=torch.ones((1, 1), dtype=torch.bool),
            guidance_clip=5.0,
        )

    torch.testing.assert_close(guided_velocity, torch.tensor([[[-0.375]]]))
    schedule = torch.stack(
        [exact_guidance_weight(torch.tensor(index / 10), 5.0) for index in range(10)]
    )
    torch.testing.assert_close(
        schedule,
        torch.tensor(
            [5.0, 5.0, 4.25, 2.7619047, 2.1666667, 2.0, 2.1666667, 2.7619047, 4.25, 5.0]
        ),
    )


def test_piperx_rtc_action_report_compares_only_matched_inference(tmp_path):
    from evaluations.piperx.print_rtc_action_comparison import (
        build_comparison,
        format_comparison,
        save_comparison,
    )

    root = tmp_path / "comparison"
    plain = root / "no_rtc" / "standard"
    rtc = root / "rtc" / "standard"
    for base in (plain, rtc):
        (base / "model_calls").mkdir(parents=True)

    metadata = [
        {
            "call": 0,
            "noise_seed": 42,
            "rtc_context": False,
            "executed_horizon": None,
            "predicted_delay_steps": None,
            "image_sha256": {"main_images": "a", "wrist_images": "b"},
        },
        {
            "call": 1,
            "noise_seed": 43,
            "rtc_context": False,
            "executed_horizon": None,
            "predicted_delay_steps": None,
            "image_sha256": {"main_images": "c", "wrist_images": "d"},
        },
    ]
    rtc_metadata = [dict(row) for row in metadata]
    rtc_metadata[1].update(
        rtc_context=True, executed_horizon=2, predicted_delay_steps=1
    )
    for base, rows in ((plain, metadata), (rtc, rtc_metadata)):
        (base / "model_calls.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    previous = np.arange(35, dtype=np.float32).reshape(1, 5, 7) / 100
    no_rtc = np.zeros((1, 5, 7), dtype=np.float32)
    guided = np.full((1, 5, 7), 0.1, dtype=np.float32)
    state = np.zeros((1, 7), dtype=np.float32)

    def decoded(actions):
        return (actions + 1) / 2 * (1 + 1e-6)

    for base in (plain, rtc):
        np.savez(
            base / "model_calls" / "00000.npz",
            model_actions=previous,
            decoded_actions=decoded(previous),
            measured_state=state,
        )
    np.savez(
        plain / "model_calls" / "00001.npz",
        model_actions=no_rtc,
        decoded_actions=decoded(no_rtc),
        measured_state=state,
    )
    np.savez(
        rtc / "model_calls" / "00001.npz",
        model_actions=guided,
        decoded_actions=decoded(guided),
        measured_state=state,
    )
    norm_path = tmp_path / "norm_stats.json"
    norm_path.write_text(
        json.dumps({"norm_stats": {"actions": {"q01": [0] * 7, "q99": [1] * 7}}})
    )
    (rtc / "env_config.json").write_text(
        json.dumps(
            {
                "piperx": {
                    "model": {
                        "openpi_data": {"norm_stats_path": str(norm_path)},
                        "openpi": {
                            "num_steps": 10,
                            "rtc_guidance_clip": 5.0,
                            "rtc_guidance_mode": "exact",
                        },
                    }
                }
            }
        )
    )

    comparison = build_comparison(root, 1)

    assert comparison.executed_horizon == 2
    assert comparison.delay_steps == 1
    assert comparison.overlap == 3
    np.testing.assert_allclose(
        comparison.target_actions[:, :3], decoded(previous[:, 2:]), rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(comparison.final_delta, 0.05, rtol=0, atol=1e-7)
    assert comparison.mask[0, 0, 0] == 1
    assert np.all((comparison.mask[0, 1:3, 0] > 0) & (comparison.mask[0, 1:3, 0] < 1))
    assert not comparison.mask[0, 3:].any()
    np.testing.assert_allclose(
        comparison.guidance_multipliers,
        comparison.mask[..., None] * comparison.guidance_scales,
        rtol=0,
        atol=0,
    )
    report = format_comparison(comparison)
    assert comparison.guidance_mode == "exact"
    assert "[00] phase=hard" in report
    assert "VJP" in report
    assert "denoise_multipliers = [5." in report
    assert "rtc-no_rtc = [0.05" in report

    text_path, csv_path, npz_path = save_comparison(comparison, tmp_path / "report")
    assert text_path.read_text() == report
    csv_rows = list(csv.DictReader(csv_path.open()))
    assert csv_rows[0]["phase"] == "hard"
    assert float(csv_rows[0]["denoise_00_multiplier"]) == 5.0
    with np.load(npz_path) as saved:
        np.testing.assert_array_equal(
            saved["guidance_multipliers"], comparison.guidance_multipliers
        )


def test_piperx_paired_joint_curves_use_matching_scene_and_real_trajectory(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluations.piperx.plot_paired_joint_curves import (
        generate_joint_comparisons,
    )

    root = tmp_path / "paired"
    trial = "pair01_seed10000"
    initial_state = np.zeros((1, 7), dtype=np.float32)
    initial_truth = {
        "positions": np.arange(9, dtype=np.float32).reshape(1, 3, 3),
        "quaternions": np.zeros((1, 3, 4), dtype=np.float32),
    }

    def write_trial(mode, steps, offset):
        directory = root / mode / "standard" / trial
        directory.mkdir(parents=True)
        measured = np.arange(steps, dtype=np.float32)[:, None] * 0.01
        measured = np.repeat(measured, 7, axis=1)
        targets = measured + offset
        final_state = np.full((1, 7), steps * 0.01, dtype=np.float32)
        np.save(directory / "initial_state.npy", initial_state)
        np.savez_compressed(directory / "initial_truth.npz", **initial_truth)
        np.savez_compressed(
            directory / "trajectory.npz",
            state=measured,
            actions_requested=targets,
            actions_applied=targets,
            wall_action_seconds=np.arange(steps, dtype=np.float64),
            chunk_start_steps=np.empty(0, dtype=np.int64),
            final_state=final_state,
        )
        (directory / "report.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "instruction": "put the red cube on the red cylinder",
                    "pair": [0, 1],
                    "scene_seed": 10000,
                    "task_result": {"success": mode == "rtc"},
                }
            )
        )

    write_trial("no_rtc", steps=3, offset=0.02)
    write_trial("rtc", steps=4, offset=0.01)

    summary_path = generate_joint_comparisons(root)

    summary = json.loads(summary_path.read_text())
    assert summary["joint_names"] == [f"joint{index}" for index in range(1, 7)]
    assert len(summary["comparisons"]) == 1
    row = summary["comparisons"][0]
    assert row["initial_scene_matches"] is True
    assert row["no_rtc"]["control_steps"] == 3
    assert row["rtc"]["control_steps"] == 4
    assert (summary_path.parent / row["plot"]).is_file()
    csv_rows = list(csv.DictReader((summary_path.parent / row["csv"]).open()))
    assert len(csv_rows) == 5
    assert float(csv_rows[0]["no_rtc_target_change_joint1"]) == pytest.approx(0.02)
    assert csv_rows[-1]["no_rtc_target_joint1"] == ""
    assert (summary_path.parent / "index.html").is_file()


@pytest.mark.parametrize("action_chunk", [10, 50])
def test_piperx_decoding_preserves_normalized_rtc_history(action_chunk):
    pytest.importorskip("openpi")
    from openpi.shared.normalize import NormStats
    from openpi.transforms import Unnormalize

    from rlinf.models.embodiment.openpi.dataconfig.piperx_dataconfig import (
        PiperXOutputs,
    )
    from rlinf.models.embodiment.openpi_rlinf.env_io import EnvIO

    io = EnvIO()
    io.action_chunk = action_chunk
    io.device = torch.device("cpu")
    low = np.array([-1, -2, -3, -4, -5, -6, 0], dtype=np.float32)
    high = np.array([1, 2, 3, 4, 5, 6, 1], dtype=np.float32)
    stats = NormStats(mean=np.zeros(7), std=np.ones(7), q01=low, q99=high)
    io.setup_transforms(
        [], [Unnormalize({"actions": stats}, use_quantiles=True), PiperXOutputs()]
    )
    model_actions = torch.linspace(-1, 1, 2 * 50 * 32).reshape(2, 50, 32)
    original = model_actions.clone()

    actions = io.decode_actions(model_actions, torch.zeros(2, 32))

    assert actions.shape == (2, action_chunk, 7)
    expected = (original[..., :7] + 1) / 2 * torch.from_numpy(high - low + 1e-6)
    expected += torch.from_numpy(low)
    torch.testing.assert_close(actions, expected[:, :action_chunk])
    torch.testing.assert_close(model_actions, original, rtol=0, atol=0)
    actions.fill_(0)
    torch.testing.assert_close(model_actions, original, rtol=0, atol=0)


@pytest.mark.parametrize("rtc_enabled", [True, False])
@pytest.mark.parametrize(
    "action_chunk,rtc_history_horizon,expected_history_horizon",
    [(10, None, 50), (50, None, 50), (30, 30, 30)],
)
def test_native_eval_controls_rtc_history_after_decoding(
    native_pi05_modules,
    rtc_enabled,
    action_chunk,
    rtc_history_horizon,
    expected_history_horizon,
):
    pytest.importorskip("openpi")
    from openpi.transforms import PadStatesAndActions

    from rlinf.models.embodiment.openpi.dataconfig.piperx_dataconfig import (
        PiperXInputs,
        PiperXOutputs,
    )
    from rlinf.models.embodiment.openpi_rlinf.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi_rlinf.sampling.rtc_guidance import (
        RTCGuidanceContext,
    )
    from rlinf.models.embodiment.openpi_rlinf.tasks.eval import Pi0Eval

    def tokenize(data):
        data = dict(data)
        data.pop("prompt")
        # A fixed prompt fits the small vocabulary used by this fixture.
        data["tokenized_prompt"] = np.array([1, 2, 3, 4], dtype=np.int64)
        data["tokenized_prompt_mask"] = np.ones(4, dtype=bool)
        return data

    model = Pi0Eval(
        Pi0Config(
            pi05=True,
            dtype="float32",
            action_dim=32,
            action_horizon=50,
            max_token_len=8,
        ),
        num_steps=2,
        action_chunk=action_chunk,
        config_name="pi05_piperx",
        rtc_enabled=rtc_enabled,
        rtc_history_horizon=rtc_history_horizon,
    ).eval()
    model.setup_transforms(
        [PiperXInputs(), PadStatesAndActions(32), tokenize], [PiperXOutputs()]
    )
    env_obs = {
        "main_images": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "states": torch.zeros(1, 7),
        "task_descriptions": ["put the red cube on the red cylinder"],
    }
    context = None
    for _ in range(2):
        actions, result = model.predict_action_batch(
            env_obs, noise=torch.zeros(1, 50, 32), rtc_context=context
        )
        full_prediction = result["forward_inputs"]["model_action"].reshape(1, 50, 32)
        assert actions.shape == (1, action_chunk, 7)
        assert result["model_actions"].shape == (
            1,
            expected_history_horizon,
            32,
        )
        torch.testing.assert_close(actions, full_prediction[:, :action_chunk, :7])
        torch.testing.assert_close(
            result["model_actions"], full_prediction[:, :expected_history_horizon]
        )
        context = RTCGuidanceContext(
            result["model_actions"], executed_horizon=10, delay_steps=5
        )
        remaining = context.get_prev_remaining()
        assert remaining.shape == (1, expected_history_horizon - 10, 32)
        torch.testing.assert_close(
            remaining,
            full_prediction[:, 10:expected_history_horizon],
            rtol=0,
            atol=0,
        )
