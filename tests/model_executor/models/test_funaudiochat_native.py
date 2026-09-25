now # SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only registration and initialization checks for native FunAudioChat."""

import pytest
import torch
from vllm.model_executor.models import ModelRegistry

from vllm_omni.engine.arg_utils import register_omni_models_to_vllm
from vllm_omni.model_executor.models.funaudiochat.funaudiochat import FunAudioChatForCausalLM
from vllm_omni.transformers_utils.configs.funaudiochat import FunAudioChatConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_funaudiochat_model_and_config_imports() -> None:
    assert FunAudioChatForCausalLM.__name__ == "FunAudioChatForCausalLM"
    assert FunAudioChatConfig.__name__ == "FunAudioChatConfig"


def test_funaudiochat_model_architecture_is_registered() -> None:
    register_omni_models_to_vllm()
    model_cls, _ = ModelRegistry.resolve_model_cls("FunAudioChatForCausalLM")

    assert model_cls is FunAudioChatForCausalLM


def test_funaudiochat_model_initializes_without_checkpoint_weights() -> None:
    config = FunAudioChatConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
    )

    with torch.device("meta"):
        model = FunAudioChatForCausalLM(config)

    assert isinstance(model, FunAudioChatForCausalLM)