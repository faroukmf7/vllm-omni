# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only smoke tests for native FunAudioChat model integration."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models import ModelRegistry

from vllm_omni.engine.arg_utils import register_omni_models_to_vllm
from vllm_omni.model_executor.models import FunAudioChatForConditionalGeneration
from vllm_omni.transformers_utils.configs.funaudiochat import FunAudioChatConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_funaudiochat_model_and_config_imports() -> None:
    assert FunAudioChatForConditionalGeneration.__name__ == "FunAudioChatForConditionalGeneration"
    assert FunAudioChatConfig.__name__ == "FunAudioChatConfig"


def test_funaudiochat_model_architecture_is_registered() -> None:
    register_omni_models_to_vllm()
    model_config = SimpleNamespace(model_impl="vllm", runner_type=None, convert_type="none")
    model_cls, _ = ModelRegistry.resolve_model_cls("FunAudioChatForConditionalGeneration", model_config)

    assert model_cls is FunAudioChatForConditionalGeneration


def test_funaudiochat_model_initializes_without_checkpoint_weights(mocker) -> None:
    config = FunAudioChatConfig(
        text_config={
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )
    native_model = mocker.Mock()
    native_model.config = config
    native_model.multimodal_config = None
    native_model.make_empty_intermediate_tensors = mocker.Mock()
    hidden_states = mocker.sentinel.hidden_states
    native_model.return_value = hidden_states
    weights = [("model.weight", mocker.sentinel.weight)]
    native_model.load_weights.return_value = {"model.weight"}
    native_model_init = mocker.patch(
        "vllm_omni.model_executor.models.funaudiochat.funaudiochat.VllmFunAudioChatForConditionalGeneration",
        return_value=native_model,
    )
    vllm_config = mocker.sentinel.vllm_config

    model = FunAudioChatForConditionalGeneration(vllm_config=vllm_config)

    native_model_init.assert_called_once_with(vllm_config=vllm_config, prefix="")
    assert isinstance(model, FunAudioChatForConditionalGeneration)
    assert model.config is config
    assert model.get_language_model() is native_model.language_model

    output = model.forward(mocker.sentinel.input_ids, mocker.sentinel.positions)
    assert output.text_hidden_states is hidden_states
    assert output.multimodal_outputs == {}
    native_model.assert_called_once()
    assert model.load_weights(weights) == {"model.weight"}
    native_model.load_weights.assert_called_once_with(weights)
