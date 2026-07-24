"""Pure QLoRA configuration tests; no CUDA packages are imported."""

import pytest

from my_llm.adapters import adapter_settings, loftq_model_path, lora_kwargs
from my_llm.posttrain import select_openr1_trace


def test_laptop_qlora_defaults_are_rank_stabilized() -> None:
    settings = adapter_settings({"peft": {"enabled": True}})
    kwargs = lora_kwargs(settings)
    assert settings["quantization"] == "nf4"
    assert settings["double_quant"] is True
    assert kwargs["target_modules"] == "all-linear"
    assert kwargs["use_rslora"] is True


def test_loftq_requires_nf4_and_a_new_adapter() -> None:
    with pytest.raises(ValueError, match="requires NF4"):
        adapter_settings({"peft": {"enabled": True, "quantization": "none", "loftq": True}})
    with pytest.raises(ValueError, match="new adapter"):
        adapter_settings({"adapter_path": "old-adapter", "peft": {"enabled": True, "loftq": True}})


def test_loftq_passes_local_directories_explicitly(tmp_path) -> None:
    # peft resolves model_path=None via snapshot_download, which treats a local
    # directory as a Hub repo id; local bases must therefore be passed through.
    assert loftq_model_path(str(tmp_path)) == str(tmp_path)
    assert loftq_model_path("Qwen/Qwen3-1.7B-Base") is None


def test_unknown_adapter_knob_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown PEFT"):
        adapter_settings({"peft": {"enabled": True, "magic": True}})


def test_openr1_prefers_a_complete_verified_generation() -> None:
    row = {
        "solution": "fallback",
        "generations": [
            "<think>wrong</think>answer",
            "<think>verified reasoning</think>answer",
        ],
        "correctness_math_verify": [False, True],
        "correctness_llama": [False, False],
        "is_reasoning_complete": [True, True],
    }
    assert select_openr1_trace(row) == "verified reasoning"
