"""Schema and arithmetic tests that run without allocating training models."""

from pathlib import Path

import pytest

from my_llm.config import (
    DataPrepConfig,
    ModelSpec,
    PretrainConfig,
    TokenizerTrainConfig,
    approximate_parameter_count,
    load_typed,
    load_yaml,
)

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("name", ["debug", "100m", "350m", "2b-fit", "2b-chinchilla"])
def test_pretrain_configs_are_valid(name: str) -> None:
    config = load_typed(ROOT / f"configs/pretrain/{name}.yaml", PretrainConfig)
    assert config.training.sequence_length <= config.model.max_position_embeddings
    assert config.training.tokens_per_step > 0


@pytest.mark.parametrize("name", ["smoke", "32k-fineweb", "32k-bilingual"])
def test_tokenizer_configs_are_valid(name: str) -> None:
    config = load_typed(ROOT / f"configs/tokenizer/{name}.yaml", TokenizerTrainConfig)
    assert sum(source.weight for source in config.sources) > 0


@pytest.mark.parametrize(
    "name",
    ["smoke", "fineweb-edu-2b", "fineweb-bilingual-2b", "fineweb-edu-7b", "fineweb-edu-40b"],
)
def test_data_configs_are_valid(name: str) -> None:
    config = load_typed(ROOT / f"configs/data/{name}.yaml", DataPrepConfig)
    assert config.shard_tokens > 0


@pytest.mark.parametrize(
    "name",
    ["sft", "sft-smoke", "sft-identity-smoke", "dpo", "dpo-smoke", "dpo-constitutional-smoke", "reasoning-sft", "reasoning-bilingual-smoke", "grpo", "grpo-smoke", "grpo-custom-smoke"],
)
def test_posttrain_configs_have_local_output(name: str) -> None:
    config = load_yaml(ROOT / f"configs/posttrain/{name}.yaml")
    assert str(config["output_dir"]).startswith("artifacts/")
    assert str(config["model_path"]).startswith("artifacts/")


@pytest.mark.parametrize(
    "name",
    [
        "qwen3-1.7b-sft",
        "qwen3-1.7b-dpo",
        "qwen3-1.7b-reasoning-sft",
        "qwen3-1.7b-grpo",
    ],
)
def test_qwen_posttrain_configs_enable_nf4_qlora(name: str) -> None:
    config = load_yaml(ROOT / f"configs/posttrain/{name}.yaml")
    assert config["peft"]["enabled"] is True
    assert config["peft"]["quantization"] == "nf4"
    assert config["training"]["per_device_train_batch_size"] == 1


def test_parameter_counts_match_profiles() -> None:
    small = load_typed(ROOT / "configs/pretrain/100m.yaml", PretrainConfig)
    stretch = load_typed(ROOT / "configs/pretrain/350m.yaml", PretrainConfig)
    experimental = load_typed(ROOT / "configs/pretrain/2b-fit.yaml", PretrainConfig)
    assert approximate_parameter_count(small.model, 32_000) == 100_092_672
    assert approximate_parameter_count(stretch.model, 32_000) == 348_447_744
    assert approximate_parameter_count(experimental.model, 32_000) == 2_027_174_400
    assert experimental.training.parameter_dtype == "bf16"
    assert experimental.training.optimizer == "paged_adamw_8bit"


def test_invalid_attention_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelSpec(
            hidden_size=100,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=6,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
