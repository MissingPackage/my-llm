"""Typed configuration models shared by the command-line tools.

The project deliberately validates YAML at the boundary.  A misspelled option in a
multi-day training run should fail immediately instead of being silently ignored.
The comments below also document *why* each non-obvious knob exists; the YAML files
then serve as small, reproducible experiment records rather than loose argument bags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base class that rejects unknown YAML keys."""

    model_config = ConfigDict(extra="forbid")


class SourceSpec(StrictModel):
    """One local or Hugging Face dataset mixed into a streaming corpus."""

    path: str
    name: str | None = None
    revision: str | None = None
    split: str = "train"
    text_column: str = "text"
    data_files: str | list[str] | dict[str, str | list[str]] | None = None
    weight: float = Field(default=1.0, gt=0)


class TokenizerTrainConfig(StrictModel):
    """Configuration for byte-level BPE tokenizer training."""

    output_dir: Path
    vocab_size: int = Field(ge=256, le=65535)
    min_frequency: int = Field(default=2, ge=1)
    max_documents: int | None = Field(default=None, ge=1)
    model_max_length: int = Field(default=2048, ge=32)
    seed: int = 42
    shuffle_buffer: int = Field(default=10_000, ge=1)
    sources: list[SourceSpec] = Field(min_length=1)


class DataPrepConfig(StrictModel):
    """Configuration for converting text streams to memory-mapped token shards."""

    tokenizer_path: Path
    output_dir: Path
    dtype: Literal["uint16", "uint32"] = "uint16"
    shard_tokens: int = Field(default=100_000_000, ge=1)
    target_train_tokens: int | None = Field(default=None, ge=1)
    target_validation_tokens: int | None = Field(default=None, ge=1)
    validation_fraction: float = Field(default=0.005, gt=0, lt=1)
    max_documents: int | None = Field(default=None, ge=1)
    seed: int = 42
    shuffle_buffer: int = Field(default=10_000, ge=1)
    # Fast tokenizers amortize Rust/Python boundary cost across a small text batch.
    tokenize_batch_size: int = Field(default=64, ge=1, le=4096)
    sources: list[SourceSpec] = Field(min_length=1)


class ModelSpec(StrictModel):
    """Shape of the randomly initialized, Llama-style decoder."""

    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_hidden_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: int = Field(gt=0)
    max_position_embeddings: int = Field(ge=32)
    rope_theta: float = Field(default=10_000.0, gt=0)
    rms_norm_eps: float = Field(default=1e-5, gt=0)
    tie_word_embeddings: bool = True

    @model_validator(mode="after")
    def validate_attention_shape(self) -> ModelSpec:
        """Reject GQA layouts that cannot be evenly partitioned into heads."""

        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        return self


class TrainingSpec(StrictModel):
    """Single-GPU pretraining options, including explicit memory trade-offs."""

    sequence_length: int = Field(ge=8)
    micro_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    min_lr_ratio: float = Field(default=0.1, ge=0, le=1)
    warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.1, ge=0)
    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.95, gt=0, lt=1)
    grad_clip: float = Field(default=1.0, gt=0)
    precision: Literal["bf16", "fp32"] = "bf16"
    # ``precision`` controls autocast.  ``parameter_dtype`` controls the stored
    # weights themselves.  BF16 weights save ~2 bytes/parameter but do not retain
    # an FP32 master copy, so they are a fit-enabling compromise for the 2B preset.
    parameter_dtype: Literal["bf16", "fp32"] = "fp32"
    gradient_checkpointing: bool = True
    # SDPA is the robust default: on CUDA, PyTorch selects an eligible flash or
    # memory-efficient kernel internally.  FlashAttention 2 remains an opt-in
    # backend because it needs an additional kernel package.
    attention_backend: Literal["sdpa", "flash_attention_2", "eager"] = "sdpa"
    optimizer: Literal["adamw_fused", "adamw_8bit", "paged_adamw_8bit"] = "adamw_fused"
    optimizer_min_8bit_size: int = Field(default=4096, ge=0)
    optimizer_percentile_clipping: int = Field(default=100, ge=1, le=100)
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    # One prefetched batch overlaps host-to-device copies with the current forward
    # pass.  Higher values spend more VRAM and normally do not help this workload.
    prefetch_batches: int = Field(default=0, ge=0, le=4)
    # Optimizer state is essential for an exact resume but can add several GiB per
    # checkpoint at 2B scale.  Fit checks may deliberately save weights only.
    save_checkpoints: bool = True
    save_optimizer_state: bool = True
    max_shard_size: str = "2GB"
    empty_cache_every: int = Field(default=0, ge=0)
    log_every: int = Field(default=10, ge=1)
    eval_every: int = Field(default=250, ge=1)
    eval_batches: int = Field(default=20, ge=1)
    save_every: int = Field(default=500, ge=1)
    keep_last_checkpoints: int = Field(default=3, ge=1)

    @property
    def tokens_per_step(self) -> int:
        """Return the effective number of tokens in one optimizer update."""

        return self.sequence_length * self.micro_batch_size * self.gradient_accumulation_steps


class PretrainConfig(StrictModel):
    """Complete random-initialization pretraining experiment."""

    name: str
    seed: int = 42
    tokenizer_path: Path
    train_manifest: Path
    validation_manifest: Path
    output_dir: Path
    log_dir: Path
    model: ModelSpec
    training: TrainingSpec

    @model_validator(mode="after")
    def validate_context(self) -> PretrainConfig:
        """Check relationships that span the model and optimizer sections."""

        if self.training.sequence_length > self.model.max_position_embeddings:
            raise ValueError("sequence_length exceeds max_position_embeddings")
        if self.training.warmup_steps >= self.training.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        return self


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject scalar/list roots with a useful error."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_typed(path: str | Path, config_type: type[BaseModel]) -> Any:
    """Load YAML and validate it with the requested Pydantic model."""

    return config_type.model_validate(load_yaml(path))


def approximate_parameter_count(spec: ModelSpec, vocab_size: int) -> int:
    """Count bias-free Llama-style parameters without allocating model weights.

    The formula includes GQA's narrower K/V projections, SwiGLU's three matrices,
    two RMSNorm vectors per block, embeddings, and the final RMSNorm.  It is useful
    for checking a 2B YAML file on a machine that could not instantiate that model.
    """
    head_dim = spec.hidden_size // spec.num_attention_heads
    kv_width = spec.num_key_value_heads * head_dim
    attention = (
        spec.hidden_size * spec.hidden_size
        + 2 * spec.hidden_size * kv_width
        + spec.hidden_size * spec.hidden_size
    )
    mlp = 3 * spec.hidden_size * spec.intermediate_size
    norms = 2 * spec.hidden_size
    embeddings = vocab_size * spec.hidden_size
    lm_head = 0 if spec.tie_word_embeddings else embeddings
    return (
        embeddings + lm_head + spec.num_hidden_layers * (attention + mlp + norms) + spec.hidden_size
    )
