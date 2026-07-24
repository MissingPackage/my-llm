"""Construction helpers for the decoder trained from random initialization.

The project borrows Transformers' well-tested Llama implementation, not Llama
weights.  Keeping model construction in one small module makes the distinction
obvious and lets the memory-sensitive trainer choose dtype/device before the first
parameter is allocated.
"""

from __future__ import annotations

from typing import Any

from my_llm.config import ModelSpec


def build_model(
    spec: ModelSpec,
    tokenizer: Any,
    *,
    attention_backend: str = "sdpa",
) -> Any:
    """Build a bias-free, GQA Llama decoder with entirely new parameters.

    The caller is responsible for entering the desired ``torch.device`` and
    default-dtype contexts.  That avoids an 8 GiB temporary FP32 allocation when
    constructing the experimental 2B model with BF16 parameters.
    """

    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install training dependencies: uv sync --extra train") from exc

    # The architecture mirrors common modern decoder choices: RoPE positions,
    # RMSNorm, SwiGLU and grouped-query attention.  There are no downloaded weights.
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        num_key_value_heads=spec.num_key_value_heads,
        max_position_embeddings=spec.max_position_embeddings,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=spec.rms_norm_eps,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=spec.tie_word_embeddings,
        rope_parameters={"rope_theta": spec.rope_theta},
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        attn_implementation=attention_backend,
    )
    model = LlamaForCausalLM(config)
    # KV caching accelerates autoregressive inference but retains activations that
    # are unnecessary (and incompatible with checkpointing) during training.
    model.config.use_cache = False
    return model


def count_parameters(model: Any) -> int:
    """Return the exact number of scalar parameters in an instantiated model."""

    return sum(parameter.numel() for parameter in model.parameters())
