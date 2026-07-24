"""Arithmetic feasibility estimates that do not allocate model parameters."""

from __future__ import annotations

import argparse
import json

from my_llm.config import PretrainConfig, approximate_parameter_count, load_typed


def _persistent_bytes_per_parameter(config: PretrainConfig) -> tuple[int, int, int]:
    """Return conservative bytes for weights, gradients and optimizer moments."""

    parameter_bytes = 2 if config.training.parameter_dtype == "bf16" else 4
    gradient_bytes = parameter_bytes
    # bitsandbytes keeps two quantized moment buffers plus block statistics.  Two
    # bytes/parameter is a useful lower-order estimate; fused AdamW is conservatively
    # counted as two FP32 moments.  Activations and allocator workspace are separate.
    optimizer_bytes = 2 if "8bit" in config.training.optimizer else 8
    return parameter_bytes, gradient_bytes, optimizer_bytes


def estimate(
    config: PretrainConfig, *, vocab_size: int, target_tokens: int, sustained_tflops: float
) -> dict[str, float | int | str]:
    """Estimate 6ND compute, token storage, update count and persistent VRAM."""

    parameters = approximate_parameter_count(config.model, vocab_size)
    training_flops = 6 * parameters * target_tokens
    lower_bound_days = training_flops / (sustained_tflops * 1e12) / 86_400
    steps = (target_tokens + config.training.tokens_per_step - 1) // config.training.tokens_per_step
    parameter_bytes, gradient_bytes, optimizer_bytes = _persistent_bytes_per_parameter(config)
    persistent_bytes = parameters * (parameter_bytes + gradient_bytes + optimizer_bytes)
    return {
        "parameters": parameters,
        "parameters_billions": round(parameters / 1e9, 3),
        "target_tokens": target_tokens,
        "tokens_per_optimizer_step": config.training.tokens_per_step,
        "optimizer_steps_for_target": steps,
        "training_flops_6ND": f"{training_flops:.4e}",
        "ideal_compute_days": round(lower_bound_days, 2),
        "assumed_sustained_tflops": sustained_tflops,
        "token_storage_gib_uint16": round(target_tokens * 2 / 1024**3, 2),
        "weight_gib": round(parameters * parameter_bytes / 1024**3, 2),
        "gradient_gib": round(parameters * gradient_bytes / 1024**3, 2),
        "optimizer_state_gib_estimate": round(parameters * optimizer_bytes / 1024**3, 2),
        "persistent_training_state_gib_estimate": round(persistent_bytes / 1024**3, 2),
        "not_included": "activations, CUDA workspaces, allocator fragmentation, paging overhead",
    }


def main() -> None:
    """Print a JSON estimate for a pretraining YAML file."""

    parser = argparse.ArgumentParser(description="Estimate model size and 6ND pretraining cost.")
    parser.add_argument("config")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--sustained-tflops", type=float, default=25.0)
    args = parser.parse_args()
    config = load_typed(args.config, PretrainConfig)
    print(
        json.dumps(
            estimate(
                config,
                vocab_size=args.vocab_size,
                target_tokens=args.target_tokens,
                sustained_tflops=args.sustained_tflops,
            ),
            indent=2,
        )
    )
    print(
        "Time is a 6ND arithmetic lower bound, not a promise; measure real tokens/s, "
        "thermals and paged-optimizer traffic on this laptop."
    )


if __name__ == "__main__":
    main()
