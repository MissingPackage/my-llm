"""Online reasoning alignment with TRL's Group Relative Policy Optimization.

On a 16 GiB GPU, generation—not LoRA optimizer state—is the dominant GRPO cost.
The laptop preset therefore uses four completions, NF4 QLoRA, a zero KL coefficient
(no reference model), truncated-completion masking and the DAPO loss normalization.
Optional continuous batching is wired in, but is not enabled for a four-sample group
because its documented gains appear at much larger generation batches.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from my_llm.adapters import adapter_settings, load_training_model
from my_llm.config import load_yaml
from my_llm.reasoning import gsm8k_prompt
from my_llm.rewards import resolve_reward_config


def load_reasoning_dataset(config: dict[str, Any]) -> Any:
    """Load GSM8K and expose prompt/reference columns expected by reward functions."""

    from datasets import load_dataset

    dataset_config = config["dataset"]
    kwargs: dict[str, Any] = {
        "path": dataset_config["path"],
        "split": dataset_config["split"],
    }
    for key in ("name", "data_files", "revision"):
        if dataset_config.get(key) is not None:
            kwargs[key] = dataset_config[key]
    dataset = load_dataset(**kwargs)
    limit = dataset_config.get("max_samples")
    if limit is not None and limit < len(dataset):
        dataset = dataset.shuffle(seed=int(config.get("seed", 42))).select(range(limit))
    columns = dataset.column_names
    return dataset.map(
        lambda row: {"prompt": gsm8k_prompt(row["question"]), "answer": row["answer"]},
        remove_columns=columns,
        desc="Formatting GSM8K for GRPO",
    )


def _continuous_batching_kwargs(training: dict[str, Any]) -> dict[str, Any]:
    """Return the optional in-process Transformers generation-engine settings."""

    if not training.get("use_transformers_continuous_batching", False):
        return {"use_transformers_continuous_batching": False}
    return {
        "use_transformers_continuous_batching": True,
        "transformers_continuous_batching_config": {
            # CUDA graphs reserve memory and are less useful with a changing online
            # policy, so the conservative single-GPU profile keeps them disabled.
            "use_cuda_graph": bool(training.get("continuous_batching_cuda_graph", False)),
            "max_memory_percent": float(
                training.get("continuous_batching_max_memory_percent", 0.35)
            ),
        },
    }


def run(config: dict[str, Any], *, resume: str | None = None) -> Path:
    """Run GRPO with YAML-selected rewards (default: exact-answer plus weak format)."""

    try:
        import torch
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install training dependencies: uv sync --extra train") from exc

    training = config["training"]
    reward_names, reward_funcs, reward_weights = resolve_reward_config(training)
    print(f"GRPO rewards: {list(zip(reward_names, reward_weights, strict=True))}")
    effective_batch = (
        training["per_device_train_batch_size"] * training["gradient_accumulation_steps"]
    )
    if effective_batch % training["num_generations"]:
        raise ValueError("Effective batch size must be divisible by num_generations")
    if training["generation_batch_size"] % training["num_generations"]:
        raise ValueError("generation_batch_size must be divisible by num_generations")

    cuda = torch.cuda.is_available()
    if adapter_settings(config)["enabled"] and not cuda:
        raise RuntimeError("The QLoRA GRPO preset requires CUDA")

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Batched autoregressive generation must left-pad so each sequence's newest
    # token occupies the same column; SFT/DPO correctly use right padding instead.
    tokenizer.padding_side = "left"
    model = load_training_model(config, torch)
    model.config.use_cache = False
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    dataset = load_reasoning_dataset(config)

    workers = int(training.get("dataloader_num_workers", 0))
    optional: dict[str, Any] = _continuous_batching_kwargs(training)
    if training.get("top_entropy_quantile") is not None:
        optional["top_entropy_quantile"] = float(training["top_entropy_quantile"])
    if workers > 0:
        optional["dataloader_prefetch_factor"] = int(training.get("dataloader_prefetch_factor", 2))

    arguments = GRPOConfig(
        output_dir=config["output_dir"],
        seed=int(config.get("seed", 42)),
        data_seed=int(config.get("seed", 42)),
        bf16=cuda and torch.cuda.is_bf16_supported(),
        tf32=cuda,
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=training.get(
            "optim",
            "paged_adamw_8bit"
            if adapter_settings(config)["enabled"]
            else ("adamw_torch_fused" if cuda else "adamw_torch"),
        ),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        report_to=training.get("report_to", ["tensorboard"]),
        per_device_train_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        max_steps=training["max_steps"],
        # GRPO has a fixed max_steps budget, so convert the YAML ratio to the
        # non-deprecated explicit warmup count.
        warmup_steps=max(0, int(training["max_steps"] * training["warmup_ratio"])),
        weight_decay=training["weight_decay"],
        adam_beta1=training.get("adam_beta1", 0.9),
        adam_beta2=training.get("adam_beta2", 0.95),
        max_grad_norm=training.get("max_grad_norm", 1.0),
        num_generations=training["num_generations"],
        generation_batch_size=training["generation_batch_size"],
        max_completion_length=training["max_completion_length"],
        temperature=training["temperature"],
        # beta=0 is intentional on this GPU: TRL then does not load a reference
        # model.  The policy is still constrained by clipping and small updates.
        beta=training["beta"],
        loss_type=training["loss_type"],
        reward_weights=reward_weights,
        use_vllm=False,
        mask_truncated_completions=bool(training.get("mask_truncated_completions", True)),
        use_liger_kernel=bool(training.get("use_liger_kernel", False)),
        torch_compile=bool(training.get("torch_compile", False)),
        torch_compile_mode=training.get("torch_compile_mode"),
        torch_empty_cache_steps=training.get("torch_empty_cache_steps"),
        logging_steps=training["logging_steps"],
        logging_first_step=True,
        log_completions=True,
        num_completions_to_print=2,
        save_strategy="steps",
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        dataloader_num_workers=workers,
        dataloader_pin_memory=cuda,
        dataloader_persistent_workers=workers > 0,
        include_num_input_tokens_seen=True,
        use_cache=False,
        use_cpu=not cuda,
        **optional,
    )
    trainer = GRPOTrainer(
        model=model,
        args=arguments,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume)

    final_dir = Path(config["output_dir"]) / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    (final_dir / "grpo-config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    return final_dir


def main() -> None:
    """CLI entry point for online reasoning post-training."""

    parser = argparse.ArgumentParser(description="Reasoning post-training with GRPO and GSM8K.")
    parser.add_argument("config")
    parser.add_argument("--resume")
    args = parser.parse_args()
    output = run(load_yaml(args.config), resume=args.resume)
    print(f"Final model or adapter saved to {output}")


if __name__ == "__main__":
    main()
