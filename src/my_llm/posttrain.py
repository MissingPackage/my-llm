"""Supervised and preference post-training with optional single-GPU QLoRA.

The same command supports the tiny from-scratch checkpoints and the practical
Qwen3 ~2B route.  The latter loads NF4 weights through :mod:`my_llm.adapters`, so
only LoRA matrices receive gradients and optimizer state.  YAML remains the source
of truth for every memory/quality trade-off used by TRL.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from my_llm.adapters import adapter_settings, load_training_model
from my_llm.config import load_yaml
from my_llm.reasoning import gsm8k_messages, math_reasoning_messages

THINK_TRACE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)


def load_split(dataset_config: dict[str, Any], split: str) -> Any:
    """Load one finite dataset split with optional subset/files/revision selectors."""

    from datasets import load_dataset

    kwargs: dict[str, Any] = {"path": dataset_config["path"], "split": split}
    for key in ("name", "data_files", "revision"):
        if dataset_config.get(key) is not None:
            kwargs[key] = dataset_config[key]
    dataset = load_dataset(**kwargs)
    columns = dataset_config.get("columns")
    if columns:
        # TRL routes examples by column shape — a stray `prompt` column next to
        # `messages` silently switches it to prompt/completion mode — so configs
        # can pin exactly which columns training may see.
        dataset = dataset.select_columns(list(columns))
    return dataset


def limit_dataset(dataset: Any, limit: int | None, seed: int) -> Any:
    """Take a deterministic shuffled subset instead of a biased leading slice."""

    if limit is not None and limit < len(dataset):
        return dataset.shuffle(seed=seed).select(range(limit))
    return dataset


def load_train_dataset(dataset_config: dict[str, Any], seed: int) -> Any:
    """Route the train split through weighted mixing when ``sources`` is present.

    The single-source branch is deliberately the exact call ``run`` has always
    made, so existing configs keep loading byte-identical datasets.
    """

    if dataset_config.get("sources"):
        # Imported lazily so single-source runs never touch the mixing module.
        from my_llm.mixing import load_mixed_dataset

        return load_mixed_dataset(dataset_config, seed)
    return load_split(dataset_config, dataset_config["split"])


def load_eval_dataset(dataset_config: dict[str, Any]) -> Any:
    """Resolve the held-out eval set, or ``None`` when the config declares none.

    ``eval_source`` (a full source spec) wins over ``eval_split`` because mixed
    runs have no top-level path for ``eval_split`` to resolve against; plain
    single-source configs keep the historical ``eval_split`` behavior.
    """

    if dataset_config.get("eval_source"):
        from my_llm.mixing import validate_eval_source

        spec = validate_eval_source(dataset_config["eval_source"])
        eval_config = spec.model_dump()
        eval_config["columns"] = dataset_config.get("columns")
        return load_split(eval_config, spec.split)
    if dataset_config.get("eval_split"):
        if dataset_config.get("sources"):
            raise ValueError(
                "dataset.eval_split cannot resolve a dataset next to dataset.sources; "
                "declare the held-out set via dataset.eval_source instead"
            )
        return load_split(dataset_config, dataset_config["eval_split"])
    return None


def select_openr1_trace(row: dict[str, Any]) -> str:
    """Select a verified generated trace, falling back to the reference solution.

    OpenR1 can hold several DeepSeek-R1 generations per problem. Prefer the first
    trace marked correct by Math Verify, then by the Llama judge, and require the
    corresponding generation to be complete when that flag is available.
    """

    generations = row.get("generations") or []
    math_flags = row.get("correctness_math_verify") or []
    judge_flags = row.get("correctness_llama") or []
    complete_flags = row.get("is_reasoning_complete") or []
    for index, generation in enumerate(generations):
        math_correct = index < len(math_flags) and math_flags[index] is True
        judge_correct = index < len(judge_flags) and judge_flags[index] is True
        complete = index >= len(complete_flags) or complete_flags[index] is not False
        if (
            isinstance(generation, str)
            and generation.strip()
            and complete
            and (math_correct or judge_correct)
        ):
            match = THINK_TRACE.search(generation)
            return match.group(1).strip() if match else generation.strip()
    return str(row.get("solution") or "").strip()


def prepare_reasoning_sft(dataset: Any, dataset_config: dict[str, Any]) -> Any:
    """Normalize supported math corpora to conversational SFT examples.

    ``gsm8k`` stores a solution and final answer in one ``answer`` field.  OpenR1
    provides separate problem/solution/answer fields; a conservative character
    filter avoids spending 4K-token batches on traces whose final answer would be
    truncated away.  Character length is only a cheap proxy, so evaluation remains
    essential.

    ``dataset.reasoning_lang`` selects the system-prompt language for ``gsm8k`` and
    ``openr1_math``.  Project language rule (revised by ruling E2, 2026-07-17): the
    answer follows the language of the question; the reasoning trace is
    unconstrained — selecting the Italian variant matches the instruction language
    to Italian questions, it does not mandate Italian thinking:

    - ``"en"`` (default): the historical behaviour — output identical to before the
      option existed;
    - ``"it"``: every example gets ``REASONING_SYSTEM_PROMPT_IT``;
    - ``"auto"``: a deterministic per-question heuristic — the question is
      lowercased, split into alphabetic words, and matched against two disjoint
      Italian/English stopword sets; Italian is chosen only when it has strictly
      more hits, every other case (tie, zero hits, symbol-only text) falls back to
      ``"en"``.  Pure string counting, no RNG: the same question always maps to the
      same language.  Honest limits: signal-free questions ("Calcola 15 * 7.")
      fall back to English by design, and other Romance languages sharing articles
      with Italian may be misread — acceptable for the IT/EN corpora this project
      mixes.

    The ``messages`` format embeds full conversations (system prompt included), so
    combining it with an explicit ``reasoning_lang`` is rejected rather than
    silently ignored.
    """

    # Imported here to keep this phase's changes confined to this formatter.
    from my_llm.reasoning import reasoning_system_prompt

    lang_setting = dataset_config.get("reasoning_lang", "en")
    if lang_setting not in {"en", "it", "auto"}:
        raise ValueError(
            f"Unsupported dataset.reasoning_lang {lang_setting!r}; valid: 'en', 'it', 'auto'"
        )

    # Disjoint by construction: tokens common to both languages (or plausible as
    # math symbols, e.g. "e", "i", "in", "a", "al") are deliberately in neither set.
    it_stopwords = frozenset(
        {"il", "lo", "la", "gli", "le", "uno", "una", "di", "del", "della", "dei", "delle"}
        | {"alla", "alle", "che", "è", "ed", "sono", "ha", "hanno", "non", "più", "ogni"}
        | {"se", "per", "con", "tra", "fra", "quanto", "quanti", "quanta", "quante", "perché"}
    )
    en_stopwords = frozenset(
        {"the", "an", "of", "is", "are", "and", "for", "with", "that", "this", "to", "from"}
        | {"at", "by", "on", "his", "her", "their", "it", "they", "how", "many", "much"}
        | {"what", "which", "when", "where", "why", "each", "every", "more", "than"}
        | {"does", "do", "has", "have", "if"}
    )
    word_pattern = re.compile(r"[a-zà-öø-ÿ]+")

    def question_lang(question: str) -> str:
        if lang_setting != "auto":
            return lang_setting
        words = word_pattern.findall(question.lower())
        it_hits = sum(1 for token in words if token in it_stopwords)
        en_hits = sum(1 for token in words if token in en_stopwords)
        return "it" if it_hits > en_hits else "en"

    format_name = dataset_config.get("format", "gsm8k")
    if format_name == "gsm8k":
        columns = dataset.column_names

        def format_gsm8k(row: dict[str, Any]) -> dict[str, Any]:
            messages = gsm8k_messages(row["question"], row["answer"])
            messages[0]["content"] = reasoning_system_prompt(question_lang(row["question"]))
            return {"messages": messages}

        return dataset.map(
            format_gsm8k,
            remove_columns=columns,
            desc="Formatting GSM8K for reasoning SFT",
        )

    if format_name == "openr1_math":
        maximum_characters = int(dataset_config.get("max_characters", 12_000))
        dataset = dataset.filter(
            lambda row: (
                isinstance(row.get("problem"), str)
                and isinstance(row.get("answer"), str)
                and bool(row["problem"].strip())
                and bool(row["answer"].strip())
                and bool(select_openr1_trace(row))
                and len(row["problem"]) + len(select_openr1_trace(row)) <= maximum_characters
            ),
            desc="Dropping OpenR1 traces that would be heavily truncated",
        )
        columns = dataset.column_names

        def format_openr1(row: dict[str, Any]) -> dict[str, Any]:
            messages = math_reasoning_messages(
                row["problem"], select_openr1_trace(row), row["answer"]
            )
            messages[0]["content"] = reasoning_system_prompt(question_lang(row["problem"]))
            return {"messages": messages}

        return dataset.map(
            format_openr1,
            remove_columns=columns,
            desc="Formatting OpenR1 math traces for reasoning SFT",
        )

    if format_name == "messages":
        if "reasoning_lang" in dataset_config:
            raise ValueError(
                "dataset.reasoning_lang is not supported with format 'messages': rows "
                "already embed their full conversation, including the system prompt"
            )
        # Retain only the column TRL needs; dropping large provenance columns lowers
        # Arrow cache pressure during tokenization.
        removable = [name for name in dataset.column_names if name != "messages"]
        return dataset.remove_columns(removable)
    raise ValueError(f"Unsupported reasoning dataset format: {format_name}")


def common_training_args(
    config: dict[str, Any],
    training: dict[str, Any],
    *,
    has_eval: bool,
    train_examples: int,
) -> dict[str, Any]:
    """Build the TrainingArguments shared by SFT and DPO.

    Every option is deliberately overridable in YAML.  Defaults favour a 16 GiB
    Ada laptop: BF16/TF32, non-reentrant checkpointing, pinned input memory, cosine
    decay and paged 8-bit AdamW when PEFT is enabled.
    """

    import torch

    cuda = torch.cuda.is_available()
    peft_enabled = bool(adapter_settings(config)["enabled"])
    workers = int(training.get("dataloader_num_workers", 0))
    # Transformers 5 deprecates a ratio stored in TrainingArguments.  Convert it to
    # an explicit update count now, including gradient accumulation.
    updates_per_epoch = math.ceil(
        train_examples
        / (training["per_device_train_batch_size"] * training["gradient_accumulation_steps"])
    )
    total_updates = max(1, math.ceil(updates_per_epoch * training["num_train_epochs"]))
    warmup_steps = math.ceil(total_updates * training["warmup_ratio"])
    shared: dict[str, Any] = {
        "output_dir": config["output_dir"],
        "seed": int(config.get("seed", 42)),
        "data_seed": int(config.get("seed", 42)),
        "bf16": cuda and torch.cuda.is_bf16_supported(),
        "tf32": cuda,
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": training.get(
            "optim",
            "paged_adamw_8bit"
            if peft_enabled
            else ("adamw_torch_fused" if cuda else "adamw_torch"),
        ),
        "lr_scheduler_type": training.get("lr_scheduler_type", "cosine"),
        "report_to": training.get("report_to", ["tensorboard"]),
        "logging_steps": training["logging_steps"],
        "logging_first_step": True,
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": training.get("eval_steps"),
        "eval_accumulation_steps": 1 if has_eval else None,
        "save_strategy": "steps",
        "save_steps": training["save_steps"],
        "save_total_limit": training["save_total_limit"],
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "per_device_eval_batch_size": training.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "learning_rate": training["learning_rate"],
        "num_train_epochs": training["num_train_epochs"],
        "warmup_steps": warmup_steps,
        "weight_decay": training["weight_decay"],
        "adam_beta1": training.get("adam_beta1", 0.9),
        "adam_beta2": training.get("adam_beta2", 0.95),
        "max_grad_norm": training.get("max_grad_norm", 1.0),
        "dataloader_num_workers": workers,
        "dataloader_pin_memory": cuda,
        "dataloader_persistent_workers": workers > 0,
        "include_num_input_tokens_seen": True,
        "use_liger_kernel": bool(training.get("use_liger_kernel", False)),
        "torch_compile": bool(training.get("torch_compile", False)),
        # TrainingArguments force-enables compile whenever mode/backend is
        # non-null, silently inverting `torch_compile: false` — forward the
        # mode only when compile is actually requested.
        "torch_compile_mode": (
            training.get("torch_compile_mode") if training.get("torch_compile") else None
        ),
        "torch_empty_cache_steps": training.get("torch_empty_cache_steps"),
        "activation_offloading": bool(training.get("activation_offloading", False)),
        "use_cache": False,
        "use_cpu": not cuda,
    }
    if workers > 0:
        shared["dataloader_prefetch_factor"] = int(training.get("dataloader_prefetch_factor", 2))
    return shared


def _validate_kernel_combination(training: dict[str, Any], method: str) -> None:
    """Reject combinations known to be unsupported instead of silently degrading."""

    backend = str(training.get("attention_backend", "sdpa"))
    packing = bool(training.get("packing", False))
    strategy = training.get("packing_strategy", "bfd")
    if packing and strategy == "bfd" and "flash" not in backend and "flash-attn" not in backend:
        raise ValueError(
            "BFD packing forces padding-free batches and therefore needs a FlashAttention backend"
        )
    if (
        method == "dpo"
        and training.get("use_liger_kernel")
        and training.get("precompute_ref_log_probs", True)
    ):
        raise ValueError("DPO cannot combine Liger with precompute_ref_log_probs")
    if training.get("use_liger_kernel") and training.get("loss_type") == "chunked_nll":
        raise ValueError("Liger fused CE and chunked_nll are alternative loss kernels")


def _package_versions() -> dict[str, str]:
    """Record the fast-moving training stack next to every final adapter."""

    versions = {}
    for package in ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    return versions


def run(config: dict[str, Any], *, resume: str | None = None) -> Path:
    """Run instruction/reasoning SFT or DPO and save the final model/adapter."""

    try:
        import torch
        from transformers import AutoTokenizer
        from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install training dependencies: uv sync --extra train") from exc

    method = config["method"]
    dataset_config = config["dataset"]
    training = config["training"]
    _validate_kernel_combination(training, method)

    train_dataset = load_train_dataset(dataset_config, int(config.get("seed", 42)))
    eval_dataset = load_eval_dataset(dataset_config)
    if eval_dataset is None and dataset_config.get("eval_fraction"):
        split = train_dataset.train_test_split(
            test_size=float(dataset_config["eval_fraction"]),
            seed=int(config.get("seed", 42)),
        )
        train_dataset, eval_dataset = split["train"], split["test"]

    if method == "reasoning_sft":
        train_dataset = prepare_reasoning_sft(train_dataset, dataset_config)
        if eval_dataset is not None:
            eval_dataset = prepare_reasoning_sft(eval_dataset, dataset_config)

    # Limit after filtering/splitting so max_samples means actual usable examples.
    train_dataset = limit_dataset(
        train_dataset,
        dataset_config.get("max_samples"),
        int(config.get("seed", 42)),
    )
    if eval_dataset is not None:
        eval_dataset = limit_dataset(
            eval_dataset,
            dataset_config.get("max_eval_samples"),
            int(config.get("seed", 42)) + 1,
        )

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], use_fast=True)
    if tokenizer.pad_token_id is None:
        # Reusing EOS avoids resizing the frozen 4-bit embedding matrix merely to
        # add padding.  Attention masks keep padding from contributing to the loss.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = load_training_model(config, torch)
    model.config.use_cache = False
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    shared = common_training_args(
        config,
        training,
        has_eval=eval_dataset is not None,
        train_examples=len(train_dataset),
    )

    if method in {"sft", "reasoning_sft"}:
        arguments = SFTConfig(
            **shared,
            max_length=training["max_length"],
            packing=bool(training.get("packing", False)),
            packing_strategy=training.get("packing_strategy", "bfd"),
            padding_free=bool(training.get("padding_free", False)),
            eval_packing=bool(training.get("eval_packing", False)),
            pad_to_multiple_of=training.get("pad_to_multiple_of", 8),
            assistant_only_loss=bool(training.get("assistant_only_loss", True)),
            loss_type=training.get("loss_type"),
            eos_token=training.get("eos_token"),
            neftune_noise_alpha=training.get("neftune_noise_alpha"),
        )
        trainer = SFTTrainer(
            model=model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
    elif method == "dpo":
        arguments = DPOConfig(
            **shared,
            max_length=training["max_length"],
            beta=training["beta"],
            loss_type=training.get("loss_type", "sigmoid"),
            label_smoothing=training.get("label_smoothing", 0.0),
            use_weighting=bool(training.get("use_weighting", False)),
            padding_free=bool(training.get("padding_free", False)),
            pad_to_multiple_of=training.get("pad_to_multiple_of", 8),
            precompute_ref_log_probs=bool(training.get("precompute_ref_log_probs", True)),
            precompute_ref_batch_size=training.get("precompute_ref_batch_size", 1),
        )
        trainer = DPOTrainer(
            model=model,
            # TRL sees the pre-wrapped PeftModel and clones its initial adapter as a
            # frozen ``ref`` adapter on the same quantized base.  This preserves the
            # LoftQ initialization without duplicating 1.7B base weights.  The
            # recommended model_path is still the merged SFT stage, so both policy
            # and reference start from the intended instruction-tuned checkpoint.
            ref_model=None,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
    else:
        raise ValueError(f"Unsupported post-training method: {method}")

    trainer.train(resume_from_checkpoint=resume)
    final_dir = Path(config["output_dir"]) / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    (final_dir / "posttrain-config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (final_dir / "package-versions.json").write_text(
        json.dumps(_package_versions(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return final_dir


def main() -> None:
    """CLI entry point for SFT, reasoning SFT and DPO."""

    parser = argparse.ArgumentParser(description="Run SFT, reasoning SFT, or DPO.")
    parser.add_argument("config")
    parser.add_argument("--resume")
    args = parser.parse_args()
    output = run(load_yaml(args.config), resume=args.resume)
    print(f"Final model or adapter saved to {output}")


if __name__ == "__main__":
    main()
