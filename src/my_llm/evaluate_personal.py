"""Batched evaluation of a checkpoint on the personal held-out benchmark.

The personal benchmark is the go/no-go gate of every pipeline stage, so scoring is
kept pure and separate from generation: :func:`score_completion` and
:func:`aggregate_results` are unit-testable without loading a model.  The gate
aggregates deterministic non-calibration items only; calibration items measure
harness health and are reported separately, and ``llm_rubric`` items are skipped
with an explicit counter until the pinned judge exists (recorded decision A5).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from my_llm.adapters import load_inference_model
from my_llm.benchmark import BenchmarkItem, load_benchmark, load_v1
from my_llm.reasoning import answers_equal, extract_final_answer, gsm8k_prompt


def score_completion(item: BenchmarkItem, text: str) -> bool | None:
    """Grade one completion deterministically; ``None`` marks an unjudged item."""

    if item.verification == "llm_rubric":
        # Judged scoring waits on the pinned judge (recorded decision A5); until then
        # these items are counted as skipped in the report, never silently dropped.
        return None
    if item.verification == "regex":
        # The pattern sees the full completion, reasoning included; items that must
        # only match the final answer have to anchor on the <answer> envelope.
        return re.search(item.reference, text) is not None
    return answers_equal(extract_final_answer(text), item.reference)


def _accuracy(flags: list[bool]) -> float | None:
    """Mean of a group; ``None`` keeps an empty group from reading as 0% accuracy."""

    return sum(flags) / len(flags) if flags else None


def aggregate_results(scored: list[tuple[BenchmarkItem, bool | None]]) -> dict[str, object]:
    """Fold per-item scores into gate, calibration and skip metrics.

    Calibration items are trivially solvable on purpose: they expose a broken
    harness, so counting them in the gate would inflate it.  They get their own
    accuracy line instead.
    """

    gate: list[bool] = []
    by_lang: dict[str, list[bool]] = {}
    by_domain: dict[str, list[bool]] = {}
    calibration: list[bool] = []
    skipped_llm_rubric = 0
    for item, correct in scored:
        if correct is None:
            # score_completion returns None exclusively for llm_rubric items.
            skipped_llm_rubric += 1
        elif item.calibration:
            calibration.append(correct)
        else:
            gate.append(correct)
            by_lang.setdefault(item.lang, []).append(correct)
            by_domain.setdefault(item.domain, []).append(correct)
    return {
        "items": len(scored),
        "gate_items": len(gate),
        "gate_accuracy": _accuracy(gate),
        "gate_accuracy_by_lang": {
            lang: _accuracy(flags) for lang, flags in sorted(by_lang.items())
        },
        "gate_accuracy_by_domain": {
            domain: _accuracy(flags) for domain, flags in sorted(by_domain.items())
        },
        "calibration_items": len(calibration),
        "calibration_accuracy": _accuracy(calibration),
        "skipped_llm_rubric": skipped_llm_rubric,
    }


def evaluate_personal(
    model_path: str,
    *,
    benchmark: Path,
    adapter_path: str | None,
    load_in_4bit: bool,
    attention_backend: str,
    batch_size: int,
    max_new_tokens: int,
    output: Path | None,
    seed: int,
    allow_draft: bool,
) -> dict[str, object]:
    """Generate one greedy completion per benchmark item and score it."""

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install training dependencies: uv sync --extra train") from exc

    # Gate runs must go through the frozen-release rules (approved-only, release
    # thresholds); the permissive loader is opt-in for smoke and draft batches.
    items = load_benchmark(benchmark) if allow_draft else load_v1(benchmark)

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_inference_model(
        model_path,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
        attention_backend=attention_backend,
        torch=torch,
    )
    device = next(model.parameters()).device
    scored: list[tuple[BenchmarkItem, bool | None]] = []
    records: list[dict[str, object]] = []

    # Batched generation amortizes Python and kernel-launch overhead.  Left padding
    # lets every decoded completion start after the same padded input width.
    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start : batch_start + batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                gsm8k_prompt(item.prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in batch
        ]
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_width = encoded["input_ids"].shape[1]
        for offset, item in enumerate(batch):
            completion = tokenizer.decode(outputs[offset][input_width:], skip_special_tokens=False)
            correct = score_completion(item, completion)
            scored.append((item, correct))
            records.append(
                {
                    "id": item.id,
                    "lang": item.lang,
                    "domain": item.domain,
                    "verification": item.verification,
                    "calibration": item.calibration,
                    "completion": completion,
                    "correct": correct,
                }
            )
        evaluated = min(batch_start + len(batch), len(items))
        if evaluated % 25 == 0 or evaluated == len(items):
            print(f"evaluated={evaluated}/{len(items)}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return aggregate_results(scored)


def main() -> None:
    """CLI entry point for evaluating a checkpoint on the personal benchmark."""

    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the personal held-out benchmark."
    )
    parser.add_argument("model_path")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Load with the permissive draft rules instead of the frozen v1 profile.",
    )
    args = parser.parse_args()
    result = evaluate_personal(
        args.model_path,
        benchmark=args.benchmark,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
        attention_backend=args.attention_backend,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output=args.output,
        seed=args.seed,
        allow_draft=args.allow_draft,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
