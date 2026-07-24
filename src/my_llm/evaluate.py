"""Batched GSM8K exact-match evaluation for merged or adapter checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from my_llm.adapters import load_inference_model
from my_llm.reasoning import answers_equal, extract_final_answer, gsm8k_prompt


def evaluate(
    model_path: str,
    *,
    adapter_path: str | None,
    load_in_4bit: bool,
    attention_backend: str,
    max_samples: int,
    samples_per_prompt: int,
    batch_size: int,
    max_new_tokens: int,
    output: Path | None,
    seed: int,
) -> dict[str, float | int]:
    """Measure greedy accuracy@1 and empirical pass@k on held-out GSM8K."""

    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install training dependencies: uv sync --extra train") from exc

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
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    first_correct = 0
    any_correct = 0
    records = []

    # Batched generation amortizes Python and kernel-launch overhead.  Left padding
    # lets every decoded completion start after the same padded input width.
    for batch_start in range(0, len(dataset), batch_size):
        rows = [
            dataset[index]
            for index in range(batch_start, min(batch_start + batch_size, len(dataset)))
        ]
        rendered = [
            tokenizer.apply_chat_template(
                gsm8k_prompt(row["question"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in rows
        ]
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": samples_per_prompt,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if samples_per_prompt > 1:
            generation_kwargs.update({"do_sample": True, "temperature": 0.7, "top_p": 0.95})
        with torch.inference_mode():
            outputs = model.generate(**encoded, **generation_kwargs)

        input_width = encoded["input_ids"].shape[1]
        for row_offset, row in enumerate(rows):
            start = row_offset * samples_per_prompt
            group = outputs[start : start + samples_per_prompt]
            completions = [
                tokenizer.decode(sequence[input_width:], skip_special_tokens=False)
                for sequence in group
            ]
            reference = extract_final_answer(row["answer"])
            correctness = [
                answers_equal(extract_final_answer(completion), reference)
                for completion in completions
            ]
            first_correct += int(correctness[0])
            any_correct += int(any(correctness))
            records.append(
                {
                    "index": batch_start + row_offset,
                    "question": row["question"],
                    "reference": reference,
                    "completions": completions,
                    "correct": correctness,
                }
            )
        evaluated = min(batch_start + len(rows), len(dataset))
        if evaluated % 25 == 0 or evaluated == len(dataset):
            print(f"evaluated={evaluated}/{len(dataset)}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    count = len(dataset)
    return {
        "samples": count,
        "samples_per_prompt": samples_per_prompt,
        "accuracy_at_1": first_correct / max(count, 1),
        "pass_at_k_empirical": any_correct / max(count, 1),
    }


def main() -> None:
    """CLI entry point for reproducible GSM8K evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate exact-match reasoning on GSM8K test.")
    parser.add_argument("model_path")
    parser.add_argument("--adapter")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = evaluate(
        args.model_path,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
        attention_backend=args.attention_backend,
        max_samples=args.max_samples,
        samples_per_prompt=args.samples_per_prompt,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output=args.output,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
