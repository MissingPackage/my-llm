"""Generate K sampled candidate completions per prompt for constitutional DPO.

Preference pairs need genuinely different candidates, so generation samples with
temperature/top-p instead of the greedy decoding evaluation uses.  Everything
that does not need a model — prompt validation and output-record construction —
lives in pure functions, so the schema the phase-3 judge consumes is testable
without loading a checkpoint.  Validation is total in the benchmark_freeze
sense: a malformed prompt row is a speaking error with a line number, never a
silent skip.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from my_llm.adapters import load_inference_model

PROMPT_FIELDS = frozenset({"id", "prompt", "principle"})


def resolve_stop_token_ids(
    tokenizer_eos: int | list[int] | None, config_eos: int | list[int] | None
) -> list[int]:
    """Union of every stop id the tokenizer and the generation config declare.

    A merged chat model can carry the base model's generation config, whose eos
    (e.g. endoftext) differs from the chat template's turn terminator (im_end).
    Generation must stop on either: a sampled config-eos that is not in the
    stop set sails through and the completion continues as noise.
    """

    ids: list[int] = []
    for value in (tokenizer_eos, config_eos):
        for item in value if isinstance(value, (list, tuple)) else [value]:
            if item is not None and item not in ids:
                ids.append(item)
    return ids


@dataclass(frozen=True)
class PromptRow:
    """One validated generation request; ``principle`` stays optional until phase 3."""

    id: str
    prompt: str
    principle: str | None


def load_prompts(path: str | Path) -> list[PromptRow]:
    """Read the prompts file and reject rows generation cannot act on."""

    path = Path(path)
    rows: list[PromptRow] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            unknown = set(row) - PROMPT_FIELDS
            if unknown:
                raise ValueError(f"{path}:{line_number}: unknown prompt fields {sorted(unknown)}")
            item_id = row.get("id")
            if not isinstance(item_id, str) or not item_id.strip():
                raise ValueError(f"{path}:{line_number}: prompt row without an id")
            if item_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate prompt id {item_id!r}")
            seen.add(item_id)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number}: prompt row {item_id!r} without a prompt")
            principle = row.get("principle")
            if principle is not None and (not isinstance(principle, str) or not principle.strip()):
                raise ValueError(
                    f"{path}:{line_number}: principle of {item_id!r} must be a non-empty string"
                )
            rows.append(PromptRow(id=item_id, prompt=prompt, principle=principle))
    if not rows:
        raise ValueError(f"{path}: no prompt rows — an empty batch is an error, not a no-op")
    return rows


def validate_sampling(k: int, temperature: float, top_p: float) -> None:
    """Reject settings that cannot yield usable preference candidates."""

    if k < 2:
        raise ValueError("k must be >= 2: preference pairs need at least two candidates")
    if temperature <= 0:
        raise ValueError("temperature must be > 0: greedy decoding makes all K candidates equal")
    if not 0 < top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")


def build_record(
    row: PromptRow,
    completions: list[str],
    *,
    model_path: str,
    seed: int,
    temperature: float,
    k: int,
) -> dict[str, object]:
    """Assemble one output row; the metadata makes every batch re-runnable and auditable."""

    if len(completions) != k:
        raise ValueError(f"prompt {row.id!r}: expected {k} completions, got {len(completions)}")
    return {
        "id": row.id,
        "prompt": row.prompt,
        # The key is always present (null when unset) so phase-3 consumers see one schema.
        "principle": row.principle,
        "completions": list(completions),
        "meta": {"model_path": model_path, "seed": seed, "temperature": temperature, "k": k},
    }


def write_records(records: list[dict[str, object]], output: Path) -> None:
    """Write JSONL exactly as downstream tools will read it: UTF-8, one object per line."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_pairs(
    model_path: str,
    *,
    prompts: Path,
    output: Path,
    k: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    max_new_tokens: int,
    seed: int,
    load_in_4bit: bool,
    attention_backend: str,
) -> list[dict[str, object]]:
    """Generate K sampled completions per prompt, batched with left padding.

    One seed before the whole run: with a fixed prompt order, batch size and
    hardware, every sampling draw — and therefore the output file — reproduces.
    Changing ``batch_size`` may legitimately change the samples (padding width
    and RNG consumption order shift), which is why the seed lands in ``meta``.
    """

    validate_sampling(k, temperature, top_p)
    rows = load_prompts(prompts)

    try:
        import torch
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
        adapter_path=None,
        load_in_4bit=load_in_4bit,
        attention_backend=attention_backend,
        torch=torch,
    )
    device = next(model.parameters()).device
    stop_token_ids = resolve_stop_token_ids(
        tokenizer.eos_token_id, model.generation_config.eos_token_id
    )

    # Each prompt appears K consecutive times, so one prompt's candidates may
    # split across batches; grouping back relies only on this fixed order.
    expanded = [row for row in rows for _ in range(k)]
    completions: dict[str, list[str]] = {row.id: [] for row in rows}
    for batch_start in range(0, len(expanded), batch_size):
        batch = expanded[batch_start : batch_start + batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
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
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )
        input_width = encoded["input_ids"].shape[1]
        for offset, row in enumerate(batch):
            # Candidates go to the judge/reviewer as plain text, so special tokens are stripped.
            completions[row.id].append(
                tokenizer.decode(outputs[offset][input_width:], skip_special_tokens=True)
            )
        print(f"generated={batch_start + len(batch)}/{len(expanded)}")

    records = [
        build_record(
            row,
            completions[row.id],
            model_path=model_path,
            seed=seed,
            temperature=temperature,
            k=k,
        )
        for row in rows
    ]
    write_records(records, output)
    return records


def main() -> None:
    """CLI entry point for generating preference candidates from a local checkpoint."""

    parser = argparse.ArgumentParser(
        description="Generate K sampled candidate completions per prompt for preference pairs."
    )
    parser.add_argument("model_path")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attention-backend", default="sdpa")
    args = parser.parse_args()
    records = generate_pairs(
        args.model_path,
        prompts=args.prompts,
        output=args.output,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        load_in_4bit=args.load_in_4bit,
        attention_backend=args.attention_backend,
    )
    print(f"Wrote {len(records)} records x {args.k} completions to {args.output}")


if __name__ == "__main__":
    main()
