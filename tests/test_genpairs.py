"""Tests for llm-genpairs prompt validation and record schema — no model loading.

The generation loop itself needs a checkpoint and is covered by the CPU smoke
run; these tests defend the pure half: total validation of the prompts file and
the exact output schema the phase-3 judge will consume.
"""

import json
from pathlib import Path

import pytest

from my_llm.genpairs import (
    PromptRow,
    build_record,
    load_prompts,
    resolve_stop_token_ids,
    validate_sampling,
    write_records,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_load_prompts_accepts_valid_rows(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "prompts.jsonl",
        [
            {"id": "p-1", "prompt": "Spiega la ricorsione.", "principle": "P1"},
            {"id": "p-2", "prompt": "What is DPO?"},
        ],
    )
    rows = load_prompts(path)
    assert rows == [
        PromptRow(id="p-1", prompt="Spiega la ricorsione.", principle="P1"),
        PromptRow(id="p-2", prompt="What is DPO?", principle=None),
    ]


def test_missing_or_empty_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="without an id"):
        load_prompts(write_jsonl(tmp_path / "a.jsonl", [{"prompt": "hi"}]))
    with pytest.raises(ValueError, match="without an id"):
        load_prompts(write_jsonl(tmp_path / "b.jsonl", [{"id": " ", "prompt": "hi"}]))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "dup.jsonl",
        [{"id": "p-1", "prompt": "one"}, {"id": "p-1", "prompt": "two"}],
    )
    with pytest.raises(ValueError, match="duplicate prompt id"):
        load_prompts(path)


def test_missing_prompt_and_unknown_fields_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="without a prompt"):
        load_prompts(write_jsonl(tmp_path / "p.jsonl", [{"id": "p-1", "prompt": ""}]))
    with pytest.raises(ValueError, match="unknown prompt fields"):
        load_prompts(
            write_jsonl(tmp_path / "f.jsonl", [{"id": "p-1", "prompt": "hi", "answer": "4"}])
        )


def test_empty_principle_and_empty_file_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="principle"):
        load_prompts(
            write_jsonl(tmp_path / "e.jsonl", [{"id": "p-1", "prompt": "hi", "principle": ""}])
        )
    with pytest.raises(ValueError, match="no prompt rows"):
        load_prompts(write_jsonl(tmp_path / "empty.jsonl", []))


def test_build_record_schema() -> None:
    row = PromptRow(id="p-1", prompt="Ciao", principle="P2")
    record = build_record(
        row, ["risposta A", "risposta B"], model_path="ckpt", seed=7, temperature=0.9, k=2
    )
    assert record == {
        "id": "p-1",
        "prompt": "Ciao",
        "principle": "P2",
        "completions": ["risposta A", "risposta B"],
        "meta": {"model_path": "ckpt", "seed": 7, "temperature": 0.9, "k": 2},
    }


def test_build_record_keeps_principle_key_when_unset() -> None:
    row = PromptRow(id="p-1", prompt="Ciao", principle=None)
    record = build_record(row, ["a", "b"], model_path="ckpt", seed=7, temperature=0.9, k=2)
    # phase-3 consumers must see one schema, so the key is present with null.
    assert record["principle"] is None


def test_build_record_requires_exactly_k_completions() -> None:
    row = PromptRow(id="p-1", prompt="Ciao", principle=None)
    with pytest.raises(ValueError, match="expected 2 completions, got 1"):
        build_record(row, ["only one"], model_path="ckpt", seed=7, temperature=0.9, k=2)


def test_records_round_trip_through_jsonl(tmp_path: Path) -> None:
    records = [
        build_record(
            PromptRow(id="p-1", prompt="Perché il cielo è blu?", principle="P1"),
            ["perché sì", "diffusione di Rayleigh"],
            model_path="ckpt",
            seed=42,
            temperature=0.9,
            k=2,
        ),
        build_record(
            PromptRow(id="p-2", prompt="Say hi", principle=None),
            ["hi", "hello"],
            model_path="ckpt",
            seed=42,
            temperature=0.9,
            k=2,
        ),
    ]
    output = tmp_path / "out.jsonl"
    write_records(records, output)
    with output.open(encoding="utf-8") as handle:
        loaded = [json.loads(line) for line in handle]
    assert loaded == records


def test_resolve_stop_token_ids_unions_tokenizer_and_config() -> None:
    # A merged chat model can keep the base generation config: the chat eos
    # (im_end) and the base eos (endoftext) must both stop generation.
    assert resolve_stop_token_ids(151645, 151643) == [151645, 151643]
    assert resolve_stop_token_ids(151645, [151645, 151643]) == [151645, 151643]
    assert resolve_stop_token_ids(2, None) == [2]
    assert resolve_stop_token_ids(None, [7, 7, 3]) == [7, 3]


def test_validate_sampling_rejects_degenerate_settings() -> None:
    with pytest.raises(ValueError, match="at least two candidates"):
        validate_sampling(1, 0.9, 0.95)
    with pytest.raises(ValueError, match="greedy"):
        validate_sampling(2, 0.0, 0.95)
    with pytest.raises(ValueError, match="top-p"):
        validate_sampling(2, 0.9, 1.5)
    validate_sampling(2, 0.9, 0.95)
