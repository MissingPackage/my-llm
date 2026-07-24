"""Integrity tests for the personal benchmark schema and loader."""

import json
from pathlib import Path

import pytest

from my_llm.benchmark import BenchmarkItem, load_benchmark, load_v1

ROOT = Path(__file__).parents[1]


def valid_item(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "it-math-001",
        "lang": "it",
        "domain": "math",
        "prompt": "Quanto fa 2 + 2?",
        "reference": "4",
        "verification": "exact_numeric",
    }
    base.update(overrides)
    return base


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_every_verification_mode_parses() -> None:
    BenchmarkItem.model_validate(valid_item())
    BenchmarkItem.model_validate(
        valid_item(id="s1", verification="exact_string", reference="Roma")
    )
    BenchmarkItem.model_validate(valid_item(id="r1", verification="regex", reference=r"^\d+$"))
    BenchmarkItem.model_validate(
        valid_item(id="j1", verification="llm_rubric", rubric="Concise and in Italian.")
    )


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="extra"):
        BenchmarkItem.model_validate(valid_item(surprise=True))


def test_llm_rubric_requires_rubric_and_others_forbid_it() -> None:
    with pytest.raises(ValueError, match="rubric"):
        BenchmarkItem.model_validate(valid_item(verification="llm_rubric"))
    with pytest.raises(ValueError, match="rubric"):
        BenchmarkItem.model_validate(valid_item(rubric="not allowed here"))


def test_ungradable_references_are_rejected() -> None:
    with pytest.raises(ValueError, match="regex"):
        BenchmarkItem.model_validate(valid_item(verification="regex", reference="(unclosed"))
    with pytest.raises(ValueError, match="canonicalize"):
        BenchmarkItem.model_validate(valid_item(reference="quattro"))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "dup.jsonl", [valid_item(), valid_item()])
    with pytest.raises(ValueError, match="duplicate"):
        load_benchmark(path)


def test_release_file_requires_approval_and_minimums(tmp_path: Path) -> None:
    rows = [valid_item(id=f"i{n}", approved=n != 3) for n in range(5)]
    path = write_jsonl(tmp_path / "v1.jsonl", rows)
    with pytest.raises(ValueError, match="unapproved"):
        load_benchmark(path, require_approved=True)
    with pytest.raises(ValueError, match="minimum"):
        load_benchmark(path, minimum_items=6)
    with pytest.raises(ValueError, match="calibration"):
        load_benchmark(path, minimum_calibration=1)
    # The full v1 profile also enforces the 100-item release threshold.
    with pytest.raises(ValueError, match="minimum"):
        load_v1(path)


def test_draft_profile_accepts_unapproved_items(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "draft.jsonl", [valid_item(id="d1"), valid_item(id="d2")])
    items = load_benchmark(path)
    assert [item.approved for item in items] == [False, False]


def test_no_training_config_references_the_benchmark() -> None:
    """The benchmark is held-out by construction: no YAML may point at it."""

    for config_path in (ROOT / "configs").rglob("*.yaml"):
        text = config_path.read_text(encoding="utf-8")
        assert "benchmarks/personal" not in text, f"{config_path} references the benchmark"
