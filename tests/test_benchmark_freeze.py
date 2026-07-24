"""Tests for the verdict-driven release freeze and the anti-drift guarantee."""

import hashlib
import json
from pathlib import Path

import pytest

from my_llm.benchmark import load_benchmark
from my_llm.benchmark_freeze import load_verdicts, promote

ROOT = Path(__file__).parents[1]
V1 = ROOT / "benchmarks/personal/v1.jsonl"


def draft_row(item_id: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": item_id,
        "lang": "en",
        "domain": "math",
        "prompt": "What is 2 + 2?",
        "reference": "4",
        "verification": "exact_numeric",
        "calibration": item_id.endswith("cal"),
    }
    base.update(overrides)
    return base


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def make_inputs(tmp_path: Path, verdict_rows: list[dict[str, object]]) -> tuple[Path, Path, Path]:
    drafts = write_jsonl(
        tmp_path / "drafts.jsonl", [draft_row("a-1"), draft_row("a-2"), draft_row("a-3cal")]
    )
    verdicts = write_jsonl(tmp_path / "verdicts.jsonl", verdict_rows)
    return drafts, verdicts, tmp_path / "v1.jsonl"


def test_promote_applies_verdicts_and_freezes(tmp_path: Path) -> None:
    drafts, verdicts, output = make_inputs(
        tmp_path,
        [
            {"id": "a-1", "verdict": "approve"},
            {"id": "a-2", "verdict": "edit", "reference": "5", "prompt": "What is 2 + 3?"},
            {"id": "a-3cal", "verdict": "approve"},
        ],
    )
    summary = promote(drafts, verdicts, output, minimum_items=3, minimum_calibration=1)
    assert summary == {"drafts": 3, "approved": 3, "edited": 1, "rejected": 0}
    released = load_benchmark(output, require_approved=True)
    assert [item.id for item in released] == ["a-1", "a-2", "a-3cal"]
    assert released[1].reference == "5"
    digest = output.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()


def test_rejects_are_excluded_and_counted(tmp_path: Path) -> None:
    drafts, verdicts, output = make_inputs(
        tmp_path,
        [
            {"id": "a-1", "verdict": "reject", "note": "ambiguous"},
            {"id": "a-2", "verdict": "approve"},
            {"id": "a-3cal", "verdict": "approve"},
        ],
    )
    summary = promote(drafts, verdicts, output, minimum_items=2, minimum_calibration=1)
    assert summary["rejected"] == 1
    assert [item.id for item in load_benchmark(output)] == ["a-2", "a-3cal"]


def test_approval_must_be_total(tmp_path: Path) -> None:
    drafts, verdicts, output = make_inputs(tmp_path, [{"id": "a-1", "verdict": "approve"}])
    with pytest.raises(ValueError, match="without a verdict"):
        promote(drafts, verdicts, output, minimum_items=1, minimum_calibration=0)


def test_undecided_and_unknown_verdicts_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="undecided"):
        load_verdicts(write_jsonl(tmp_path / "u.jsonl", [{"id": "a-1", "verdict": "_"}]))
    with pytest.raises(ValueError, match="unknown verdict"):
        load_verdicts(write_jsonl(tmp_path / "w.jsonl", [{"id": "a-1", "verdict": "maybe"}]))
    with pytest.raises(ValueError, match="unknown verdict fields"):
        load_verdicts(
            write_jsonl(tmp_path / "f.jsonl", [{"id": "a-1", "verdict": "edit", "answer": "5"}])
        )


def test_verdicts_for_unknown_ids_are_rejected(tmp_path: Path) -> None:
    drafts, verdicts, output = make_inputs(
        tmp_path,
        [
            {"id": "a-1", "verdict": "approve"},
            {"id": "a-2", "verdict": "approve"},
            {"id": "a-3cal", "verdict": "approve"},
            {"id": "ghost", "verdict": "approve"},
        ],
    )
    with pytest.raises(ValueError, match="unknown item ids"):
        promote(drafts, verdicts, output, minimum_items=1, minimum_calibration=0)


def test_edit_cannot_break_gradability(tmp_path: Path) -> None:
    drafts, verdicts, output = make_inputs(
        tmp_path,
        [
            {"id": "a-1", "verdict": "edit", "reference": "not a number"},
            {"id": "a-2", "verdict": "approve"},
            {"id": "a-3cal", "verdict": "approve"},
        ],
    )
    with pytest.raises(ValueError, match="canonicalize"):
        promote(drafts, verdicts, output, minimum_items=1, minimum_calibration=0)


def test_release_is_frozen(tmp_path: Path) -> None:
    drafts, verdicts, output = make_inputs(
        tmp_path,
        [
            {"id": "a-1", "verdict": "approve"},
            {"id": "a-2", "verdict": "approve"},
            {"id": "a-3cal", "verdict": "approve"},
        ],
    )
    promote(drafts, verdicts, output, minimum_items=1, minimum_calibration=0)
    with pytest.raises(FileExistsError, match="frozen"):
        promote(drafts, verdicts, output, minimum_items=1, minimum_calibration=0)


def test_published_v1_has_not_drifted() -> None:
    """Anti-drift: once v1 exists, its bytes must match the recorded fingerprint."""

    if not V1.exists():
        pytest.skip("v1.jsonl not released yet (phase 4 pending)")
    recorded = V1.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    assert recorded == hashlib.sha256(V1.read_bytes()).hexdigest()
