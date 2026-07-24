"""Tests for llm-judge-pairs — total verdicts, principle traceability, no model.

The valid principle ids are parsed from docs/CONSTITUTION.md at run time (the
a recorded decision can amend the set), so the real file is exercised only for parsing; the
behavioural tests pass their own principle sets and constitution stubs.
"""

import json
import re
from pathlib import Path

import pytest

from my_llm.genpairs import PromptRow, build_record, write_records
from my_llm.judge_pairs import (
    Pair,
    build_preferences,
    judge,
    load_pairs,
    load_principles,
    load_verdicts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTITUTION = REPO_ROOT / "docs" / "CONSTITUTION.md"
PRINCIPLES = frozenset({"P1", "P2"})


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def make_pair_file(tmp_path: Path, *pairs: tuple[str, list[str]]) -> Path:
    """Write a pairs file through the real genpairs record builder (round-trip)."""

    records = [
        build_record(
            PromptRow(id=pair_id, prompt=f"prompt for {pair_id}", principle=None),
            completions,
            model_path="ckpt",
            seed=42,
            temperature=0.9,
            k=len(completions),
        )
        for pair_id, completions in pairs
    ]
    output = tmp_path / "pairs.jsonl"
    write_records(records, output)
    return output


def test_load_principles_from_real_constitution() -> None:
    principles = load_principles(CONSTITUTION)
    # The GOAL requires >= 8 principles; the exact set stays unhardcoded (C2).
    assert len(principles) >= 8
    assert all(re.fullmatch(r"P\d+", principle) for principle in principles)


def test_load_principles_rejects_file_without_headings(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"
    path.write_text("# Not a constitution\n\nNo principle headings here.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no principle headings"):
        load_principles(path)


def test_load_principles_rejects_duplicate_headings(tmp_path: Path) -> None:
    path = tmp_path / "dup.md"
    path.write_text("## P1 · Uno\n\n## P1 · Ancora\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate principle headings"):
        load_principles(path)


def test_load_pairs_round_trips_genpairs_output(tmp_path: Path) -> None:
    path = make_pair_file(tmp_path, ("p-1", ["risposta A", "risposta B"]), ("p-2", ["a", "b", "c"]))
    assert load_pairs(path) == [
        Pair(id="p-1", prompt="prompt for p-1", completions=("risposta A", "risposta B")),
        Pair(id="p-2", prompt="prompt for p-2", completions=("a", "b", "c")),
    ]


def test_load_pairs_rejects_malformed_rows(tmp_path: Path) -> None:
    base = {"id": "p-1", "prompt": "hi", "completions": ["a", "b"]}
    with pytest.raises(ValueError, match="unknown pair fields"):
        load_pairs(write_jsonl(tmp_path / "a.jsonl", [{**base, "extra": 1}]))
    with pytest.raises(ValueError, match="duplicate pair id"):
        load_pairs(write_jsonl(tmp_path / "b.jsonl", [base, base]))
    with pytest.raises(ValueError, match="without an id"):
        load_pairs(write_jsonl(tmp_path / "c.jsonl", [{"prompt": "hi", "completions": ["a", "b"]}]))
    with pytest.raises(ValueError, match="without a prompt"):
        load_pairs(write_jsonl(tmp_path / "d.jsonl", [{"id": "p-1", "completions": ["a", "b"]}]))
    with pytest.raises(ValueError, match=">= 2 string completions"):
        load_pairs(write_jsonl(tmp_path / "e.jsonl", [{**base, "completions": ["only one"]}]))
    with pytest.raises(ValueError, match="no pair rows"):
        load_pairs(write_jsonl(tmp_path / "f.jsonl", []))


def test_preference_without_principle_is_rejected(tmp_path: Path) -> None:
    for row in (
        {"id": "p-1", "chosen": 0, "rejected": 1},
        {"id": "p-1", "chosen": 0, "rejected": 1, "principle": " "},
    ):
        with pytest.raises(ValueError, match="no principle"):
            load_verdicts(write_jsonl(tmp_path / "v.jsonl", [row]), PRINCIPLES)


def test_principle_unknown_to_the_constitution_is_rejected(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "v.jsonl", [{"id": "p-1", "chosen": 0, "rejected": 1, "principle": "P99"}]
    )
    with pytest.raises(ValueError, match=r"unknown principle 'P99'.*\['P1', 'P2'\]"):
        load_verdicts(path, PRINCIPLES)


def test_chosen_equal_to_rejected_is_rejected(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "v.jsonl", [{"id": "p-1", "chosen": 1, "rejected": 1, "principle": "P1"}]
    )
    with pytest.raises(ValueError, match="chosen == rejected"):
        load_verdicts(path, PRINCIPLES)


def test_non_index_chosen_or_rejected_is_rejected(tmp_path: Path) -> None:
    for broken in (
        {"id": "p-1", "rejected": 1, "principle": "P1"},  # missing
        {"id": "p-1", "chosen": -1, "rejected": 1, "principle": "P1"},  # negative
        {"id": "p-1", "chosen": True, "rejected": 1, "principle": "P1"},  # bool
        {"id": "p-1", "chosen": "0", "rejected": 1, "principle": "P1"},  # string
    ):
        with pytest.raises(ValueError, match="completion index"):
            load_verdicts(write_jsonl(tmp_path / "v.jsonl", [broken]), PRINCIPLES)


def test_duplicate_and_unknown_shape_verdicts_are_rejected(tmp_path: Path) -> None:
    good = {"id": "p-1", "chosen": 0, "rejected": 1, "principle": "P1"}
    with pytest.raises(ValueError, match="duplicate verdict"):
        load_verdicts(write_jsonl(tmp_path / "a.jsonl", [good, good]), PRINCIPLES)
    with pytest.raises(ValueError, match="unknown verdict fields"):
        load_verdicts(write_jsonl(tmp_path / "b.jsonl", [{**good, "score": 3}]), PRINCIPLES)
    with pytest.raises(ValueError, match="without an id"):
        load_verdicts(
            write_jsonl(tmp_path / "c.jsonl", [{"chosen": 0, "rejected": 1, "principle": "P1"}]),
            PRINCIPLES,
        )


def test_discard_needs_a_note_and_a_known_verdict_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a note"):
        load_verdicts(
            write_jsonl(tmp_path / "a.jsonl", [{"id": "p-1", "verdict": "discard"}]), PRINCIPLES
        )
    with pytest.raises(ValueError, match="unknown verdict 'approve'"):
        load_verdicts(
            write_jsonl(tmp_path / "b.jsonl", [{"id": "p-1", "verdict": "approve"}]), PRINCIPLES
        )
    with pytest.raises(ValueError, match="unknown discard fields"):
        load_verdicts(
            write_jsonl(
                tmp_path / "c.jsonl",
                [{"id": "p-1", "verdict": "discard", "note": "ambigua", "chosen": 0}],
            ),
            PRINCIPLES,
        )


def test_note_and_unreviewed_types_are_validated(tmp_path: Path) -> None:
    base = {"id": "p-1", "chosen": 0, "rejected": 1, "principle": "P1"}
    with pytest.raises(ValueError, match=r"unreviewed.*must be a boolean"):
        load_verdicts(write_jsonl(tmp_path / "a.jsonl", [{**base, "unreviewed": "yes"}]), PRINCIPLES)
    with pytest.raises(ValueError, match=r"note.*must be a string"):
        load_verdicts(write_jsonl(tmp_path / "b.jsonl", [{**base, "note": 3}]), PRINCIPLES)


def test_totality_missing_and_unknown_ids_are_rejected() -> None:
    pairs = [
        Pair(id="p-1", prompt="uno", completions=("a", "b")),
        Pair(id="p-2", prompt="due", completions=("a", "b")),
    ]
    only_first = {"p-1": {"id": "p-1", "chosen": 0, "rejected": 1, "principle": "P1"}}
    with pytest.raises(ValueError, match=r"1 pairs without a verdict.*'p-2'"):
        build_preferences(pairs, only_first)
    stray = {
        **only_first,
        "p-2": {"id": "p-2", "verdict": "discard", "note": "ambigua"},
        "p-9": {"id": "p-9", "chosen": 0, "rejected": 1, "principle": "P1"},
    }
    with pytest.raises(ValueError, match=r"unknown pair ids.*'p-9'"):
        build_preferences(pairs, stray)


def test_out_of_range_index_is_rejected() -> None:
    pairs = [Pair(id="p-1", prompt="uno", completions=("a", "b"))]
    verdicts = {"p-1": {"id": "p-1", "chosen": 0, "rejected": 2, "principle": "P1"}}
    with pytest.raises(ValueError, match="rejected index 2 out of range"):
        build_preferences(pairs, verdicts)


def test_unreviewed_propagates_and_defaults_to_false() -> None:
    pairs = [
        Pair(id="p-1", prompt="uno", completions=("a", "b")),
        Pair(id="p-2", prompt="due", completions=("a", "b")),
    ]
    verdicts = {
        "p-1": {"id": "p-1", "chosen": 0, "rejected": 1, "principle": "P1", "unreviewed": True},
        "p-2": {"id": "p-2", "chosen": 1, "rejected": 0, "principle": "P2"},
    }
    preferences, discarded = build_preferences(pairs, verdicts)
    assert discarded == 0
    assert [row["unreviewed"] for row in preferences] == [True, False]
    assert [row["principle"] for row in preferences] == ["P1", "P2"]


def test_judge_end_to_end_round_trip(tmp_path: Path) -> None:
    constitution = tmp_path / "constitution.md"
    constitution.write_text("## P1 · Uno\n\n## P7 · Sette\n", encoding="utf-8")
    pairs_path = make_pair_file(
        tmp_path, ("p-1", ["risposta A", "risposta B"]), ("p-2", ["hi", "hello"])
    )
    verdicts_path = write_jsonl(
        tmp_path / "verdicts.jsonl",
        [
            {"id": "p-1", "chosen": 1, "rejected": 0, "principle": "P7", "unreviewed": True},
            {"id": "p-2", "verdict": "discard", "note": "nessun principio discrimina"},
        ],
    )
    output = tmp_path / "preferences.jsonl"
    summary = judge(pairs_path, verdicts_path, output, constitution)
    assert summary == {"pairs": 2, "preferences": 1, "discarded": 1, "unreviewed": 1}
    with output.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    # TRL conversational preference format, exactly as sample_data/preferences.jsonl.
    assert rows == [
        {
            "id": "p-1",
            "prompt": [{"role": "user", "content": "prompt for p-1"}],
            "chosen": [{"role": "assistant", "content": "risposta B"}],
            "rejected": [{"role": "assistant", "content": "risposta A"}],
            "principle": "P7",
            "unreviewed": True,
        }
    ]
