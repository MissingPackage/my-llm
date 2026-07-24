"""Tests for the mechanical unreviewed smoke verdicts (decision C4)."""

import pytest

from my_llm.smoke_verdicts import synthetic_verdicts


def record(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "sp-001",
        "prompt": "q?",
        "principle": "P1",
        "completions": ["short", "a longer completion"],
        "meta": {"seed": 42},
    }
    row.update(overrides)
    return row


def test_chosen_is_shortest_and_verdict_is_marked_mechanical() -> None:
    [verdict] = synthetic_verdicts([record(completions=["a longer completion", "short"])])
    assert verdict["chosen"] == 1
    assert verdict["rejected"] == 0
    assert verdict["principle"] == "P1"
    assert verdict["unreviewed"] is True
    assert "meccanico" in str(verdict["note"])


def test_equal_lengths_break_ties_by_index() -> None:
    [verdict] = synthetic_verdicts([record(completions=["aaa", "bbb"])])
    assert (verdict["chosen"], verdict["rejected"]) == (0, 1)


def test_empty_completion_becomes_explicit_discard() -> None:
    [verdict] = synthetic_verdicts([record(completions=["   ", "text"])])
    assert verdict["verdict"] == "discard"
    assert verdict["unreviewed"] is True
    assert str(verdict["note"]).strip()


def test_identical_completions_become_explicit_discard() -> None:
    [verdict] = synthetic_verdicts([record(completions=["same", "same"])])
    assert verdict["verdict"] == "discard"


def test_missing_principle_is_an_error() -> None:
    with pytest.raises(ValueError, match="principle"):
        synthetic_verdicts([record(principle=None)])
