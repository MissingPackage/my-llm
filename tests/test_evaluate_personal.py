"""Unit tests for personal-benchmark scoring and aggregation (no model, no downloads)."""

from my_llm.benchmark import BenchmarkItem
from my_llm.evaluate_personal import aggregate_results, score_completion


def make_item(**overrides: object) -> BenchmarkItem:
    base: dict[str, object] = {
        "id": "en-math-001",
        "lang": "en",
        "domain": "math",
        "prompt": "What is 2 + 2?",
        "reference": "4",
        "verification": "exact_numeric",
    }
    base.update(overrides)
    return BenchmarkItem.model_validate(base)


def test_exact_numeric_scores_canonical_equivalents() -> None:
    item = make_item(reference="1000")
    assert score_completion(item, "<think>10^3</think>\n<answer>1,000</answer>") is True
    assert score_completion(item, "<think>10^3</think>\n<answer>1000.0</answer>") is True


def test_exact_numeric_ignores_numbers_in_the_reasoning() -> None:
    # Adversarial: the right number appears in the trace but the final answer is wrong.
    item = make_item(reference="4")
    assert score_completion(item, "<think>2 + 2 = 4, but wait</think>\n<answer>5</answer>") is False


def test_exact_string_is_case_insensitive_but_exact() -> None:
    item = make_item(id="s1", verification="exact_string", reference="Roma")
    assert score_completion(item, "<answer>roma</answer>") is True
    # Adversarial: extra words around the right string are not an exact match.
    assert score_completion(item, "<answer>Roma antica</answer>") is False


def test_exact_string_requires_an_extractable_answer() -> None:
    # Adversarial: the right string in free text has no <answer>/####/boxed envelope,
    # so extraction yields nothing and the completion cannot score.
    item = make_item(id="s2", verification="exact_string", reference="Roma")
    assert score_completion(item, "The capital of Italy is Roma.") is False


def test_regex_searches_the_full_completion_including_reasoning() -> None:
    # Pinned behavior: the pattern sees reasoning too, so a match inside <think>
    # counts.  Items that must match only the answer anchor on the envelope.
    item = make_item(id="r1", verification="regex", reference=r"\bRoma\b")
    assert score_completion(item, "<think>Roma or Milano?</think>\n<answer>Milano</answer>") is True
    anchored = make_item(id="r2", verification="regex", reference=r"<answer>\s*Roma\s*</answer>")
    assert (
        score_completion(anchored, "<think>Roma or Milano?</think>\n<answer>Milano</answer>")
        is False
    )
    assert score_completion(anchored, "<think>easy</think>\n<answer>Roma</answer>") is True


def test_llm_rubric_is_skipped_not_graded() -> None:
    item = make_item(id="j1", verification="llm_rubric", rubric="Concise and in Italian.")
    # Adversarial: even a completion quoting the reference verbatim stays unjudged.
    assert score_completion(item, f"<answer>{item.reference}</answer>") is None


def test_aggregate_excludes_calibration_from_the_gate() -> None:
    gate_hit = make_item(id="g1")
    gate_miss = make_item(id="g2")
    calibration = make_item(id="c1", calibration=True)
    summary = aggregate_results([(gate_hit, True), (gate_miss, False), (calibration, True)])
    assert summary["gate_items"] == 2
    assert summary["gate_accuracy"] == 0.5
    assert summary["calibration_items"] == 1
    assert summary["calibration_accuracy"] == 1.0
    assert summary["items"] == 3


def test_aggregate_splits_by_lang_and_domain() -> None:
    scored = [
        (make_item(id="a", lang="it", domain="math"), True),
        (make_item(id="b", lang="it", domain="code"), False),
        (make_item(id="c", lang="en", domain="math"), True),
    ]
    summary = aggregate_results(scored)
    assert summary["gate_accuracy_by_lang"] == {"en": 1.0, "it": 0.5}
    assert summary["gate_accuracy_by_domain"] == {"code": 0.0, "math": 1.0}


def test_aggregate_counts_llm_rubric_as_explicitly_skipped() -> None:
    rubric = make_item(id="j1", verification="llm_rubric", rubric="Styled.")
    summary = aggregate_results([(rubric, None), (make_item(id="g1"), True)])
    assert summary["skipped_llm_rubric"] == 1
    assert summary["gate_items"] == 1
    assert summary["gate_accuracy"] == 1.0


def test_aggregate_reports_empty_groups_as_none_not_zero() -> None:
    assert aggregate_results([]) == {
        "items": 0,
        "gate_items": 0,
        "gate_accuracy": None,
        "gate_accuracy_by_lang": {},
        "gate_accuracy_by_domain": {},
        "calibration_items": 0,
        "calibration_accuracy": None,
        "skipped_llm_rubric": 0,
    }
    # All-rubric input must not fabricate a 0% gate out of zero gradable items.
    rubric = make_item(id="j1", verification="llm_rubric", rubric="Styled.")
    summary = aggregate_results([(rubric, None), (rubric, None)])
    assert summary["gate_accuracy"] is None
    assert summary["skipped_llm_rubric"] == 2
