"""Tests for the bilingual IT/EN reasoning prompt and the formatter language rule.

Covers phase 1 of the bilingual-reasoning goal: the Italian system-prompt variant,
the ``reasoning_system_prompt`` selector, and ``dataset.reasoning_lang`` in
``prepare_reasoning_sft`` ("en" default byte-identical, "it", deterministic "auto"
stopword heuristic with a declared English fallback on ambiguous questions).

Phase 3 (clause E4) adds the IT/EN proportion contract: the ratio is the pair of
``dataset.sources`` weights fed to weighted mixing, and the realized language
distribution is deterministic for a fixed seed — pinned here with exact counts.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from my_llm.config import load_yaml
from my_llm.posttrain import load_train_dataset, prepare_reasoning_sft
from my_llm.reasoning import (
    REASONING_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT_IT,
    gsm8k_messages,
    reasoning_system_prompt,
)

pytest.importorskip("datasets")

from datasets import Dataset

ROOT = Path(__file__).parents[1]

GSM8K_FIXTURE = {
    "question": [
        "What is six plus six?",
        "A farmer has 3 fields with 12 rows each. How many rows are there?",
    ],
    "answer": ["First add.\n#### 12", "Multiply 3 by 12.\n#### 36"],
}

# Clear-cut questions for the auto heuristic: articles, verbs and question words
# from exactly one language.
CLEAR_CASES = [
    ("Quanti metri percorre il treno in tre ore se viaggia a velocità costante?", "it"),
    ("Maria ha dodici mele e ne regala cinque alla sorella. Quante mele le restano?", "it"),
    ("Il costo totale della spesa è di quaranta euro. Quanto spende ogni persona?", "it"),
    ("How many miles does the train travel in three hours at a constant speed?", "en"),
    ("Maria has twelve apples and gives five to her sister. How many apples are left?", "en"),
    ("The total cost is forty dollars. How much does each of the four people pay?", "en"),
]

# Signal-free questions: no stopword of either language, so the declared fallback
# ("en") applies — even to "Calcola 15 * 7.", which is Italian but carries no
# stopword signal.  That limit is by design and documented in the formatter.
AMBIGUOUS_CASES = ["2 + 2 = ?", "Calcola 15 * 7."]


def _gsm8k_config(**extra: str) -> dict[str, str]:
    return {"format": "gsm8k", **extra}


def _system_prompts(formatted: Dataset) -> list[str]:
    return [messages[0]["content"] for messages in formatted["messages"]]


def test_italian_prompt_is_a_faithful_variant() -> None:
    assert REASONING_SYSTEM_PROMPT_IT != REASONING_SYSTEM_PROMPT
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        assert tag in REASONING_SYSTEM_PROMPT_IT


def test_selector_returns_matching_prompt() -> None:
    assert reasoning_system_prompt("en") == REASONING_SYSTEM_PROMPT
    assert reasoning_system_prompt("it") == REASONING_SYSTEM_PROMPT_IT


@pytest.mark.parametrize("lang", ["fr", "auto", "", "EN"])
def test_selector_rejects_unknown_language(lang: str) -> None:
    with pytest.raises(ValueError, match="valid"):
        reasoning_system_prompt(lang)


def test_default_gsm8k_output_is_byte_identical() -> None:
    """Without reasoning_lang the formatter must reproduce the historical output."""

    expected = [
        gsm8k_messages(question, answer)
        for question, answer in zip(GSM8K_FIXTURE["question"], GSM8K_FIXTURE["answer"], strict=True)
    ]
    formatted = prepare_reasoning_sft(Dataset.from_dict(GSM8K_FIXTURE), _gsm8k_config())
    assert formatted.column_names == ["messages"]
    assert formatted["messages"] == expected
    explicit_en = prepare_reasoning_sft(
        Dataset.from_dict(GSM8K_FIXTURE), _gsm8k_config(reasoning_lang="en")
    )
    assert explicit_en["messages"] == expected


def test_default_messages_format_unchanged() -> None:
    conversation = [
        {"role": "system", "content": "custom"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>t</think>\n<answer>2</answer>"},
    ]
    dataset = Dataset.from_dict({"messages": [conversation], "source": ["fixture"]})
    formatted = prepare_reasoning_sft(dataset, {"format": "messages"})
    assert formatted.column_names == ["messages"]
    assert formatted["messages"] == [conversation]


def test_italian_prompt_selected_for_lang_it() -> None:
    formatted = prepare_reasoning_sft(
        Dataset.from_dict(GSM8K_FIXTURE), _gsm8k_config(reasoning_lang="it")
    )
    assert _system_prompts(formatted) == [REASONING_SYSTEM_PROMPT_IT] * len(formatted)
    # Only the system prompt moves: user/assistant turns stay the historical ones.
    expected_tail = [
        gsm8k_messages(question, answer)[1:]
        for question, answer in zip(GSM8K_FIXTURE["question"], GSM8K_FIXTURE["answer"], strict=True)
    ]
    assert [messages[1:] for messages in formatted["messages"]] == expected_tail


def test_italian_prompt_selected_for_openr1_lang_it() -> None:
    dataset = Dataset.from_dict(
        {
            "problem": ["Quanto fa due più due?"],
            "answer": ["4"],
            "generations": [["<think>2 + 2 = 4</think><answer>4</answer>"]],
            "correctness_math_verify": [[True]],
        }
    )
    formatted = prepare_reasoning_sft(
        dataset, {"format": "openr1_math", "reasoning_lang": "it"}
    )
    assert _system_prompts(formatted) == [REASONING_SYSTEM_PROMPT_IT]


def test_auto_heuristic_on_clear_cases() -> None:
    questions = [question for question, _ in CLEAR_CASES]
    dataset = Dataset.from_dict(
        {"question": questions, "answer": ["#### 1"] * len(questions)}
    )
    formatted = prepare_reasoning_sft(dataset, _gsm8k_config(reasoning_lang="auto"))
    expected = [reasoning_system_prompt(lang) for _, lang in CLEAR_CASES]
    assert _system_prompts(formatted) == expected


def test_auto_heuristic_falls_back_to_english_when_ambiguous() -> None:
    dataset = Dataset.from_dict(
        {"question": AMBIGUOUS_CASES, "answer": ["#### 1"] * len(AMBIGUOUS_CASES)}
    )
    formatted = prepare_reasoning_sft(dataset, _gsm8k_config(reasoning_lang="auto"))
    assert _system_prompts(formatted) == [REASONING_SYSTEM_PROMPT] * len(AMBIGUOUS_CASES)


def test_auto_heuristic_is_deterministic() -> None:
    questions = [question for question, _ in CLEAR_CASES] + AMBIGUOUS_CASES
    fixture = {"question": questions, "answer": ["#### 1"] * len(questions)}
    first = prepare_reasoning_sft(Dataset.from_dict(fixture), _gsm8k_config(reasoning_lang="auto"))
    second = prepare_reasoning_sft(Dataset.from_dict(fixture), _gsm8k_config(reasoning_lang="auto"))
    assert first["messages"] == second["messages"]


def test_unknown_reasoning_lang_rejected() -> None:
    with pytest.raises(ValueError, match="reasoning_lang"):
        prepare_reasoning_sft(
            Dataset.from_dict(GSM8K_FIXTURE), _gsm8k_config(reasoning_lang="de")
        )


def test_reasoning_lang_rejected_for_messages_format() -> None:
    dataset = Dataset.from_dict({"messages": [[{"role": "user", "content": "hi"}]]})
    with pytest.raises(ValueError, match="messages"):
        prepare_reasoning_sft(dataset, {"format": "messages", "reasoning_lang": "it"})


# --- E4 (phase 3): IT/EN proportion via weighted sources --------------------------
#
# The proportion lives in the mixing weights (goal B's mixing.py), not in the
# formatter: ``dataset.sources`` weights are relative sampling probabilities, so
# realized counts fluctuate around the configured ratio but are exactly
# reproducible for a fixed seed.  The fixtures carry an explicit ``lang`` column
# so counting needs no heuristic; the smoke-config test below crosses through the
# auto heuristic instead.

# Counts realized by interleave_datasets(all_exhausted) on the 8+8 fixture with
# seed 42.  1:1 realizes 10:8 and 3:1 realizes 28:8 — the ratio is honoured in
# expectation, while ``all_exhausted`` guarantees every EN row still appears.
EXPECTED_MIX_COUNTS = {1.0: {"it": 10, "en": 8}, 3.0: {"it": 28, "en": 8}}


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _mix_config(tmp_path: Path, it_weight: float) -> dict[str, object]:
    en_file = tmp_path / "en.jsonl"
    it_file = tmp_path / "it.jsonl"
    _write_jsonl(
        en_file,
        [
            {"question": f"How many items are in box {i}?", "answer": "#### 1", "lang": "en"}
            for i in range(8)
        ],
    )
    _write_jsonl(
        it_file,
        [
            {"question": f"Quanti oggetti ci sono nella scatola {i}?", "answer": "#### 1", "lang": "it"}
            for i in range(8)
        ],
    )
    return {
        "sources": [
            {"path": "json", "data_files": str(en_file), "split": "train", "weight": 1.0},
            {"path": "json", "data_files": str(it_file), "split": "train", "weight": it_weight},
        ]
    }


@pytest.mark.parametrize("it_weight", [1.0, 3.0])
def test_mix_distribution_exact_counts_for_fixed_seed(tmp_path: Path, it_weight: float) -> None:
    counts = Counter(load_train_dataset(_mix_config(tmp_path, it_weight), 42)["lang"])
    assert counts == EXPECTED_MIX_COUNTS[it_weight]


@pytest.mark.parametrize("it_weight", [1.0, 3.0])
def test_mix_distribution_is_seed_deterministic(tmp_path: Path, it_weight: float) -> None:
    config = _mix_config(tmp_path, it_weight)
    first = Counter(load_train_dataset(config, 42)["lang"])
    second = Counter(load_train_dataset(config, 42)["lang"])
    assert first == second == EXPECTED_MIX_COUNTS[it_weight]
    # The counts are a property of the seed, not of the weights alone: seed 7
    # realizes different draws (measured: 8:8 for 1:1 and 25:8 for 3:1).
    assert Counter(load_train_dataset(config, 7)["lang"]) != first


def test_bilingual_smoke_config_trains_on_both_languages() -> None:
    """The real smoke config's train set mixes IT and EN with matching prompts.

    Mirrors the ``reasoning_sft`` branch of ``run``: ``load_train_dataset`` already
    routes ``sources`` through weighted mixing and ``prepare_reasoning_sft`` is
    applied to the mixed result, so no posttrain change was needed for phase 3.
    Exact counts pinned for seed 42 (EN appears 7 times: the 3-row EN source is
    oversampled by ``all_exhausted`` against the 10-row IT source).
    """

    config = load_yaml(ROOT / "configs/posttrain/reasoning-bilingual-smoke.yaml")
    dataset_config = config["dataset"]
    # data_files paths are repo-relative; anchor them so the test runs from any cwd.
    for source in dataset_config["sources"]:
        source["data_files"] = str(ROOT / source["data_files"])
    mixed = load_train_dataset(dataset_config, int(config["seed"]))
    formatted = prepare_reasoning_sft(mixed, dataset_config)
    prompts = Counter(messages[0]["content"] for messages in formatted["messages"])
    assert prompts == {REASONING_SYSTEM_PROMPT_IT: 10, REASONING_SYSTEM_PROMPT: 7}
