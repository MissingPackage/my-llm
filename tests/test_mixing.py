"""Weighted mixing tests: proportions, determinism, validation, non-regression.

``load_mixed_dataset`` needs the optional ``datasets`` dependency, so the whole
module skips on the core suite (``uv run --extra dev pytest``) and runs when the
``train`` extra is installed.  The my_llm imports below stay safe either way
because both modules defer their heavy imports to call time.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from my_llm.mixing import load_mixed_dataset, validate_eval_source, validate_sources
from my_llm.posttrain import limit_dataset, load_eval_dataset, load_split, load_train_dataset

pytest.importorskip("datasets")

pytestmark = pytest.mark.training

REPO_ROOT = Path(__file__).resolve().parents[1]


def two_source_config(tmp_path: Path) -> dict:
    """Two 60-row JSONL sources with distinguishable rows, weighted 3:1."""

    for tag in ("a", "b"):
        lines = [json.dumps({"tag": tag, "idx": index}) for index in range(60)]
        (tmp_path / f"{tag}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "sources": [
            {"path": "json", "data_files": str(tmp_path / "a.jsonl"), "weight": 3.0},
            {"path": "json", "data_files": str(tmp_path / "b.jsonl"), "weight": 1.0},
        ]
    }


def test_weighted_proportions_exact_for_seed(tmp_path: Path) -> None:
    mixed = load_mixed_dataset(two_source_config(tmp_path), seed=42)
    # Empirical pin for datasets' seeded interleave: 181/60 realizes the 3:1
    # weights (probabilities 0.75/0.25) with the b source as the stopping bound.
    assert Counter(mixed["tag"]) == {"a": 181, "b": 60}
    # all_exhausted guarantees every example of every source appears at least once.
    assert len({row["idx"] for row in mixed if row["tag"] == "a"}) == 60
    assert len({row["idx"] for row in mixed if row["tag"] == "b"}) == 60


def test_same_seed_reproduces_identical_sequence(tmp_path: Path) -> None:
    config = two_source_config(tmp_path)
    first = load_mixed_dataset(config, seed=7)
    second = load_mixed_dataset(config, seed=7)
    assert list(first) == list(second)


def test_unknown_source_key_is_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_sources({"sources": [{"path": "json", "wieght": 2.0}]})
    assert "wieght" in str(excinfo.value)
    assert "allowed keys" in str(excinfo.value)


def test_non_positive_weight_is_rejected() -> None:
    for weight in (0, -1.5):
        with pytest.raises(ValueError, match="greater than 0"):
            validate_sources({"sources": [{"path": "json", "weight": weight}]})


def test_sources_conflicting_with_top_level_path_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_sources({"path": "json", "sources": [{"path": "json"}]})


def test_empty_sources_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        validate_sources({"sources": []})


def test_columns_pin_what_training_sees(tmp_path: Path) -> None:
    # A stray `prompt` column next to `messages` silently switches TRL to
    # prompt/completion mode; `columns` lets configs pin the visible schema.
    rows = [
        json.dumps({"messages": [{"role": "user", "content": "hi"}], "prompt": "hi", "id": i})
        for i in range(4)
    ]
    source = tmp_path / "chat.jsonl"
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    dataset_config = {"path": "json", "data_files": str(source), "columns": ["messages"]}
    dataset = load_split(dataset_config, "train")
    assert dataset.column_names == ["messages"]
    assert len(dataset) == 4


def test_columns_apply_to_every_source(tmp_path: Path) -> None:
    # Heterogeneous corpora (stray columns in one source only) can interleave
    # only if the top-level `columns` pin converges them on one schema.
    wide = [
        json.dumps({"messages": [{"role": "user", "content": "hi"}], "prompt": "hi", "id": i})
        for i in range(6)
    ]
    narrow = [json.dumps({"messages": [{"role": "user", "content": "yo"}]}) for _ in range(6)]
    (tmp_path / "wide.jsonl").write_text("\n".join(wide) + "\n", encoding="utf-8")
    (tmp_path / "narrow.jsonl").write_text("\n".join(narrow) + "\n", encoding="utf-8")
    mixed = load_mixed_dataset(
        {
            "sources": [
                {"path": "json", "data_files": str(tmp_path / "wide.jsonl")},
                {"path": "json", "data_files": str(tmp_path / "narrow.jsonl")},
            ],
            "columns": ["messages"],
        },
        seed=42,
    )
    assert mixed.column_names == ["messages"]
    assert len(mixed) >= 12


def test_struct_field_order_is_aligned_across_sources(tmp_path: Path) -> None:
    # Arrow bakes struct field order into the schema; two JSONL files with the
    # same logical columns must still interleave.
    ordered = [json.dumps({"messages": [{"role": "user", "content": "hi"}]}) for _ in range(4)]
    swapped = ['{"messages": [{"content": "yo", "role": "user"}]}' for _ in range(4)]
    (tmp_path / "ordered.jsonl").write_text("\n".join(ordered) + "\n", encoding="utf-8")
    (tmp_path / "swapped.jsonl").write_text("\n".join(swapped) + "\n", encoding="utf-8")
    mixed = load_mixed_dataset(
        {
            "sources": [
                {"path": "json", "data_files": str(tmp_path / "ordered.jsonl")},
                {"path": "json", "data_files": str(tmp_path / "swapped.jsonl")},
            ]
        },
        seed=42,
    )
    assert len(mixed) >= 8
    assert {row["messages"][0]["content"] for row in mixed} == {"hi", "yo"}


def test_eval_source_resolves_holdout_next_to_sources(tmp_path: Path) -> None:
    config = two_source_config(tmp_path)
    holdout = [json.dumps({"tag": "eval", "idx": i, "extra": i}) for i in range(5)]
    (tmp_path / "eval.jsonl").write_text("\n".join(holdout) + "\n", encoding="utf-8")
    config["eval_source"] = {
        "path": "json",
        "data_files": str(tmp_path / "eval.jsonl"),
        "split": "train",
    }
    config["columns"] = ["tag", "idx"]
    eval_dataset = load_eval_dataset(config)
    assert len(eval_dataset) == 5
    # The top-level columns pin covers the eval side too.
    assert eval_dataset.column_names == ["tag", "idx"]


def test_eval_split_next_to_sources_is_rejected(tmp_path: Path) -> None:
    config = two_source_config(tmp_path)
    config["eval_split"] = "test"
    with pytest.raises(ValueError, match="eval_source"):
        load_eval_dataset(config)


def test_eval_source_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="allowed keys") as excinfo:
        validate_eval_source({"path": "json", "weight": 1.0})
    assert "weight" in str(excinfo.value)


def test_eval_split_without_sources_keeps_loading(tmp_path: Path) -> None:
    rows = [json.dumps({"tag": "solo", "idx": i}) for i in range(3)]
    (tmp_path / "solo.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    config = {"path": "json", "data_files": str(tmp_path / "solo.jsonl"), "eval_split": "train"}
    assert len(load_eval_dataset(config)) == 3
    assert load_eval_dataset({"path": "json"}) is None


def test_single_source_config_loads_identical_examples() -> None:
    dataset_config = {
        "path": "json",
        "data_files": str(REPO_ROOT / "sample_data" / "sft.jsonl"),
        "split": "train",
        "max_samples": 8,
    }
    # The exact recipe run() used before the sources hook existed.
    baseline = limit_dataset(load_split(dataset_config, "train"), 8, 42)
    routed = limit_dataset(load_train_dataset(dataset_config, 42), 8, 42)
    assert list(routed) == list(baseline)
    assert len(routed) == 8
