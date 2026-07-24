"""Format, determinism and biography-blacklist tests for the persona dataset.

The blacklist runs on the committed file AND on a fresh generation because the
committed artifact and the generator can drift independently: a hand edit could
poison the file, a code change could poison future regenerations. Both paths
must stay clean (docs/DATA_GOVERNANCE.md §6).
"""

import json
import re
from pathlib import Path

import pytest

from my_llm.persona import (
    CANDIDATES,
    DEFAULT_COUNT,
    DEFAULT_SEED,
    LEAD_CANDIDATE,
    generate_records,
)

ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data" / "identity" / "persona-v1-draft.jsonl"
MANIFEST = ROOT / "data" / "identity" / "persona-v1-draft.manifest.json"

# Minimal biographical blacklist: the persona must refer to its creator only as
# "my developer" / "il mio sviluppatore" — never a real name, employer, or contact
# address. The email regex below is the always-on guard; maintainers can extend
# FORBIDDEN_SUBSTRINGS with any additional identifiers they want kept out of
# identity data. Case-insensitive on purpose (docs/DATA_GOVERNANCE.md §6).
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = ()
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

VALID_ROLES = {"system", "user", "assistant"}


def load_committed() -> list[dict]:
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture(scope="module")
def committed_records() -> list[dict]:
    return load_committed()


@pytest.fixture(scope="module")
def fresh_records() -> list[dict]:
    return generate_records(LEAD_CANDIDATE, seed=DEFAULT_SEED, count=DEFAULT_COUNT)


def assert_valid_record(record: dict) -> None:
    # Only the `messages` key: interleave_datasets needs features identical to
    # sample_data/sft.jsonl, so per-row metadata is banned (it lives in the manifest).
    assert set(record) == {"messages"}
    messages = record["messages"]
    assert isinstance(messages, list) and len(messages) >= 2
    for message in messages:
        assert set(message) == {"role", "content"}
        assert message["role"] in VALID_ROLES
        assert isinstance(message["content"], str)
        assert message["content"].strip()
    conversation = messages[1:] if messages[0]["role"] == "system" else messages
    assert len(conversation) >= 2 and len(conversation) % 2 == 0
    for index, message in enumerate(conversation):
        assert message["role"] == ("user" if index % 2 == 0 else "assistant")


def assert_no_biography(records: list[dict]) -> None:
    for record in records:
        blob = json.dumps(record, ensure_ascii=False).lower()
        for term in FORBIDDEN_SUBSTRINGS:
            assert term not in blob, f"forbidden term {term!r} found in: {blob[:200]}"
        assert not EMAIL_RE.search(blob), f"email address found in: {blob[:200]}"


def test_committed_dataset_is_large_enough(committed_records: list[dict]) -> None:
    assert len(committed_records) >= 300


def test_committed_dataset_format(committed_records: list[dict]) -> None:
    for record in committed_records:
        assert_valid_record(record)


def test_fresh_generation_format(fresh_records: list[dict]) -> None:
    for record in fresh_records:
        assert_valid_record(record)


def test_manifest_marks_draft_and_matches_generator_inputs(
    committed_records: list[dict],
) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["draft"] is True
    assert manifest["candidate"] == LEAD_CANDIDATE.name
    assert manifest["candidate"] in CANDIDATES
    assert manifest["generator_seed"] == DEFAULT_SEED
    assert manifest["count"] == len(committed_records)


def test_same_seed_reproduces_same_dataset(fresh_records: list[dict]) -> None:
    again = generate_records(LEAD_CANDIDATE, seed=DEFAULT_SEED, count=DEFAULT_COUNT)
    assert again == fresh_records


def test_different_seed_changes_dataset(fresh_records: list[dict]) -> None:
    other = generate_records(LEAD_CANDIDATE, seed=DEFAULT_SEED + 1, count=DEFAULT_COUNT)
    assert other != fresh_records


def test_committed_file_matches_fresh_generation(
    committed_records: list[dict], fresh_records: list[dict]
) -> None:
    # Ties the artifact to the code: any hand edit or silent generator drift fails here.
    assert committed_records == fresh_records


def test_no_biographical_facts_in_committed_file(committed_records: list[dict]) -> None:
    assert_no_biography(committed_records)


def test_no_biographical_facts_in_fresh_generation(fresh_records: list[dict]) -> None:
    assert_no_biography(fresh_records)


def test_no_duplicate_conversations(committed_records: list[dict]) -> None:
    keys = {
        tuple((m["role"], m["content"]) for m in record["messages"] if m["role"] != "system")
        for record in committed_records
    }
    assert len(keys) == len(committed_records)


def test_language_mix_is_roughly_bilingual(committed_records: list[dict]) -> None:
    # Coarse proxy for the ~60/40 IT/EN target: Italian first turns should be a
    # clear majority but far from a monolingual dataset. Loose bounds by design —
    # this guards the mix, not the exact ratio.
    italian_markers = re.compile(
        r"\b(chi|cosa|cos'è|sei|presenti|presentati|chiami|chiamarti|qual|quale|quante|quanto"
        r"|nome|vieni|spiegami|definizione|scrivimi|fammi|riempi|rispondi|rispondimi|dammi"
        r"|che|esiste|puoi|posso|apri|parli|ricordi|ricorderai)\b",
        re.I,
    )
    first_turns = [
        next(m["content"] for m in record["messages"] if m["role"] == "user")
        for record in committed_records
    ]
    italian = sum(1 for turn in first_turns if italian_markers.search(turn))
    share = italian / len(first_turns)
    assert 0.40 < share < 0.85
