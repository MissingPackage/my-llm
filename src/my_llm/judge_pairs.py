"""Turn a verdicts file over generated pairs into constitutional DPO preferences.

Manual mode only: verdicts are a human act, one row per pair, filled by hand
(the API judge is decision C3 and stays unimplemented until ruled).  Two
integrity rules shape everything here.  First, verdicts are *total* in the
benchmark_freeze sense: every generated pair gets either a preference or an
explicit discard — a missing verdict is a speaking error, never a silent skip.
Second, every preference records the id of the constitution principle that
decided it, validated against the principles actually present in
docs/CONSTITUTION.md at run time: the a recorded decision can amend the set, so the valid
ids are parsed from the file, never hardcoded.

The output rows use TRL's conversational preference format — ``prompt`` is a
one-message user turn, ``chosen``/``rejected`` are one-message assistant turns
— because the repo's DPO branch feeds the file to DPOTrainer untouched, the
local reference (sample_data/preferences.jsonl, consumed by dpo-smoke.yaml) is
conversational, and this format makes DPOTrainer re-apply at training time the
same chat template llm-genpairs used at generation time.  The standard string
format would instead concatenate raw text without the chat markup the
completions were conditioned on.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from my_llm.genpairs import write_records

# · is the middle dot in the constitution's "## P<n> · Title" headings.
PRINCIPLE_HEADING = re.compile(r"^## (P\d+) ·", re.MULTILINE)

PAIR_FIELDS = frozenset({"id", "prompt", "principle", "completions", "meta"})
PREFERENCE_FIELDS = frozenset({"id", "chosen", "rejected", "principle", "note", "unreviewed"})
DISCARD_FIELDS = frozenset({"id", "verdict", "note", "unreviewed"})
DISCARD = "discard"


@dataclass(frozen=True)
class Pair:
    """One validated candidate group from the llm-genpairs output file."""

    id: str
    prompt: str
    completions: tuple[str, ...]


def load_principles(path: str | Path) -> frozenset[str]:
    """Parse the principle ids the constitution currently defines.

    The single source of truth is the file itself — a recorded decision (C2) can amend
    the set, so a hardcoded list would validate against a dead constitution.
    """

    path = Path(path)
    found = PRINCIPLE_HEADING.findall(path.read_text(encoding="utf-8"))
    duplicates = sorted({principle for principle in found if found.count(principle) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate principle headings {duplicates}")
    if not found:
        raise ValueError(
            f"{path}: no principle headings ('## P<n> ·') found — wrong file or "
            "broken constitution"
        )
    return frozenset(found)


def load_pairs(path: str | Path) -> list[Pair]:
    """Read the llm-genpairs output and reject rows the judge cannot act on.

    The generation-time ``principle`` (a target hint) and ``meta`` are accepted
    but not consumed: the verdict's principle is the authoritative one, because
    the judge decides which principle actually discriminated the pair.
    """

    path = Path(path)
    pairs: list[Pair] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            unknown = set(row) - PAIR_FIELDS
            if unknown:
                raise ValueError(f"{path}:{line_number}: unknown pair fields {sorted(unknown)}")
            pair_id = row.get("id")
            if not isinstance(pair_id, str) or not pair_id.strip():
                raise ValueError(f"{path}:{line_number}: pair row without an id")
            if pair_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate pair id {pair_id!r}")
            seen.add(pair_id)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number}: pair {pair_id!r} without a prompt")
            completions = row.get("completions")
            if (
                not isinstance(completions, list)
                or len(completions) < 2
                or not all(isinstance(item, str) for item in completions)
            ):
                raise ValueError(
                    f"{path}:{line_number}: pair {pair_id!r} needs a list of >= 2 string "
                    "completions"
                )
            pairs.append(Pair(id=pair_id, prompt=prompt, completions=tuple(completions)))
    if not pairs:
        raise ValueError(f"{path}: no pair rows — an empty batch is an error, not a no-op")
    return pairs


def load_verdicts(path: str | Path, principles: frozenset[str]) -> dict[str, dict[str, object]]:
    """Read the verdicts file and reject rows judging cannot act on.

    Shape-level rules live here with line numbers; the cross-checks against the
    pairs file (totality, index ranges) live in :func:`build_preferences`.
    """

    path = Path(path)
    verdicts: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            where = f"{path}:{line_number}"
            pair_id = row.get("id")
            if not isinstance(pair_id, str) or not pair_id.strip():
                raise ValueError(f"{where}: verdict row without an id")
            if pair_id in verdicts:
                raise ValueError(f"{where}: duplicate verdict for {pair_id!r}")
            note = row.get("note")
            if note is not None and not isinstance(note, str):
                raise ValueError(f"{where}: note of {pair_id!r} must be a string")
            unreviewed = row.get("unreviewed")
            if unreviewed is not None and not isinstance(unreviewed, bool):
                raise ValueError(f"{where}: unreviewed of {pair_id!r} must be a boolean")
            if "verdict" in row:
                if row["verdict"] != DISCARD:
                    raise ValueError(
                        f"{where}: unknown verdict {row['verdict']!r} — a pair is either judged "
                        f"(chosen/rejected) or explicitly discarded ({DISCARD!r})"
                    )
                unknown = set(row) - DISCARD_FIELDS
                if unknown:
                    raise ValueError(f"{where}: unknown discard fields {sorted(unknown)}")
                if not isinstance(note, str) or not note.strip():
                    raise ValueError(
                        f"{where}: discard of {pair_id!r} needs a note — an unmotivated discard "
                        "is indistinguishable from an accidental omission"
                    )
            else:
                unknown = set(row) - PREFERENCE_FIELDS
                if unknown:
                    raise ValueError(f"{where}: unknown verdict fields {sorted(unknown)}")
                for field in ("chosen", "rejected"):
                    value = row.get(field)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError(
                            f"{where}: {field} of {pair_id!r} must be a completion index >= 0"
                        )
                if row["chosen"] == row["rejected"]:
                    raise ValueError(
                        f"{where}: {pair_id!r} has chosen == rejected — a preference needs "
                        "two different completions"
                    )
                principle = row.get("principle")
                if not isinstance(principle, str) or not principle.strip():
                    raise ValueError(
                        f"{where}: {pair_id!r} has no principle — a preference without a "
                        "principle is invalid by construction (docs/CONSTITUTION.md)"
                    )
                if principle not in principles:
                    ordered = sorted(principles, key=lambda item: int(item[1:]))
                    raise ValueError(
                        f"{where}: unknown principle {principle!r} for {pair_id!r} — the "
                        f"constitution defines {ordered}"
                    )
            verdicts[pair_id] = row
    return verdicts


def build_preferences(
    pairs: list[Pair], verdicts: dict[str, dict[str, object]]
) -> tuple[list[dict[str, object]], int]:
    """Apply total verdicts to the pairs; return (preference rows, discarded count).

    ``unreviewed`` is always present in the output (False when unset) so phase-4
    smoke material stays distinguishable from human-labelled preferences under one
    schema; ``id`` is kept so every preference traces back to its source pair.
    """

    pair_ids = {pair.id for pair in pairs}
    unknown = sorted(set(verdicts) - pair_ids)
    if unknown:
        raise ValueError(f"Verdicts for unknown pair ids: {unknown[:10]}")
    missing = sorted(pair_ids - set(verdicts))
    if missing:
        raise ValueError(
            f"{len(missing)} pairs without a verdict, e.g. {missing[:10]} — every pair needs "
            f"a preference or an explicit {DISCARD!r}"
        )

    preferences: list[dict[str, object]] = []
    discarded = 0
    for pair in pairs:
        row = verdicts[pair.id]
        if row.get("verdict") == DISCARD:
            discarded += 1
            continue
        chosen, rejected = int(row["chosen"]), int(row["rejected"])  # type: ignore[arg-type]
        limit = len(pair.completions)
        for field, index in (("chosen", chosen), ("rejected", rejected)):
            if index >= limit:
                raise ValueError(
                    f"pair {pair.id!r}: {field} index {index} out of range — the pair has "
                    f"{limit} completions"
                )
        preferences.append(
            {
                "id": pair.id,
                "prompt": [{"role": "user", "content": pair.prompt}],
                "chosen": [{"role": "assistant", "content": pair.completions[chosen]}],
                "rejected": [{"role": "assistant", "content": pair.completions[rejected]}],
                "principle": row["principle"],
                "unreviewed": bool(row.get("unreviewed", False)),
            }
        )
    return preferences, discarded


def judge(
    pairs_path: str | Path,
    verdicts_path: str | Path,
    output_path: str | Path,
    constitution_path: str | Path,
) -> dict[str, int]:
    """Run the manual judging pipeline and write the preferences file."""

    principles = load_principles(constitution_path)
    pairs = load_pairs(pairs_path)
    verdicts = load_verdicts(verdicts_path, principles)
    preferences, discarded = build_preferences(pairs, verdicts)
    write_records(preferences, Path(output_path))
    return {
        "pairs": len(pairs),
        "preferences": len(preferences),
        "discarded": discarded,
        "unreviewed": sum(1 for row in preferences if row["unreviewed"]),
    }


def main() -> None:
    """CLI entry point for turning manual verdicts into DPO preferences."""

    parser = argparse.ArgumentParser(
        description="Turn a reviewer verdicts file over generated pairs into DPO preference rows."
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--constitution", type=Path, default=Path("docs/CONSTITUTION.md"))
    args = parser.parse_args()
    summary = judge(args.pairs, args.verdicts, args.output, args.constitution)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary['preferences']} preferences to {args.output}")


if __name__ == "__main__":
    main()
