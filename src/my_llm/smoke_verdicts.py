# MECHANICAL BY DESIGN (decision C4): the verdicts produced here do NOT apply the
# constitution and carry no claim of discriminance.  They exist only to exercise
# the genpairs -> judge-pairs -> DPOTrainer pipeline on smoke material.
"""Mechanical unreviewed smoke verdicts for the phase-4 constitutional DPO chain.

The smoke model's completions are word salad (decision C4), so no real judgement
is possible: this module replaces the reviewer's verdicts file with a deterministic
rule purely to test the pipeline.  Chosen = shortest completion (index breaks
ties), rejected = longest, principle = the prompt's declared target, and every
verdict carries ``unreviewed: true`` plus a note declaring its mechanical origin.
Pairs that even a mechanical rule cannot order — an empty completion or
byte-identical candidates — become explicit discards, never silent skips, so
judge-pairs totality holds by construction.  Real verdicts are a human act
and arrive only in phase 5; this module is deliberately absent from
``pyproject.toml`` scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from my_llm.genpairs import write_records

DISCARD_NOTE = (
    "verdetto sintetico C4 (meccanico, unreviewed): {reason} — coppia non ordinabile "
    "nemmeno meccanicamente, nessuna applicazione della costituzione"
)
PREFERENCE_NOTE = (
    "verdetto sintetico C4 (meccanico, unreviewed): chosen = completion più corta, "
    "principio = target del prompt, nessuna applicazione reale della costituzione"
)


def synthetic_verdicts(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Derive one mechanical verdict per llm-genpairs record.

    Deterministic and total: the same records always produce the same verdicts,
    and every record yields either a preference or an explicit discard.
    """

    verdicts: list[dict[str, object]] = []
    for record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("pair record without an id")
        principle = record.get("principle")
        if not isinstance(principle, str) or not principle.strip():
            raise ValueError(
                f"pair {record_id!r} has no target principle — smoke prompts must declare "
                "one, the mechanical rule cannot invent it"
            )
        completions = record.get("completions")
        if (
            not isinstance(completions, list)
            or len(completions) < 2
            or not all(isinstance(item, str) for item in completions)
        ):
            raise ValueError(f"pair {record_id!r} needs a list of >= 2 string completions")
        if any(not item.strip() for item in completions):
            verdicts.append(_discard(record_id, "una completion è vuota"))
            continue
        lengths = [(len(text), index) for index, text in enumerate(completions)]
        chosen, rejected = min(lengths)[1], max(lengths)[1]
        if completions[chosen] == completions[rejected]:
            verdicts.append(_discard(record_id, "le completions sono identiche"))
            continue
        verdicts.append(
            {
                "id": record_id,
                "chosen": chosen,
                "rejected": rejected,
                "principle": principle,
                "note": PREFERENCE_NOTE,
                "unreviewed": True,
            }
        )
    return verdicts


def _discard(record_id: str, reason: str) -> dict[str, object]:
    """Build an explicit discard verdict; the note keeps it distinguishable from omission."""

    return {
        "id": record_id,
        "verdict": "discard",
        "note": DISCARD_NOTE.format(reason=reason),
        "unreviewed": True,
    }


def main() -> None:
    """Turn an llm-genpairs output file into a mechanical unreviewed verdicts file."""

    parser = argparse.ArgumentParser(
        description="Generate mechanical unreviewed smoke verdicts (decision C4) from llm-genpairs "
        "output. Smoke material only: no constitution is applied."
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.pairs.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    verdicts = synthetic_verdicts(records)
    write_records(verdicts, args.output)
    discards = sum(1 for row in verdicts if row.get("verdict") == "discard")
    print(f"Wrote {len(verdicts)} verdicts ({discards} discards) to {args.output}")


if __name__ == "__main__":
    main()
