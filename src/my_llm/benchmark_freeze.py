"""Promote reviewer-approved draft items into a frozen benchmark release.

Approval is a human act and happens in a verdicts file (recorded decision A4): one
row per draft item, filled by hand.  This tool makes that act *total* — a draft
item without an explicit verdict is an error, never a silent omission — and turns
the outcome into an immutable release: ``v1.jsonl`` plus its SHA-256 fingerprint,
which the test suite then defends against drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from my_llm.benchmark import (
    V1_MINIMUM_CALIBRATION,
    V1_MINIMUM_ITEMS,
    BenchmarkItem,
    load_benchmark,
)

VERDICT_VALUES = frozenset({"approve", "reject", "edit"})
# Fields the reviewer may replace in an "edit" verdict; anything else stays as drafted.
EDITABLE_FIELDS = frozenset({"prompt", "reference", "rubric", "domain", "lang"})
UNDECIDED = "_"


def load_verdicts(path: str | Path) -> dict[str, dict[str, object]]:
    """Read the verdicts file and reject rows the promotion cannot act on."""

    path = Path(path)
    verdicts: dict[str, dict[str, object]] = {}
    undecided: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row.get("id", ""))
            if not item_id:
                raise ValueError(f"{path}:{line_number}: verdict row without an id")
            if item_id in verdicts:
                raise ValueError(f"{path}:{line_number}: duplicate verdict for {item_id!r}")
            verdict = str(row.get("verdict", UNDECIDED))
            if verdict == UNDECIDED:
                undecided.append(item_id)
                continue
            if verdict not in VERDICT_VALUES:
                raise ValueError(f"{path}:{line_number}: unknown verdict {verdict!r}")
            unknown_fields = set(row) - {"id", "verdict", "note"} - EDITABLE_FIELDS
            if unknown_fields:
                raise ValueError(
                    f"{path}:{line_number}: unknown verdict fields {sorted(unknown_fields)}"
                )
            verdicts[item_id] = row
    if undecided:
        raise ValueError(
            f"{path}: {len(undecided)} items still undecided ('_'), e.g. {undecided[:10]}"
        )
    return verdicts


def promote(
    drafts_path: str | Path,
    verdicts_path: str | Path,
    output_path: str | Path,
    *,
    minimum_items: int = V1_MINIMUM_ITEMS,
    minimum_calibration: int = V1_MINIMUM_CALIBRATION,
) -> dict[str, int]:
    """Apply verdicts to drafts and write a frozen, fingerprinted release file."""

    output_path = Path(output_path)
    if output_path.exists():
        # A frozen release never changes in place; the next release is a new file.
        raise FileExistsError(
            f"{output_path} already exists — releases are frozen, write a v2 instead"
        )
    drafts = load_benchmark(drafts_path)
    verdicts = load_verdicts(verdicts_path)

    draft_ids = {item.id for item in drafts}
    unknown = sorted(set(verdicts) - draft_ids)
    if unknown:
        raise ValueError(f"Verdicts for unknown item ids: {unknown[:10]}")
    missing = sorted(draft_ids - set(verdicts))
    if missing:
        raise ValueError(
            f"{len(missing)} draft items without a verdict, e.g. {missing[:10]}"
        )

    approved: list[BenchmarkItem] = []
    rejected = 0
    edited = 0
    for item in drafts:
        row = verdicts[item.id]
        if row["verdict"] == "reject":
            rejected += 1
            continue
        payload = item.model_dump()
        if row["verdict"] == "edit":
            edited += 1
            replacements = {key: row[key] for key in EDITABLE_FIELDS if key in row}
            payload.update(replacements)
        payload["approved"] = True
        # Re-validate: an edit may not break gradability rules the schema enforces.
        approved.append(BenchmarkItem.model_validate(payload))

    approved.sort(key=lambda item: item.id)
    temporary = output_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(item.model_dump(), ensure_ascii=False) + "\n" for item in approved),
        encoding="utf-8",
    )
    # Validate the file we are about to publish exactly as consumers will load it.
    load_benchmark(
        temporary,
        require_approved=True,
        minimum_items=minimum_items,
        minimum_calibration=minimum_calibration,
    )
    temporary.replace(output_path)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return {
        "drafts": len(drafts),
        "approved": len(approved),
        "edited": edited,
        "rejected": rejected,
    }


def main() -> None:
    """CLI entry point for the verdict-driven release freeze."""

    parser = argparse.ArgumentParser(description="Freeze reviewer-approved benchmark items.")
    parser.add_argument("--drafts", default="benchmarks/personal/drafts/batch-001.jsonl")
    parser.add_argument(
        "--verdicts", default="benchmarks/personal/drafts/verdicts-batch-001.jsonl"
    )
    parser.add_argument("--output", default="benchmarks/personal/v1.jsonl")
    args = parser.parse_args()
    summary = promote(args.drafts, args.verdicts, args.output)
    print(json.dumps(summary, indent=2))
    print(f"Frozen release written to {args.output} (+ .sha256)")


if __name__ == "__main__":
    main()
