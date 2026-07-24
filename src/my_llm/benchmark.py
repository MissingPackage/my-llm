"""Schema and loader for the personal held-out benchmark.

The benchmark is the per-stage quality gate of the personalization goals, so its
integrity rules live in code rather than in convention: items are validated at the
boundary, only reviewer-approved items may enter a frozen release, and the go/no-go gate
uses exclusively deterministic verification.  ``llm_rubric`` items exist for the
qualities exact matching cannot measure (persona, style); they are reported as a
separate trend and never enter the gate (recorded decision A5, 2026-07-16).
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from my_llm.config import StrictModel
from my_llm.reasoning import canonical_answer

# Only these verification modes may contribute to the go/no-go gate.  llm_rubric is
# deliberately excluded: a judged score is a trend signal, not a release criterion.
GATE_VERIFICATIONS = frozenset({"exact_numeric", "exact_string", "regex"})

# Release thresholds for a frozen v1 file (goal contract DONE-WHEN, ruling A1).
V1_MINIMUM_ITEMS = 100
V1_MINIMUM_CALIBRATION = 5


class BenchmarkItem(StrictModel):
    """One benchmark item; the schema rejects anything the scorer cannot grade."""

    id: str = Field(min_length=1)
    lang: Literal["it", "en"]
    domain: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    verification: Literal["exact_numeric", "exact_string", "regex", "llm_rubric"]
    rubric: str | None = None
    # Calibration items are trivially solvable on purpose: they expose a broken
    # harness or an always-fail grader instead of measuring the model.
    calibration: bool = False
    # Only a human reviewer flips this to True; the loader enforces it for release files.
    approved: bool = False

    @model_validator(mode="after")
    def validate_verification_payload(self) -> BenchmarkItem:
        """Check that the reference is actually gradable by the declared mode."""

        if self.verification == "llm_rubric":
            if not (self.rubric or "").strip():
                raise ValueError("llm_rubric items require a non-empty rubric")
            return self
        if self.rubric is not None:
            raise ValueError("rubric is only meaningful for llm_rubric items")
        if self.verification == "regex":
            try:
                re.compile(self.reference)
            except re.error as exc:
                raise ValueError(f"reference is not a valid regex: {exc}") from exc
        if self.verification == "exact_numeric" and not isinstance(
            canonical_answer(self.reference), Fraction
        ):
            raise ValueError("exact_numeric reference must canonicalize to a number")
        return self


def load_benchmark(
    path: str | Path,
    *,
    require_approved: bool = False,
    minimum_items: int = 0,
    minimum_calibration: int = 0,
) -> list[BenchmarkItem]:
    """Load and validate one JSONL benchmark file.

    Draft batches load with the permissive defaults.  Release files load through
    :func:`load_v1`, which turns the GOAL thresholds and the approved-only rule
    into hard errors instead of review checklist items.
    """

    path = Path(path)
    items: list[BenchmarkItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            item = BenchmarkItem.model_validate(payload)
            if item.id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate item id {item.id!r}")
            seen.add(item.id)
            items.append(item)

    if len(items) < minimum_items:
        raise ValueError(f"{path}: {len(items)} items, minimum is {minimum_items}")
    calibration_count = sum(1 for item in items if item.calibration)
    if calibration_count < minimum_calibration:
        raise ValueError(
            f"{path}: {calibration_count} calibration items, minimum is {minimum_calibration}"
        )
    if require_approved:
        unapproved = [item.id for item in items if not item.approved]
        if unapproved:
            raise ValueError(f"{path}: unapproved items in release file: {unapproved[:5]}")
    return items


def load_v1(path: str | Path) -> list[BenchmarkItem]:
    """Load a frozen release file with every integrity rule enforced."""

    return load_benchmark(
        path,
        require_approved=True,
        minimum_items=V1_MINIMUM_ITEMS,
        minimum_calibration=V1_MINIMUM_CALIBRATION,
    )
