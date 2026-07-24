"""Weighted multi-source mixing for the SFT stage.

``dataset.sources`` lets one post-training run draw from several corpora (for
example a large general SFT set plus a small identity set) with explicit
relative weights.  Validation happens at the boundary, mirroring
:mod:`my_llm.config`: a misspelled key in a source mapping fails immediately
with the list of accepted keys instead of being silently ignored mid-run.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, ValidationError

from my_llm.config import StrictModel


class EvalSourceSpec(StrictModel):
    """Held-out eval set of a mixed run: one plain, unweighted source.

    ``dataset.sources`` forbids the top-level ``path`` that ``eval_split``
    loading relies on, so mixed runs declare their held-out set as an explicit
    source spec instead of inheriting it from the train-side selectors.
    """

    path: str
    name: str | None = None
    data_files: str | list[str] | dict[str, str | list[str]] | None = None
    revision: str | None = None
    split: str = "test"


class SFTSourceSpec(StrictModel):
    """One weighted source of an SFT mixture.

    The post-training twin of :class:`my_llm.config.SourceSpec`: same loader
    selectors, same strict-keys philosophy.  It drops ``text_column`` (SFT rows
    already carry structured columns such as ``messages``) and adds
    ``max_samples`` so each source can be capped to a deterministic subset
    before mixing.
    """

    path: str
    name: str | None = None
    data_files: str | list[str] | dict[str, str | list[str]] | None = None
    revision: str | None = None
    split: str = "train"
    weight: float = Field(default=1.0, gt=0)
    max_samples: int | None = Field(default=None, ge=1)


# Top-level keys that select a single dataset; with ``sources`` they would be
# silently ignored, so their presence is treated as a config error instead.
_SINGLE_SOURCE_KEYS = ("path", "name", "data_files", "revision")


def validate_sources(dataset_config: dict[str, Any]) -> list[SFTSourceSpec]:
    """Validate ``dataset.sources`` entries, rejecting ambiguous or misspelled configs."""

    conflicting = [key for key in _SINGLE_SOURCE_KEYS if dataset_config.get(key) is not None]
    if conflicting:
        raise ValueError(
            f"dataset.sources cannot be combined with top-level {', '.join(conflicting)}; "
            "move every source into its own sources entry"
        )
    sources = dataset_config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("dataset.sources must be a non-empty list of source mappings")
    specs: list[SFTSourceSpec] = []
    for index, raw in enumerate(sources):
        try:
            specs.append(SFTSourceSpec.model_validate(raw))
        except ValidationError as exc:
            allowed = ", ".join(sorted(SFTSourceSpec.model_fields))
            raise ValueError(
                f"dataset.sources[{index}] is invalid (allowed keys: {allowed}): {exc}"
            ) from exc
    return specs


def validate_eval_source(raw: Any) -> EvalSourceSpec:
    """Validate ``dataset.eval_source``, failing with the accepted keys listed."""

    try:
        return EvalSourceSpec.model_validate(raw)
    except ValidationError as exc:
        allowed = ", ".join(sorted(EvalSourceSpec.model_fields))
        raise ValueError(
            f"dataset.eval_source is invalid (allowed keys: {allowed}): {exc}"
        ) from exc


def load_mixed_dataset(dataset_config: dict[str, Any], seed: int) -> Any:
    """Load every source and interleave them by normalized weight.

    Weights are relative sampling probabilities, not exact quotas: each output
    row is drawn from source ``i`` with probability ``weight_i / sum(weights)``,
    so realized counts fluctuate around the ratio but are fully reproducible for
    a given ``seed`` (which also fixes each per-source ``max_samples`` subset).

    Stopping strategy: ``all_exhausted``.  Sampling continues, restarting
    already-exhausted sources, until every source has been seen in full.  The
    ``first_exhausted`` default would stop once the most-sampled source ran
    out, silently discarding most of the other corpora.  For identity SFT the
    small persona set is precisely the source that must not be truncated, and
    oversampling it against the larger general corpus at the configured ratio
    is the intended behavior; every example of every source is guaranteed to
    appear at least once.
    """

    from datasets import interleave_datasets

    # Imported here to keep the module import edge one-way: posttrain imports
    # this module lazily inside its hook, so a top-level import back would risk
    # a cycle if that hook ever moved to module scope.
    from my_llm.posttrain import limit_dataset, load_split

    specs = validate_sources(dataset_config)
    columns = dataset_config.get("columns")
    parts = []
    for index, spec in enumerate(specs):
        source_config = spec.model_dump(exclude={"weight", "max_samples"})
        # Top-level `columns` applies to every source: heterogeneous corpora
        # (a hub set carrying stray columns next to a messages-only local set)
        # must converge on one schema before they can interleave.
        source_config["columns"] = columns
        part = load_split(source_config, spec.split)
        # Offset the seed per source so equally-sized sources do not select the
        # same row indices when both are capped.
        parts.append(limit_dataset(part, spec.max_samples, seed + index))
    features = parts[0].features
    # Arrow bakes struct field order into the schema, so sources with the same
    # logical columns can still disagree; align on the first source's features
    # (a real incompatibility keeps failing loudly inside cast).
    parts = [
        part if part.features == features else part.cast(features) for part in parts
    ]
    total_weight = sum(spec.weight for spec in specs)
    probabilities = [spec.weight / total_weight for spec in specs]
    return interleave_datasets(
        parts, probabilities=probabilities, seed=seed, stopping_strategy="all_exhausted"
    )
