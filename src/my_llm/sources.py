"""Streaming dataset loading and reproducible weighted source mixing."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from my_llm.config import SourceSpec


def iter_records(
    sources: list[SourceSpec], *, seed: int, shuffle_buffer: int
) -> Iterator[tuple[dict[str, Any], SourceSpec]]:
    """Stream one or more sources, preserving the source attached to each record.

    Hugging Face IterableDatasets avoid downloading a corpus before processing it.
    The finite shuffle buffer is an explicit approximation: larger buffers improve
    mixing but consume more host RAM.
    """
    try:
        from datasets import interleave_datasets, load_dataset
    except ImportError as exc:  # pragma: no cover - exercised by the installed CLI
        raise RuntimeError("Install training dependencies: uv sync --extra train") from exc

    streams = []
    source_by_index: dict[int, SourceSpec] = {}
    for index, source in enumerate(sources):
        kwargs: dict[str, Any] = {
            "path": source.path,
            "split": source.split,
            "streaming": True,
        }
        if source.name is not None:
            kwargs["name"] = source.name
        if source.revision is not None:
            kwargs["revision"] = source.revision
        if source.data_files is not None:
            kwargs["data_files"] = source.data_files
        stream = load_dataset(**kwargs)
        stream = stream.shuffle(seed=seed + index, buffer_size=shuffle_buffer)
        # Carry a tiny source ID through interleaving so each record can use its own
        # text column and provenance metadata after datasets are mixed.
        stream = stream.map(lambda row, i=index: {**row, "__source_index": i})
        streams.append(stream)
        source_by_index[index] = source

    if len(streams) == 1:
        combined = streams[0]
    else:
        total_weight = sum(source.weight for source in sources)
        probabilities = [source.weight / total_weight for source in sources]
        # ``all_exhausted`` prevents a small bilingual source from ending the entire
        # mixture early; probabilities implement the declared sampling weights.
        combined = interleave_datasets(
            streams,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy="all_exhausted",
        )

    for record in combined:
        index = int(record.pop("__source_index"))
        yield record, source_by_index[index]


def iter_texts(
    sources: list[SourceSpec], *, seed: int, shuffle_buffer: int, max_documents: int | None
) -> Iterator[str]:
    """Yield non-empty text strings for tokenizer training."""

    for index, (record, source) in enumerate(
        iter_records(sources, seed=seed, shuffle_buffer=shuffle_buffer)
    ):
        if max_documents is not None and index >= max_documents:
            return
        text = record.get(source.text_column)
        if isinstance(text, str) and text.strip():
            yield text
