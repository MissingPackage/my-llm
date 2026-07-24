"""Stream raw text, tokenize in batches and write compact train/validation shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from my_llm.binary import BinaryShardWriter
from my_llm.config import DataPrepConfig, load_typed
from my_llm.sources import iter_records


def tokenizer_fingerprint(tokenizer_path: Path) -> str:
    """Hash the tokenizer files that determine the text-to-ID mapping."""

    digest = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = tokenizer_path / name
        if path.exists():
            # Include the filename so concatenated byte streams cannot collide merely
            # because file boundaries differ.
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def is_validation_record(record: dict[str, Any], text: str, *, seed: int, fraction: float) -> bool:
    """Assign documents deterministically, independent of streaming order."""

    identity = record.get("id") or record.get("url") or text[:2048]
    payload = f"{seed}:{identity}".encode("utf-8", errors="replace")
    bucket = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / 2**64
    return bucket < fraction


def reached(current: int, target: int | None) -> bool:
    """Treat ``None`` as an unlimited target."""

    return target is not None and current >= target


def _flush_tokenization_batch(
    pending: list[tuple[str, BinaryShardWriter, int | None]],
    *,
    tokenizer: Any,
) -> int:
    """Tokenize several documents in Rust and append EOS-separated token streams."""

    if not pending:
        return 0
    encoded = tokenizer(
        [text for text, _, _ in pending],
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    written_documents = 0
    for token_ids, (_, writer, target) in zip(encoded, pending, strict=True):
        if reached(writer.total_tokens, target):
            continue
        # EOS is a document boundary.  Random windows may cross it, teaching the
        # decoder that one document ended instead of concatenating unrelated prose.
        token_ids.append(tokenizer.eos_token_id)
        writer.add(token_ids)
        written_documents += 1
    pending.clear()
    return written_documents


def prepare(config: DataPrepConfig) -> tuple[Path, Path]:
    """Create memory-mapped token shards and return train/validation manifests."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install training dependencies: uv sync --extra train") from exc

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token")
    if config.dtype == "uint16" and len(tokenizer) > 65_535:
        raise ValueError("Tokenizer vocabulary does not fit in uint16")

    metadata = {
        "tokenizer_path": str(config.tokenizer_path),
        "tokenizer_sha256": tokenizer_fingerprint(config.tokenizer_path),
        "eos_token_id": tokenizer.eos_token_id,
        "sources": [source.model_dump(mode="json") for source in config.sources],
        "seed": config.seed,
    }
    train_writer = BinaryShardWriter(
        config.output_dir,
        "train",
        dtype=config.dtype,
        shard_tokens=config.shard_tokens,
        metadata=metadata,
    )
    validation_writer = BinaryShardWriter(
        config.output_dir,
        "validation",
        dtype=config.dtype,
        shard_tokens=config.shard_tokens,
        metadata=metadata,
    )

    document_count = 0
    skipped_count = 0
    pending: list[tuple[str, BinaryShardWriter, int | None]] = []
    try:
        for record, source in iter_records(
            config.sources, seed=config.seed, shuffle_buffer=config.shuffle_buffer
        ):
            if (
                config.max_documents is not None
                and document_count + len(pending) >= config.max_documents
            ):
                break
            if reached(train_writer.total_tokens, config.target_train_tokens) and reached(
                validation_writer.total_tokens, config.target_validation_tokens
            ):
                break
            text = record.get(source.text_column)
            if not isinstance(text, str) or not text.strip():
                skipped_count += 1
                continue
            use_validation = is_validation_record(
                record, text, seed=config.seed, fraction=config.validation_fraction
            )
            writer = validation_writer if use_validation else train_writer
            target = (
                config.target_validation_tokens if use_validation else config.target_train_tokens
            )
            if reached(writer.total_tokens, target):
                continue
            pending.append((text, writer, target))
            if len(pending) >= config.tokenize_batch_size:
                document_count += _flush_tokenization_batch(pending, tokenizer=tokenizer)
                if document_count and document_count % 10_000 == 0:
                    print(
                        f"documents={document_count:,} "
                        f"train_tokens={train_writer.total_tokens:,} "
                        f"validation_tokens={validation_writer.total_tokens:,}"
                    )
    finally:
        # Preserve documents already read even if the stream naturally ends before
        # reaching a full tokenizer batch.
        document_count += _flush_tokenization_batch(pending, tokenizer=tokenizer)
        train_manifest = train_writer.close()
        validation_manifest = validation_writer.close()

    summary = {
        "documents": document_count,
        "skipped": skipped_count,
        "train_tokens": train_writer.total_tokens,
        "validation_tokens": validation_writer.total_tokens,
        "train_manifest": str(train_manifest),
        "validation_manifest": str(validation_manifest),
    }
    print(json.dumps(summary, indent=2))
    return train_manifest, validation_manifest


def main() -> None:
    """CLI entry point for streaming dataset preparation."""

    parser = argparse.ArgumentParser(
        description="Tokenize streaming text into memory-mapped shards."
    )
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_typed(args.config, DataPrepConfig)
    prepare(config)


if __name__ == "__main__":
    main()
