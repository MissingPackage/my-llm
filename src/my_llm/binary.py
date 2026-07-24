"""Compact token-shard writer and memory-mapped random-window sampler.

Token IDs are stored without Python/Arrow overhead.  A 32K tokenizer fits in
``uint16`` (two bytes/token), making the 40B-token thought experiment ~74.5 GiB.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class BinaryShardWriter:
    """Append token IDs to fixed-size binary shards and write an atomic manifest."""

    def __init__(
        self,
        output_dir: str | Path,
        prefix: str,
        *,
        dtype: str,
        shard_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a writer; the first file is opened lazily on the first token."""

        if dtype not in {"uint16", "uint32"}:
            raise ValueError(f"Unsupported token dtype: {dtype}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.dtype = np.dtype(dtype)
        self.shard_tokens = shard_tokens
        self.metadata = metadata or {}
        self.shards: list[dict[str, Any]] = []
        self.total_tokens = 0
        self._file = None
        self._path: Path | None = None
        self._tokens_in_shard = 0

    def _open_shard(self) -> None:
        """Open the next monotonically numbered shard."""

        index = len(self.shards)
        self._path = self.output_dir / f"{self.prefix}-{index:05d}.bin"
        self._file = self._path.open("wb")
        self._tokens_in_shard = 0

    def _close_shard(self) -> None:
        """Flush one shard and add its exact token count to the manifest."""

        if self._file is None or self._path is None:
            return
        self._file.flush()
        self._file.close()
        self.shards.append({"path": self._path.name, "tokens": self._tokens_in_shard})
        self._file = None
        self._path = None
        self._tokens_in_shard = 0

    def add(self, tokens: list[int] | np.ndarray) -> int:
        """Append tokens, splitting a document across shard boundaries if needed."""

        array = np.asarray(tokens, dtype=np.int64)
        if not array.size:
            return 0
        if array.min() < 0 or array.max() > np.iinfo(self.dtype).max:
            raise ValueError(f"Token id does not fit in {self.dtype.name}")
        written = 0
        while written < array.size:
            if self._file is None:
                self._open_shard()
            room = self.shard_tokens - self._tokens_in_shard
            count = min(room, array.size - written)
            # Convert only the slice being written; a very long document never needs
            # an additional full-size uint16 copy in memory.
            array[written : written + count].astype(self.dtype).tofile(self._file)
            self._tokens_in_shard += count
            self.total_tokens += count
            written += count
            if self._tokens_in_shard == self.shard_tokens:
                self._close_shard()
        return int(array.size)

    def close(self) -> Path:
        """Close the last shard and atomically publish its JSON manifest."""

        self._close_shard()
        manifest = {
            "version": 1,
            "dtype": self.dtype.name,
            "total_tokens": self.total_tokens,
            "shards": self.shards,
            **self.metadata,
        }
        destination = self.output_dir / f"{self.prefix}-manifest.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
        return destination

    def __enter__(self) -> BinaryShardWriter:
        """Support ``with BinaryShardWriter(...)`` usage."""

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Publish a manifest only after a successful context-manager body."""

        if exc_type is None:
            self.close()
        elif self._file is not None:
            self._file.close()


class TokenShardCorpus:
    """Memory-map validated shards and sample windows proportionally by capacity."""

    def __init__(self, manifest_path: str | Path, sequence_length: int) -> None:
        """Validate byte sizes before exposing any shard to the trainer."""

        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.dtype = np.dtype(manifest["dtype"])
        self.sequence_length = sequence_length
        base = self.manifest_path.parent
        self.shards: list[np.memmap] = []
        weights = []
        for shard in manifest["shards"]:
            token_count = int(shard["tokens"])
            if token_count < sequence_length:
                continue
            expected_bytes = token_count * self.dtype.itemsize
            path = base / shard["path"]
            if path.stat().st_size != expected_bytes:
                raise ValueError(f"Size mismatch for {path}")
            self.shards.append(np.memmap(path, mode="r", dtype=self.dtype, shape=(token_count,)))
            # Weight by valid start positions, not merely by shard count, so every
            # possible training window has the same sampling probability.
            weights.append(token_count - sequence_length + 1)
        if not self.shards:
            raise ValueError(f"No shard in {manifest_path} is long enough for {sequence_length=}")
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights /= self.weights.sum()

    def sample_numpy(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        """Copy random contiguous windows into an int64 batch accepted by embeddings."""

        result = np.empty((batch_size, self.sequence_length), dtype=np.int64)
        choices = rng.choice(len(self.shards), size=batch_size, p=self.weights)
        for row, shard_index in enumerate(choices):
            shard = self.shards[int(shard_index)]
            start = int(rng.integers(0, len(shard) - self.sequence_length + 1))
            result[row] = shard[start : start + self.sequence_length]
        return result
