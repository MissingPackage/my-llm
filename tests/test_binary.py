"""Round-trip and corruption checks for compact token shards."""

import json

import numpy as np
import pytest

from my_llm.binary import BinaryShardWriter, TokenShardCorpus


def test_shards_round_trip_and_sample(tmp_path) -> None:
    writer = BinaryShardWriter(
        tmp_path, "train", dtype="uint16", shard_tokens=10, metadata={"test": True}
    )
    writer.add(list(range(25)))
    manifest_path = writer.close()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["total_tokens"] == 25
    assert [item["tokens"] for item in manifest["shards"]] == [10, 10, 5]

    corpus = TokenShardCorpus(manifest_path, sequence_length=4)
    batch = corpus.sample_numpy(8, np.random.default_rng(42))
    assert batch.shape == (8, 4)
    assert batch.dtype == np.int64
    assert np.all(np.diff(batch, axis=1) == 1)


def test_writer_rejects_token_overflow(tmp_path) -> None:
    writer = BinaryShardWriter(tmp_path, "train", dtype="uint16", shard_tokens=10)
    with pytest.raises(ValueError, match="does not fit"):
        writer.add([70_000])


def test_corpus_detects_truncated_shard(tmp_path) -> None:
    writer = BinaryShardWriter(tmp_path, "train", dtype="uint16", shard_tokens=100)
    writer.add(list(range(20)))
    manifest_path = writer.close()
    shard = tmp_path / "train-00000.bin"
    shard.write_bytes(shard.read_bytes()[:-2])
    with pytest.raises(ValueError, match="Size mismatch"):
        TokenShardCorpus(manifest_path, sequence_length=4)
