"""Train a byte-level BPE tokenizer and an assistant-loss-aware chat template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from my_llm.config import TokenizerTrainConfig, load_typed
from my_llm.sources import iter_texts

PAD = "<pad>"
BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"

# TRL reads the generation block to build an assistant-only loss mask.  Removing
# these markers would silently train SFT on user/system tokens as well.
CHAT_TEMPLATE = r"""{%- if messages %}{{ bos_token }}{%- endif %}
{%- for message in messages %}
{%- if message['role'] == 'system' %}
{{ '<|system|>\n' + message['content'] | trim + eos_token + '\n' }}
{%- elif message['role'] == 'user' %}
{{ '<|user|>\n' + message['content'] | trim + eos_token + '\n' }}
{%- elif message['role'] == 'assistant' %}
{{ '<|assistant|>\n' }}{% generation %}{{ message['content'] | trim + eos_token }}{% endgeneration %}{{ '\n' }}
{%- else %}
{{ raise_exception('Unsupported chat role: ' + message['role']) }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{ '<|assistant|>\n' }}{%- endif %}"""


def train_tokenizer(config: TokenizerTrainConfig) -> Path:
    """Train byte-level BPE from streamed text and save a Transformers tokenizer."""

    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install training dependencies: uv sync --extra train") from exc

    # Byte fallback guarantees that every Unicode input is representable, including
    # rare Italian text or malformed web bytes absent from the training sample.
    raw = Tokenizer(models.BPE(unk_token=UNK, byte_fallback=True))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    raw.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=[PAD, BOS, EOS, UNK, SYSTEM, USER, ASSISTANT],
        # Seed all 256 byte symbols so fallback cannot emit an unknown byte.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    texts = iter_texts(
        config.sources,
        seed=config.seed,
        shuffle_buffer=config.shuffle_buffer,
        max_documents=config.max_documents,
    )
    train_kwargs = {"trainer": trainer}
    if config.max_documents is not None:
        train_kwargs["length"] = config.max_documents
    raw.train_from_iterator(texts, **train_kwargs)

    # Wrap the Rust tokenizer in Transformers metadata so model checkpoints, TRL
    # and inference all share exactly the same IDs and chat formatting.
    fast = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token=PAD,
        bos_token=BOS,
        eos_token=EOS,
        unk_token=UNK,
        additional_special_tokens=[SYSTEM, USER, ASSISTANT],
        model_max_length=config.model_max_length,
        clean_up_tokenization_spaces=False,
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.padding_side = "right"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(config.output_dir)
    metadata = {
        "vocab_size": len(fast),
        "requested_vocab_size": config.vocab_size,
        "model_max_length": config.model_max_length,
        "sources": [source.model_dump(mode="json") for source in config.sources],
        "seed": config.seed,
    }
    (config.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Catch a broken Jinja template before a multi-hour tokenizer/data run proceeds.
    rendered = fast.apply_chat_template(
        [{"role": "user", "content": "tokenizer smoke test"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if USER not in rendered or ASSISTANT not in rendered:
        raise RuntimeError("Saved chat template failed its smoke test")
    return config.output_dir


def main() -> None:
    """CLI entry point for tokenizer training."""

    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer from scratch.")
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_typed(args.config, TokenizerTrainConfig)
    output = train_tokenizer(config)
    print(f"Tokenizer saved to {output}")


if __name__ == "__main__":
    main()
