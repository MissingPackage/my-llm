# my-llm

A didactic, reproducible lab for studying the full life cycle of a language model on a single **16 GiB RTX 4090 Laptop GPU**:

`tokenizer -> pretraining -> instruction SFT -> DPO -> reasoning SFT -> GRPO -> evaluation`

The repository offers two distinct tracks:

1. **Recommended, actually attainable track:** QLoRA post-training of [`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base) (positioned by Hugging Face in the "2B" class). It is text-only, Apache-2.0, pre-trained but not instruction-tuned — an excellent starting point for learning alignment and reasoning on modest hardware.
2. **Experimental from-scratch track:** a Llama-style decoder with **2,027,174,400 parameters**. The memory fit is attempted with BF16, paged AdamW 8-bit, gradient checkpointing, and 512-token sequences. Useful pretraining, however, remains a many-month project (see below).

## Key features

- **One codebase, both tracks.** The same tooling drives from-scratch pretraining and QLoRA post-training, so you can compare the two honestly.
- **Single-GPU by design.** Every default is chosen to fit 16 GiB: NF4 + double quantization for the frozen base, `all-linear` QLoRA with rsLoRA and LoftQ, gradient checkpointing, paged AdamW 8-bit, BF16/TF32, and BFD packing with assistant-only loss.
- **Full post-training pipeline:** instruction SFT (UltraChat) → DPO (UltraFeedback Binarized) → reasoning SFT (OpenR1-Math) → GRPO (GSM8K) with DAPO loss and truncation masking.
- **Reproducible smoke tests.** A tiny corpus ships in `sample_data/` so the entire pipeline can be exercised end-to-end on CPU, without downloading a billion-parameter model.
- **Evaluation and governance built in:** GSM8K pass@k evaluation, a frozen personal benchmark harness, a preference-data constitution, and documented data governance.
- **Honest feasibility accounting** for the from-scratch 2B path, so expectations match arithmetic.

## An honest answer about 2B from scratch

The `2b-chinchilla` preset holds ~2.027B parameters and 40B tokens, i.e. ~19.7 tokens per parameter. The bare `6ND` arithmetic alone is about `4.86e20` FLOP:

| Hypothetical sustained throughput | Arithmetic lower bound |
|---:|---:|
| 20 TFLOP/s | 282 days |
| 25 TFLOP/s | 225 days |
| 30 TFLOP/s | 188 days |

These figures exclude evaluation, checkpointing, paging, data loading, thermal throttling, or the laptop simply being switched off. Real wall-clock time can therefore approach or exceed a year. The 40B `uint16` tokens also occupy ~74.5 GiB. For those reasons the project keeps the from-scratch 2B as a feasibility experiment and focuses the useful work on post-training the Qwen base.

## Installation

Requirements: Fedora/Linux, a working NVIDIA driver, Python 3.12, [`uv`](https://docs.astral.sh/uv/), and free disk space for models/datasets/checkpoints.

```bash
git clone git@github.com:MissingPackage/my-llm.git
cd my-llm
uv sync --extra train --extra fast --extra dev
uv run --extra train --extra fast llm-doctor
```

The `fast` extra installs Liger and the Hugging Face kernel loader. The fast presets download FlashAttention 2 from the Kernel Hub on first use, avoiding a local `flash-attn` build. For the most conservative fallback, set in the YAML:

```yaml
attention_backend: sdpa
packing: false
padding_free: false
use_liger_kernel: false
```

PyTorch SDPA still selects an appropriate CUDA kernel automatically.

## Quickstart (Makefile targets)

The `Makefile` wraps the common workflows:

```bash
make install        # uv sync with train + fast + dev extras
make doctor         # hardware/environment check
make test           # unit tests (no model downloads)
make lint           # ruff check

# End-to-end smoke run on the tiny bundled corpus (CPU-capable):
make smoke-grpo     # runs data -> pretrain -> SFT -> DPO -> GRPO on toy sizes

# Recommended ~2B post-training track (downloads Qwen3-1.7B-Base):
make qwen-sft            && make qwen-merge-sft
make qwen-dpo            && make qwen-merge-dpo
make qwen-reasoning-sft  && make qwen-merge-reasoning
make qwen-grpo

# From-scratch feasibility:
make estimate       # FLOP/time estimate for the 2B preset
make fit-2b         # two updates only — proves the config fits in memory
```

Each post-training stage produces a small adapter that is merged into BF16 on CPU before the next stage, making every stage's starting point explicit. See [`docs/POSTTRAIN_2B.md`](docs/POSTTRAIN_2B.md) for the full walkthrough.

## What is *not* included

- **Trained weights and adapters.** Nothing under `artifacts/` (checkpoints, merged models, LoRA adapters) is shipped. Train them with the Makefile targets above.
- **External datasets.** UltraChat, UltraFeedback Binarized, GSM8K, OpenR1-Math, and FineWeb are **not redistributed**. The configs reference them by their Hugging Face path and the loaders download them separately at run time. Review licences, provenance, PII, and opt-out terms in [`docs/DATA_GOVERNANCE.md`](docs/DATA_GOVERNANCE.md) before use.
- Only a tiny illustrative corpus (`sample_data/`) and the project's own small identity/preference/benchmark data are included, for smoke tests and reproducibility.

## Layout

```text
configs/               commented, versioned presets (tokenizer, data, pretrain, posttrain)
docs/                  feasibility, optimizations, research, data governance, persona, runbook
sample_data/           tiny corpus for smoke tests
data/                  project identity + preference datasets (small, first-party)
benchmarks/            frozen personal evaluation benchmark
src/my_llm/            tokenizer, sharding, pretraining, QLoRA, DPO, GRPO, evaluation
tests/                 core tests that never download a model
```

The code carries module/function docstrings and comments on non-obvious decisions: the comments explain the mental model rather than restating trivial assignments.

## Guardrails

1. Run `llm-doctor`, the tests, and the smoke chain before any long run.
2. Version commits, YAML, seeds, tokenizer, data manifests, and package versions together.
3. Keep train/validation/test separate; the GSM8K `test` split never enters GRPO.
4. Track accuracy@1, pass@k, reward std, KL when present, length, and qualitative samples.
5. DPO aligns to the dataset's preferences; GRPO optimizes the defined reward. Neither one automatically makes a model safe, truthful, or production-ready.
6. Read licences, provenance, PII, and opt-out terms in [`docs/DATA_GOVERNANCE.md`](docs/DATA_GOVERNANCE.md).

## Further reading

[`docs/POSTTRAIN_2B.md`](docs/POSTTRAIN_2B.md), [`docs/2B_FEASIBILITY.md`](docs/2B_FEASIBILITY.md), [`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md), [`docs/RUNBOOK.md`](docs/RUNBOOK.md), and [`docs/RESEARCH.md`](docs/RESEARCH.md).

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 MissingPackage.
