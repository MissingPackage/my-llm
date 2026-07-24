.PHONY: install test lint format doctor smoke-data smoke-pretrain smoke-posttrain smoke-align smoke-grpo estimate fit-2b qwen-sft qwen-merge-sft qwen-dpo qwen-merge-dpo qwen-reasoning-sft qwen-merge-reasoning qwen-grpo

# ``fast`` installs optional Liger and Hub FlashAttention kernels used by Qwen YAMLs.
install:
	uv sync --extra train --extra fast --extra dev

test:
	uv run --extra dev pytest

lint:
	uv run --extra dev ruff check .

format:
	uv run --extra dev ruff format .

doctor:
	uv run --extra train --extra fast llm-doctor

estimate:
	uv run llm-estimate configs/pretrain/2b-chinchilla.yaml --target-tokens 40000000000

# The smoke chain remains CPU-capable and never downloads a billion-parameter model.
smoke-data:
	uv run --extra train llm-tokenizer configs/tokenizer/smoke.yaml
	uv run --extra train llm-prepare configs/data/smoke.yaml

smoke-pretrain: smoke-data
	uv run --extra train llm-pretrain configs/pretrain/debug.yaml --max-steps 20

smoke-posttrain: smoke-pretrain
	uv run --extra train llm-posttrain configs/posttrain/sft-smoke.yaml

smoke-align: smoke-posttrain
	uv run --extra train llm-posttrain configs/posttrain/dpo-smoke.yaml

smoke-grpo: smoke-align
	uv run --extra train llm-grpo configs/posttrain/grpo-smoke.yaml

# This performs two very expensive updates only; it is not useful pretraining.
fit-2b:
	uv run --extra train llm-pretrain configs/pretrain/2b-fit.yaml

# Practical ~2B path. Merge on CPU between stages to make each reference explicit.
qwen-sft:
	uv run --extra train --extra fast llm-posttrain configs/posttrain/qwen3-1.7b-sft.yaml

qwen-merge-sft:
	uv run --extra train llm-merge-adapter --base-model Qwen/Qwen3-1.7B-Base --adapter artifacts/qwen3-1.7b-sft/final --output artifacts/qwen3-1.7b-sft-merged

qwen-dpo:
	uv run --extra train --extra fast llm-posttrain configs/posttrain/qwen3-1.7b-dpo.yaml

qwen-merge-dpo:
	uv run --extra train llm-merge-adapter --base-model artifacts/qwen3-1.7b-sft-merged --adapter artifacts/qwen3-1.7b-dpo/final --output artifacts/qwen3-1.7b-dpo-merged

qwen-reasoning-sft:
	uv run --extra train --extra fast llm-posttrain configs/posttrain/qwen3-1.7b-reasoning-sft.yaml

qwen-merge-reasoning:
	uv run --extra train llm-merge-adapter --base-model artifacts/qwen3-1.7b-dpo-merged --adapter artifacts/qwen3-1.7b-reasoning-sft/final --output artifacts/qwen3-1.7b-reasoning-sft-merged

qwen-grpo:
	uv run --extra train --extra fast llm-grpo configs/posttrain/qwen3-1.7b-grpo.yaml
