# Runbook Qwen3 ~2B: alignment and reasoning

## Why Qwen3-1.7B-Base

The [official model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base) declares 1.7B parameters (1.4B non-embedding), 28 layers, GQA 16/8, 32K context, 119 languages, an Apache-2.0 license, and a "Pretraining" stage. Hugging Face displays it in the 2B class. It is therefore large enough to make post-training interesting, yet simple to load with `AutoModelForCausalLM` as text-only.

The more recent [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) is exactly 2B but includes a vision encoder, a 248,320-token vocabulary, and a hybrid Gated DeltaNet/attention layout. It could become a future profile; it is not the default teaching target on 16 GiB.

## Preflight

```bash
uv sync --extra train --extra fast --extra dev
uv run --extra train --extra fast llm-doctor
make test
```

Close applications that use CUDA. Leave room for the Hugging Face cache, datasets, and at least three merged BF16 checkpoints (~3–4 GiB each, on top of the adapters).

## Stage 1 — general SFT

```bash
make qwen-sft
make qwen-merge-sft
```

Output:

- adapter: `artifacts/qwen3-1.7b-sft/final`;
- merged: `artifacts/qwen3-1.7b-sft-merged`.

Check that loss is finite, along with token accuracy, VRAM, and a few chats. Do not call a run an "instruct model" just because its loss is lower: verify follow-ups, roles, and termination.

## Stage 2 — DPO

```bash
make qwen-dpo
make qwen-merge-dpo
```

The new adapter starts from the merged SFT. Since the model arrives at DPO already wrapped in PEFT, TRL 1.8 copies the initial adapter under the name `ref` and freezes it on the same quantized base. The reference therefore also includes the initial LoftQ compensation, without duplicating the base weights. `precompute_ref_log_probs` runs an initial pass and then uses the cache during updates.

Monitor `rewards/margins`, `rewards/accuracies`, chosen/rejected length, and regressions on the SFT prompts. DPO can learn verbosity or the bias of the synthetic judge.

## Stage 3 — reasoning SFT

```bash
make qwen-reasoning-sft
make qwen-merge-reasoning
```

OpenR1-Math-220k `default` contains problems and multiple generations. The formatter picks the first complete trace marked correct by Math Verify (or by the Llama judge), falling back to the reference solution. The preset uses 20K examples: a quantity you can iterate on with the laptop. Increase it only after a held-out comparison. The final answer is made easily verifiable with tags.

Mandatory gate:

```bash
uv run --extra train llm-eval-gsm8k \
  artifacts/qwen3-1.7b-reasoning-sft-merged \
  --load-in-4bit --max-samples 200 --samples-per-prompt 4 \
  --batch-size 2 --output runs/qwen-reasoning-sft-gsm8k.jsonl
```

Proceed only if pass@4 and reward variance are clearly non-zero.

## Stage 4 — GRPO

```bash
make qwen-grpo
```

Start with 50–100 steps, evaluate, then extend up to the configured 500. Do not use the GSM8K test as a reward. With four completions, a prompt that is entirely correct or entirely wrong has a null normalized advantage; watch `frac_reward_zero_std`.

Final base/adapter comparison:

```bash
uv run --extra train llm-eval-gsm8k \
  artifacts/qwen3-1.7b-reasoning-sft-merged \
  --adapter artifacts/qwen3-1.7b-grpo/final --load-in-4bit \
  --max-samples 500 --samples-per-prompt 4 --batch-size 2 \
  --output runs/qwen-grpo-gsm8k.jsonl
```

## QLoRA OOM ladder

In order:

1. kill GPU processes and restart the Python process;
2. halve `max_length` (4K → 2K → 1K) or `max_completion_length`;
3. keep microbatch at 1 and raise accumulation if you want to preserve the effective batch;
4. enable `activation_offloading: true` in SFT/DPO;
5. reduce LoRA rank 64 → 32;
6. disable packing and use SDPA if the Kernel Hub is not compatible, accepting lower throughput;
7. for GRPO, reduce completion length before dropping below four generations.

Do not enable colocated vLLM by reflex: with 16 GiB it duplicates/reserves the KV cache and competes with the backward pass. The in-process Transformers backend is more predictable.

## Merge or adapter?

- **To iterate/infer:** base + 4-bit adapter minimizes disk/VRAM.
- **Between stages:** a BF16 merge makes the reference point clear and freezes the snapshot.
- **To distribute:** evaluate both the adapter (small, requires the exact base) and the merged model (self-contained, larger). Always keep licenses and provenance.

The merge does not quantize permanently: it reloads the base in BF16, applies LoRA with `safe_merge=True`, removes the wrappers, and saves sharded safetensors.
