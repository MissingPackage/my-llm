# Operational runbook

## Fedora preflight

```bash
nvidia-smi
uv sync --extra train --extra fast --extra dev
uv run --extra train --extra fast llm-doctor
make test
make smoke-pretrain
```

If PyTorch does not see CUDA, use the official [PyTorch Start Locally](https://pytorch.org/get-started/locally/) selector instead of installing random toolkits/cuDNN. The PyTorch wheels bundle the expected runtime; what matters most is a compatible host driver.

## Choose the path

- Useful ~2B result: follow `POSTTRAIN_2B.md` and the `qwen-*` targets.
- Full study from scratch: use `100m.yaml`.
- Systems experiment: use `2b-fit.yaml`, never the full 2B as your first command.

## Calibration

Every real profile starts with 100–500 steps or a data subset. Record:

- token/s or step/s after kernel warm-up;
- allocated/reserved VRAM and temperature after 20–30 minutes;
- loss, grad norm, token accuracy/reward variance;
- checkpoint size and duration;
- A/B difference against a single changed optimization.

The best time estimate is `target tokens / measured token/s`, corrected for evaluation/checkpointing/downtime.

## Pretraining OOM ladder

1. reduce sequence length;
2. bring microbatch to 1 and increase gradient accumulation to preserve the batch;
3. enable checkpointing, disable compile/prefetch;
4. on the 2B, verify BF16 stored parameters and paged AdamW8bit;
5. kill GPU processes;
6. avoid sustained paging: it "fits" but may not be trainable within a reasonable time.

## QLoRA OOM ladder

1. reduce `max_length` / `max_completion_length`;
2. microbatch 1, higher accumulation;
3. `activation_offloading: true` for SFT/DPO;
4. LoRA rank 64 → 32;
5. GRPO: keep at least 4 generations if possible, reduce length first;
6. SDPA fallback without packing if the fast kernels do not work.

## Resume

Pretraining:

```bash
uv run --extra train llm-pretrain configs/pretrain/100m.yaml \
  --resume artifacts/pretrain-100m/checkpoint-010000
```

The checkpoint includes the optimizer, scheduler, RNG, and prefetched queue. `2b-fit.yaml` deliberately saves only weights and is not resumable.

TRL:

```bash
uv run --extra train --extra fast llm-posttrain CONFIG.yaml --resume OUTPUT/checkpoint-N
uv run --extra train --extra fast llm-grpo CONFIG.yaml --resume OUTPUT/checkpoint-N
```

Do not change the base, tokenizer, dataset schema, or adapter shape during a resume. A change in LR/batch is a new experiment.

## Quality gates

| Phase | Minimum check |
|---|---|
| tokenizer | round-trip, rare bytes, EN/IT length |
| pretraining | finite train/validation loss, no divergence |
| SFT | roles, instruction following, EOS, base regressions |
| DPO | reward margin/accuracy and absence of length collapse |
| reasoning SFT | non-zero GSM8K accuracy@1/pass@4 |
| GRPO | non-zero reward std, better held-out, no format hacking |

Always keep qualitative examples. A rising training reward with an unchanged test is a failure, not a success.

## Safe cleanup

Datasets, cache, and artifacts are not versioned. Before deleting a merged checkpoint, verify that later stages do not use it as a base, and keep the adapter, YAML, commit, and manifest. Do not delete the only optimizer checkpoint of a long run.
