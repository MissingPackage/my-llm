# Constitutional DPO preferences

Goal `constitutional-dpo`. The constitution (`docs/CONSTITUTION.md`, RATIFIED)
governs labeling; every preference records the applied principle
(field `principle`, mandatory — the pipeline rejects preferences without it).

- `smoke-prompts.jsonl` — 60 principle-targeted prompts (P1-P9), reusable.
- `smoke-pairs.jsonl` · `smoke-verdicts.jsonl` · `constitutional-smoke.jsonl`
  — the pipeline's test chain (phase 4): completions from a debug
  checkpoint, synthetic unreviewed verdicts (convention C4). Test material, never
  real training.
- `constitutional-v1.jsonl` — human-labelled preferences (phase 5). Contains the SEED:
  the 10 A/B pairs from the ratified constitution (ruling C5(c), 2026-07-17),
  promoted via `ratified-pairs.jsonl` + `ratified-verdicts.jsonl` (chosen=A
  by construction of the ratification; `unreviewed: false`). The batch generated on the
  laptop is APPENDED here once labeled.

## Phase 5 — first real batch (procedure, laptop with GPU)

The smoke checkpoint's completions are word salad: discriminable pairs
require a real model. On the laptop (Qwen3-1.7B or a post-train of it):

```bash
# 1. generate K responses per prompt from the real checkpoint
uv run llm-genpairs <checkpoint> \
  --prompts data/preferences/smoke-prompts.jsonl \
  --output data/preferences/real-pairs-batch-001.jsonl \
  --k 2 --temperature 0.8 --seed 20260717

# 2. prepare the verdicts file (one line per pair, verdict: "_")
#    and fill it in: chosen/rejected + principle, following the constitution's
#    reading rules (ambiguous pairs: discard, do not force)

# 3. promote the labeled preferences
uv run llm-judge-pairs --pairs data/preferences/real-pairs-batch-001.jsonl \
  --verdicts <verdicts-file> \
  --output data/preferences/constitutional-v1.jsonl
```

The exact source of the first batch (checkpoint, K, how many
prompts) is decision C5 — labeling is, by contract, a human responsibility.
