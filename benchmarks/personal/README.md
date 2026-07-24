# Personal benchmark

Held-out gate for each stage of the personalization pipeline (goal
`personal-benchmark`). Never used as a training source: a test verifies that
no YAML config references it.

- `drafts/` — batches proposed by the loop, always `approved: false`.
- `v1.jsonl` — frozen release: only `approved: true` items (set by hand),
  >= 100 items, >= 5 calibration items. The freeze is recorded in `v1.sha256`.
- Go/no-go gate = deterministic items only (`exact_numeric`, `exact_string`,
  `regex`). The `llm_rubric` items are a separate trend (ruling A5).

Item schema: see `src/my_llm/benchmark.py` (`BenchmarkItem`).

## Approval via a verdicts file (ruling A4)

1. Open `drafts/batch-001.jsonl` and `drafts/verdicts-batch-001.jsonl` side by side.
2. For each line of the verdicts file, replace `"_"` with:
   - `"approve"` — the item enters v1 as is;
   - `"reject"` — excluded (add `"note"` if you want the reason on record);
   - `"edit"` — enters corrected: add the fields to replace among
     `prompt`, `reference`, `rubric`, `domain`, `lang`
     (e.g. `{"id": "...", "verdict": "edit", "reference": "42"}`).
3. `uv run llm-benchmark-freeze` — rejects missing or `"_"` verdicts, applies
   the edits while re-validating the schema, writes a sorted `v1.jsonl` + `v1.sha256`.
   It needs >= 100 approved and >= 5 calibration items, or the freeze fails.
4. From that point on, v1 is immutable (anti-drift test on the fingerprint); subsequent
   changes become a v2.
