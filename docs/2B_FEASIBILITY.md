# Feasibility of the 2B from scratch on 16 GiB

## What "fits in memory" means

The `configs/pretrain/2b-fit.yaml` profile has:

- vocabulary: 32,000, tied embedding/head;
- hidden size: 2,560;
- SwiGLU intermediate: 6,912;
- 28 blocks;
- GQA: 20 query heads, 5 KV heads;
- computed parameters: **2,027,174,400**.

The estimate is tested without allocating the model, via `approximate_parameter_count`. The fit requires that at least the following coexist:

| Persistent state | Estimate |
|---|---:|
| BF16 weights | 3.78 GiB |
| BF16 gradients | 3.78 GiB |
| two 8-bit moments | ~3.78 GiB + metadata |
| subtotal | ~11.33 GiB |

Less than 5 GiB remain for checkpointed activations, CUDA workspace, allocator, loss/logits, and batch. This is why microbatch 1, sequence 512, gradient checkpointing, and paged AdamW 8-bit are used. There is no universal guarantee: driver, kernel version, and already-occupied VRAM change the peak.

With the classic prudent budget of 16 bytes/parameter, the state alone would be ~30.2 GiB. So the 2B does not fit with traditional FP32 AdamW.

## What "trained" means

The fit does not imply a good model. Following the compute-optimal starting point of [Chinchilla](https://arxiv.org/abs/2203.15556), 2.027B parameters require roughly 40B tokens. The standard estimate:

```text
FLOP ~= 6 * parameters * tokens ~= 4.865e20
```

yields 188–282 ideal days at 30–20 sustained TFLOP/s. It is a lower bound: the paged optimizer can transfer pages over PCIe, checkpoints synchronize, evaluation interrupts training, and the TGP/cooling of the 4090 Laptop varies by chassis. The real run is therefore likely to take many months, potentially over a year.

## Why the sequence is 512

Activations grow with the microbatch length. Accumulation of 128 brings the effective batch to 65,536 tokens/update without raising the peak of a single forward. The model declares 2,048 positions, but doing all pretraining at 512 does not train the long context well: a subsequent context-extension curriculum would be needed, probably on more capable hardware.

## Correct procedure

1. Prepare the tokenizer and at least the `fineweb-edu-2b` corpus.
2. Run `llm-doctor` and close every unnecessary GPU process.
3. Launch `make fit-2b`.
4. Check that the first `optimizer.step()` actually happens: bitsandbytes allocates part of the state lazily, so a single forward is not a sufficient test.
5. Measure allocated peak, reserved, temperature, and page-fault/PCIe traffic.
6. Do not proceed with the full run until 100–500 steps show finite loss and thermally stable throughput.

## OOM mitigation order

1. reduce `sequence_length` to 384 or 256;
2. verify `parameter_dtype: bf16` and `optimizer: paged_adamw_8bit`;
3. disable `compile`;
4. reduce `prefetch_batches` to 0;
5. close compositor/browser/CUDA processes;
6. only as a last resort, increase reliance on paging, accepting the heavy performance cost.

Reducing `gradient_accumulation_steps` does not lower the microbatch memory peak. Reducing the LoRA rank does not apply to pretraining from scratch: QLoRA adapts frozen weights, it does not create a complete base model.

## Project decision

The 2B from scratch remains a didactic systems-engineering experiment. The useful result that can be iterated on the stated machine is the QLoRA path on Qwen3-1.7B-Base described in `POSTTRAIN_2B.md`.
