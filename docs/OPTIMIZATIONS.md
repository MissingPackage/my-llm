# Optimizations and compatibility

"All the optimizations" does not mean enabling them all at once: some address the same bottleneck, others are incompatible or hurt throughput once the model already fits comfortably. The presets encode combinations we have actually measured.

## Practical matrix

| Technique | Random pretraining | SFT QLoRA | DPO QLoRA | GRPO QLoRA | Default |
|---|---:|---:|---:|---:|---|
| BF16 compute / TF32 | yes | yes | yes | yes | on for Ada |
| BF16 stored weights | only to fit 2B | n/a, NF4 base | n/a | n/a | 2B only |
| NF4 + double quant | no | yes | yes | yes | on for Qwen |
| all-linear rsLoRA | no | yes | yes | yes | rank 64/64/32 |
| one-step LoftQ | no | yes | yes | yes | on, can be disabled |
| DoRA | no | possible | possible | possible | off: more overhead |
| gradient checkpointing | yes | yes | yes | yes | on |
| SDPA | yes | fallback | fallback | fallback | robust |
| Hub FlashAttention 2 | opt-in | yes | yes | yes | fast presets |
| BFD packing/padding-free | no | yes | off in TRL 1.8 | n/a | requires FlashAttention |
| Liger | possible | yes | no | yes | no with DPO precompute |
| chunked NLL | possible via TRL | alternative | n/a | n/a | not with Liger |
| reference precompute | n/a | n/a | yes | n/a | on for DPO |
| beta=0 / no reference | n/a | n/a | no | yes | on for GRPO |
| activation offload | n/a | OOM fallback | OOM fallback | not default | off |
| torch.compile | opt-in | ablation | ablation | ablation | off |
| continuous batching | n/a | n/a | n/a | large N | off at N=4 |
| vLLM colocated | n/a | n/a | n/a | possible | excluded on 16 GiB |

## Quantization and adapters

The official [PEFT Quantization](https://huggingface.co/docs/peft/developer_guides/quantization) guide recommends NF4, BF16 compute, double quantization, `prepare_model_for_kbit_training` and `all-linear` targets for QLoRA. The project applies these steps explicitly.

[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora) documents:

- rsLoRA: `alpha/sqrt(rank)` scaling, more stable at high ranks;
- DoRA: separates direction and magnitude, but costs more compute/memory;
- LoftQ: initializes the adapter to compensate for quantization error;
- `all-linear`: adapts every linear projection except the output layer.

LoftQ is used only on new adapters and requires safetensors weights. DoRA remains an ablation: at rank 64 the gain is not guaranteed and the overhead is real.

## Attention, packing and fused kernels

The Transformers guide on [attention backends](https://huggingface.co/docs/transformers/attention_interface) explains that SDPA automatically picks the fastest eligible CUDA backend. The fast presets use `kernels-community/flash-attn2`: the Kernel Hub avoids the fragile coupling between the `flash-attn` wheel, the PyTorch ABI and the local CUDA.

TRL enables padding-free automatically with BFD packing. This is only correct with FlashAttention 2/3, which understands the boundaries of packed sequences; for that reason the code rejects BFD+SDPA rather than risking contamination across examples.

[Liger](https://huggingface.co/docs/trl/liger_kernel_integration) fuses RMSNorm, RoPE, SwiGLU and the loss. The "20% throughput / 60% memory" figures are project benchmarks on different configurations, not a promise for this laptop. We always measure A/B.

## Loss memory

TRL 1.8 offers `chunked_nll`, which avoids materializing the full `[batch, seq, vocab]` logits and reports roughly 30% less memory on Qwen3-1.7B in a single-GPU benchmark. But it is an alternative to Liger's fused CE, not an addition. In the QLoRA+Liger presets we use `nll`; in the non-Liger fallbacks the chunked default can be left in place only after verifying the PEFT compatibility of the pinned version.

DPO clones the small initial adapter as `ref` on the same NF4 base and precomputes its log-probs. It therefore does not duplicate the base weights. TRL declares the precompute incompatible with Liger: the preset chooses the precompute because VRAM is the main constraint. In addition, the TRL 1.8 code temporarily disables padding-free DPO after a refactor; the YAML leaves it explicitly `false`.

## Optimizer

[bitsandbytes](https://huggingface.co/docs/bitsandbytes/optimizers) quantizes Adam's moments and keeps the small tensors in FP32. `min_8bit_size=4096` avoids quantizing parameters where noise and overhead outweigh the savings. The paged optimizer uses unified memory as protection against spikes, but continual page faults turn the GPU into a PCIe-bound system.

In pretraining, weight decay is applied to the matrices and not to the RMSNorm vectors. `zero_grad(set_to_none=True)` avoids pointless memsets; gradient clipping happens only once, after accumulation.

## Input pipeline and resume

- Rust tokenizer called on batches of documents;
- `uint16` tokens in memory-mapped shards;
- sampling proportional to the possible start offsets;
- pinned host tensors and H2D copy on a dedicated stream;
- prefetched queue saved together with the RNG state, avoiding skips/repeats on resume.

## Flags to benchmark, not to assume

- `torch_compile`: can speed up fixed shapes, but adds startup cost, graph breaks and memory;
- `activation_offloading`: saves VRAM by moving activations over PCIe;
- `top_entropy_quantile: 0.2`: updates only high-entropy tokens in GRPO; it is an ablation;
- continuous batching: TRL recommends it mainly for generation batches of `N>=32`; the laptop uses N=4;
- DoRA/ephemeral offload: quality/overhead to be measured;
- sequence length 4K: dropping to 2K/1K is the first practical mitigation.
