# Research and technical decisions

Research verified on 11 July 2026. Official documentation, the authors' model/dataset cards and original papers are given priority.

## Hardware and scale

The [NVIDIA RTX 40 Laptop page](https://www.nvidia.com/en-us/geforce/laptops/40-series/) gives the RTX 4090 Laptop 9,728 CUDA cores and **16 GB GDDR6**. It should not be confused with the 24 GB desktop 4090. TGP, clocks and cooling depend on the chassis; for that reason every time estimate must be recalibrated against sustained tokens/s and real temperature.

[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) motivates the ~20 tokens/parameter starting point. The project applies `FLOP ~= 6ND`, but labels it a lower bound. For 2,027,174,400 parameters and 40B tokens this works out to ~4.865e20 FLOP, i.e. 188–282 ideal days at 30–20 TFLOP/s; the real run is longer.

## Choosing the ~2B base

[`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base) is Apache-2.0, a text-only causal LM, pretraining-only, 1.7B parameters (1.4B non-embedding), 28 layers, GQA with 16 query/8 KV heads and a 32,768 context. The card reports pretraining on 36T tokens and 119 languages; the Hugging Face interface also classifies it as model size 2B. These properties make it the practical default.

[`Qwen/Qwen3.5-2B-Base`](https://huggingface.co/Qwen/Qwen3.5-2B-Base) is exactly 2B and Apache-2.0, but it is a causal LM with a vision encoder, a 248,320 vocabulary, a 262K context and a hybrid Gated DeltaNet/attention architecture. It requires `AutoModelForMultimodalLM`; it was not chosen for a memory-constrained, text-only lab.

## Architecture from scratch

The [Transformers Llama documentation](https://huggingface.co/docs/transformers/model_doc/llama) provides a decoder with RoPE, RMSNorm, SwiGLU and GQA. The repository uses the implementation and the checkpoint format, not Llama weights. The random-init architecture has tied embedding/head, no bias and reduced KV heads.

The 2B profile builds BF16 parameters directly on the device to avoid a temporary ~8 GiB FP32 model. It is a quality/stability concession needed to fit; the small profiles keep FP32 parameters and BF16 autocast.

## Tokenizer and data

[Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index) supports `train_from_iterator`. Byte-Level BPE guarantees coverage of every byte; 32K IDs fit in `uint16`. The chat template includes the Jinja generation markers that TRL uses for assistant-only loss.

[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) exposes 10BT/100BT samples and declares ODC-By. The [FineWeb](https://arxiv.org/abs/2406.17557) paper describes the filtering and deduplication. The Italian variant uses [FineWeb 2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2). The local shards keep source, seed and tokenizer fingerprint; EOS separates documents.

## Single-GPU pretraining

Primary sources:

- [PyTorch automatic mixed precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html);
- [scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html);
- [activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html);
- [bitsandbytes 8-bit optimizers](https://huggingface.co/docs/bitsandbytes/optimizers).

bitsandbytes reports up to 75% less optimizer state compared to 32-bit moments, keeps small tensors in FP32 and documents `min_8bit_size`/percentile clipping. The 2B preset uses PagedAdamW8bit: paging protects against spikes but does not make exceeding VRAM free.

## QLoRA and PEFT

The [PEFT Quantization](https://huggingface.co/docs/peft/developer_guides/quantization) guide prescribes NF4, double quant, BF16 compute, `prepare_model_for_kbit_training` and `target_modules="all-linear"`. The [LoRA reference](https://huggingface.co/docs/peft/package_reference/lora) documents rsLoRA, DoRA, LoftQ and all-linear targets.

The [TRL PEFT integration](https://huggingface.co/docs/trl/peft_integration) notes that all trainers support PEFT/QLoRA and suggests higher learning rates for adapters than for full fine-tuning. The code applies PEFT directly before the Trainer so it can use one-step LoftQ and print the trainable parameters.

## TRL kernels and memory

The Transformers guide on [attention backends](https://huggingface.co/docs/transformers/attention_interface) states that SDPA automatically picks the CUDA kernel and that the Kernel Hub can load FlashAttention without installing the native package. The fast presets use `kernels-community/flash-attn2`; SDPA is the fallback.

[TRL Reducing Memory Usage](https://huggingface.co/docs/trl/reducing_memory_usage) covers PEFT, Liger, chunked cross-entropy, padding-free, activation offload and checkpointing. Points carried over into the presets:

- BFD packing implies padding-free and requires FlashAttention 2/3;
- chunked NLL and Liger fused CE are alternatives;
- DPO with Liger does not support reference log-prob precompute;
- activation offload saves VRAM at the cost of CPU/GPU transfers.

[Liger integration](https://huggingface.co/docs/trl/liger_kernel_integration) declares support for SFT, DPO and GRPO and publishes benchmarks of up to +20% throughput/-60% memory on different hardware/configurations. These figures are hypotheses to benchmark, not promises for Ada mobile.

## SFT and alignment

[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) supports conversational datasets, assistant-only loss, packing, PEFT and memory-efficient loss. UltraChat 200K declares an MIT license in its [dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k).

The [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) paper removes the separate reward model of the classic PPO pipeline. [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer) supports adapters and reference log-prob precompute; UltraFeedback Binarized declares MIT in its [dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized). DPO moves the model closer to the dataset's preferences; it does not guarantee safety or factuality.

## Reasoning

[OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) declares Apache-2.0 and 220K problems with multiple DeepSeek-R1 traces, at least one correct per problem, verified mostly with Math Verify. The card points to the ~94K `default` subset as best for SFT; the laptop initially uses 20K of it.

[GSM8K](https://arxiv.org/abs/2110.14168) provides multi-step problems and a held-out test, with an MIT dataset card. It is used for verifiable numeric reward and evaluation, keeping the test out of training.

[DeepSeekMath](https://arxiv.org/abs/2402.03300) introduces GRPO for math reasoning; [DeepSeek-R1](https://arxiv.org/abs/2501.12948) shows an SFT/RL curriculum at much larger scale. [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) documents:

- `beta=0` as the default: no reference model loaded;
- DAPO and masking of truncated completions;
- Transformers continuous batching as a single-GPU option, useful mainly at N>=32;
- `frac_reward_zero_std` to detect prompts with no relative signal.

At N=4 the project does not enable continuous batching by default and does not use colocated vLLM: the KV cache/engine would compete with the backward pass in the same VRAM.

## Intentional exclusions

- QLoRA does not replace random-init pretraining: it adapts a frozen base.
- Full PPO/RLHF would require policy/reference/reward models and more memory.
- DeepSpeed/FSDP do not magically reduce memory on a single GPU without paying for CPU offload/complexity.
- Full-parameter APOLLO/GaLore remain future ablations: they change the optimization and require convergence validation; they are not a free switch.
- `torch.compile`, DoRA, activation offload and entropy-token masking are exposed but not added to the defaults without a benchmark.
