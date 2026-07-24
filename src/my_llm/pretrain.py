"""Single-GPU pretraining loop for a randomly initialized causal language model.

This module intentionally keeps the loop explicit instead of hiding it behind a
high-level Trainer.  It is therefore possible to see where each byte goes: weights,
gradients, optimizer state, activations, input batches, and resumable RNG state.
The experimental 2B preset uses every *compatible* fit-oriented option here, while
the smaller presets keep more conservative FP32 parameters for training quality.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from my_llm.binary import TokenShardCorpus
from my_llm.config import PretrainConfig, load_typed
from my_llm.model import build_model, count_parameters


def seed_everything(seed: int, torch: Any) -> None:
    """Seed Python, NumPy and every CUDA device used by this single process."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_multiplier(step: int, *, warmup_steps: int, max_steps: int, min_ratio: float) -> float:
    """Linear warmup followed by cosine decay to a non-zero learning-rate floor."""

    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_ratio + (1.0 - min_ratio) * cosine


@contextmanager
def temporary_default_dtype(torch: Any, dtype: Any) -> Iterator[None]:
    """Construct modules directly in ``dtype`` without a full-size FP32 copy.

    Allocating 2.027B parameters in FP32 and converting them afterwards creates an
    avoidable ~8 GiB host-memory peak.  The context is process-global, so it is kept
    as small as possible and restored even when model initialization fails.
    """

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def tensor_batch(array: np.ndarray, *, device: Any, torch: Any) -> Any:
    """Move one NumPy token batch to the target device efficiently."""

    tensor = torch.from_numpy(array)
    if device.type == "cuda":
        # Page-locked host memory makes a truly asynchronous H2D copy possible.
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=device.type == "cuda")


class BatchProvider:
    """Sample token windows and optionally prefetch them on a dedicated CUDA stream.

    Prefetching advances the NumPy RNG ahead of the consumed batch.  To retain exact
    resume semantics, checkpoints persist both the advanced RNG state and the tiny
    queue of already-sampled CPU tensors.  This detail is easy to miss in otherwise
    "reproducible" training loops.
    """

    def __init__(
        self,
        corpus: TokenShardCorpus,
        *,
        batch_size: int,
        rng: np.random.Generator,
        device: Any,
        torch: Any,
        depth: int,
        initial_batches: list[Any] | None = None,
    ) -> None:
        self.corpus = corpus
        self.batch_size = batch_size
        self.rng = rng
        self.device = device
        self.torch = torch
        self.depth = depth if device.type == "cuda" else 0
        self.stream = torch.cuda.Stream(device=device) if self.depth else None
        self.queue: deque[tuple[Any, Any, Any]] = deque()

        # Restore queued batches before drawing new samples; the RNG in a checkpoint
        # already points *after* these batches.
        for host_batch in initial_batches or []:
            if len(self.queue) >= self.depth:
                break
            self._enqueue_host(host_batch)
        while len(self.queue) < self.depth:
            self._enqueue_sample()

    def _sample_host(self) -> Any:
        """Create one contiguous CPU tensor backed by a sampled NumPy array."""

        array = self.corpus.sample_numpy(self.batch_size, self.rng)
        host = self.torch.from_numpy(array)
        return host.pin_memory() if self.depth else host

    def _enqueue_host(self, host: Any) -> None:
        """Schedule a non-blocking H2D copy and remember its completion event."""

        if not self.depth or self.stream is None:
            return
        if not host.is_pinned():
            host = host.pin_memory()
        with self.torch.cuda.stream(self.stream):
            device_batch = host.to(self.device, non_blocking=True)
            ready = self.torch.cuda.Event()
            ready.record(self.stream)
        # Keeping ``host`` alive until ``ready`` completes prevents the pinned-memory
        # allocator from reusing it while DMA is still reading it.
        self.queue.append((host, device_batch, ready))

    def _enqueue_sample(self) -> None:
        self._enqueue_host(self._sample_host())

    def next(self) -> Any:
        """Return the next token batch and immediately schedule its replacement."""

        if not self.depth:
            return tensor_batch(
                self.corpus.sample_numpy(self.batch_size, self.rng),
                device=self.device,
                torch=self.torch,
            )

        host, device_batch, ready = self.queue.popleft()
        # In the steady state the previous forward/backward pass hides this wait.
        ready.synchronize()
        del host
        self._enqueue_sample()
        return device_batch

    def pending_for_checkpoint(self) -> list[Any]:
        """Clone queued CPU tensors so a resumed run consumes the exact same data."""

        pending = []
        for host, _, ready in self.queue:
            ready.synchronize()
            pending.append(host.detach().cpu().clone())
        return pending


def optimizer_parameter_groups(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    """Apply AdamW decay to matrices, but not to norms or scalar/vector biases."""

    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        # Matrix weights benefit from decoupled decay; RMSNorm scales do not.
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_optimizer(model: Any, config: PretrainConfig, torch: Any, device: Any) -> Any:
    """Create fused AdamW or a bitsandbytes 8-bit/paged variant.

    Paged AdamW is the 2B fit lever: 8-bit moments cut optimizer-state VRAM and
    unified-memory paging can absorb brief peaks.  Paging is a safety valve, not a
    speed feature; sustained page faults make training dramatically slower.
    """

    groups = optimizer_parameter_groups(model, config.training.weight_decay)
    kwargs = {
        "lr": config.training.learning_rate,
        "betas": (config.training.beta1, config.training.beta2),
        "eps": 1e-8,
    }
    name = config.training.optimizer
    if name in {"adamw_8bit", "paged_adamw_8bit"}:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:  # pragma: no cover - depends on the CUDA extra
            raise RuntimeError(
                "This preset needs bitsandbytes. Run: uv sync --extra train"
            ) from exc
        optimizer_class = (
            bnb.optim.PagedAdamW8bit if name == "paged_adamw_8bit" else bnb.optim.AdamW8bit
        )
        return optimizer_class(
            groups,
            min_8bit_size=config.training.optimizer_min_8bit_size,
            percentile_clipping=config.training.optimizer_percentile_clipping,
            **kwargs,
        )

    # Fused AdamW combines elementwise update kernels and is measurably faster on
    # CUDA.  Older/CPU PyTorch builds fall back cleanly to the reference optimizer.
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(groups, **kwargs)


def save_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    batch_provider: BatchProvider,
    step: int,
    output_dir: Path,
    config: PretrainConfig,
    train_rng: np.random.Generator,
    validation_rng: np.random.Generator,
    torch: Any,
) -> Path:
    """Atomically save weights plus enough state for a bitwise data-stream resume."""

    destination = output_dir / f"checkpoint-{step:06d}"
    temporary = output_dir / f".checkpoint-{step:06d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    # Safetensors avoids arbitrary code execution on model load.  Sharding keeps
    # individual files manageable on common laptop filesystems and cloud storage.
    model.save_pretrained(
        temporary,
        safe_serialization=True,
        max_shard_size=config.training.max_shard_size,
    )
    tokenizer.save_pretrained(temporary)
    state = {
        "step": step,
        "optimizer": (optimizer.state_dict() if config.training.save_optimizer_state else None),
        "scheduler": scheduler.state_dict(),
        "train_rng_state": train_rng.bit_generator.state,
        "validation_rng_state": validation_rng.bit_generator.state,
        "prefetched_batches": batch_provider.pending_for_checkpoint(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, temporary / "training-state.pt")
    (temporary / "run-config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")

    # A rename on one filesystem is atomic: an interrupted save leaves the prior
    # checkpoint intact instead of a directory containing half-written tensors.
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    return destination


def prune_checkpoints(output_dir: Path, keep: int) -> None:
    """Delete only the oldest generated checkpoints, never arbitrary directories."""

    checkpoints = sorted(path for path in output_dir.glob("checkpoint-*") if path.is_dir())
    for path in checkpoints[:-keep]:
        shutil.rmtree(path)


def load_training_state(path: Path, torch: Any) -> dict[str, Any]:
    """Load our trusted local training state across supported PyTorch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location="cpu")


def evaluate(
    model: Any,
    corpus: TokenShardCorpus,
    rng: np.random.Generator,
    *,
    batches: int,
    batch_size: int,
    device: Any,
    use_bf16: bool,
    torch: Any,
) -> float:
    """Estimate held-out next-token loss without retaining gradients."""

    model.eval()
    losses = []
    with torch.inference_mode():
        for _ in range(batches):
            input_ids = tensor_batch(
                corpus.sample_numpy(batch_size, rng), device=device, torch=torch
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                loss = model(input_ids=input_ids, labels=input_ids).loss
            losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def _parameter_dtype(config: PretrainConfig, torch: Any) -> Any:
    """Translate the human-readable YAML dtype to a PyTorch dtype."""

    return torch.bfloat16 if config.training.parameter_dtype == "bf16" else torch.float32


def _load_or_build_model(
    config: PretrainConfig,
    tokenizer: Any,
    *,
    resume: Path | None,
    device: Any,
    torch: Any,
    auto_model_class: Any,
) -> Any:
    """Load a sharded checkpoint efficiently or initialize directly on the GPU."""

    dtype = _parameter_dtype(config, torch)
    if resume is not None:
        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "attn_implementation": config.training.attention_backend,
            "low_cpu_mem_usage": True,
        }
        if device.type == "cuda":
            load_kwargs["device_map"] = {"": device.index or 0}
        model = auto_model_class.from_pretrained(resume, **load_kwargs)
        if device.type != "cuda":
            model.to(device)
        return model

    # Direct device construction avoids holding both CPU and GPU copies.  Direct
    # BF16 construction is mandatory for the 2B fit profile's memory envelope.
    with torch.device(device), temporary_default_dtype(torch, dtype):
        return build_model(
            config.model,
            tokenizer,
            attention_backend=config.training.attention_backend,
        )


def train(
    config: PretrainConfig,
    *,
    resume: Path | None = None,
    max_steps_override: int | None = None,
    requested_device: str = "auto",
) -> Path:
    """Run pretraining and return the final Transformers-compatible directory."""

    try:
        import torch
        from torch.utils.tensorboard import SummaryWriter
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised by the installed CLI
        raise RuntimeError("Install training dependencies: uv sync --extra train") from exc

    seed_everything(config.seed, torch)
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)

    use_bf16 = config.training.precision == "bf16"
    if use_bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("This config requires a CUDA GPU with BF16 support")
    if config.training.parameter_dtype == "bf16" and not use_bf16:
        raise ValueError("BF16 parameters require precision: bf16")
    if config.training.optimizer != "adamw_fused" and device.type != "cuda":
        raise RuntimeError("bitsandbytes pretraining optimizers require CUDA")

    if device.type == "cuda":
        # TF32 accelerates any remaining FP32 GEMMs on Ada without changing tensor
        # storage.  SDPA independently chooses a flash/memory-efficient kernel.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision("high")

    tokenizer_source = resume if resume is not None else config.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    model = _load_or_build_model(
        config,
        tokenizer,
        resume=resume,
        device=device,
        torch=torch,
        auto_model_class=AutoModelForCausalLM,
    )
    model.config.use_cache = False
    if config.training.gradient_checkpointing:
        # Non-reentrant checkpointing is the current PyTorch path and handles modern
        # model graphs better than the legacy reentrant implementation.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_model = model
    if config.training.compile:
        # Compilation can improve a long, fixed-shape run but costs startup time and
        # extra memory.  It remains opt-in because some bitsandbytes/kernel mixes
        # graph-break frequently on consumer CUDA installations.
        train_model = torch.compile(
            model,
            mode=config.training.compile_mode,
            dynamic=False,
        )

    optimizer = make_optimizer(model, config, torch, device)
    max_steps = max_steps_override or config.training.max_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_multiplier(
            step,
            warmup_steps=config.training.warmup_steps,
            max_steps=max_steps,
            min_ratio=config.training.min_lr_ratio,
        ),
    )

    # Memory maps keep the multi-billion-token corpus off RAM and sample windows
    # without copying entire shard files.
    train_corpus = TokenShardCorpus(
        config.train_manifest, sequence_length=config.training.sequence_length
    )
    validation_corpus = TokenShardCorpus(
        config.validation_manifest, sequence_length=config.training.sequence_length
    )
    train_rng = np.random.default_rng(config.seed)
    validation_rng = np.random.default_rng(config.seed + 1)
    start_step = 0
    prefetched_batches: list[Any] = []

    if resume is not None:
        state = load_training_state(resume / "training-state.pt", torch)
        if state.get("optimizer") is None:
            raise RuntimeError(
                "This weights-only checkpoint cannot resume exactly; start a new run from it."
            )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        train_rng.bit_generator.state = state["train_rng_state"]
        validation_rng.bit_generator.state = state["validation_rng_state"]
        prefetched_batches = list(state.get("prefetched_batches", []))
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        start_step = int(state["step"])

    batch_provider = BatchProvider(
        train_corpus,
        batch_size=config.training.micro_batch_size,
        rng=train_rng,
        device=device,
        torch=torch,
        depth=config.training.prefetch_batches,
        initial_batches=prefetched_batches,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(config.log_dir)
    parameter_count = count_parameters(model)
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": parameter_count,
                "parameter_dtype": str(next(model.parameters()).dtype),
                "optimizer": config.training.optimizer,
                "attention_backend": config.training.attention_backend,
                "start_step": start_step,
                "max_steps": max_steps,
                "tokens_per_step": config.training.tokens_per_step,
            },
            indent=2,
        )
    )

    train_model.train()
    interval_start = time.perf_counter()
    interval_tokens = 0
    try:
        for step in range(start_step + 1, max_steps + 1):
            # ``set_to_none`` avoids a full gradient-buffer memset and lets PyTorch
            # allocate gradients lazily on the first backward pass.
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(config.training.gradient_accumulation_steps):
                input_ids = batch_provider.next()
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    loss = train_model(input_ids=input_ids, labels=input_ids).loss
                    scaled_loss = loss / config.training.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
                scaled_loss.backward()
                accumulated_loss += float(loss.detach().cpu())

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.grad_clip
            )
            optimizer.step()
            scheduler.step()
            interval_tokens += config.training.tokens_per_step

            if step % config.training.log_every == 0 or step == 1:
                elapsed = max(time.perf_counter() - interval_start, 1e-9)
                tokens_per_second = interval_tokens / elapsed
                mean_loss = accumulated_loss / config.training.gradient_accumulation_steps
                lr = scheduler.get_last_lr()[0]
                metrics = {
                    "step": step,
                    "loss": round(mean_loss, 5),
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tokens_per_second": round(tokens_per_second, 1),
                    "tokens_seen": step * config.training.tokens_per_step,
                }
                if device.type == "cuda":
                    metrics["max_vram_gib"] = round(
                        torch.cuda.max_memory_allocated(device) / 1024**3, 2
                    )
                    metrics["reserved_vram_gib"] = round(
                        torch.cuda.memory_reserved(device) / 1024**3, 2
                    )
                print(json.dumps(metrics))
                writer.add_scalar("train/loss", mean_loss, step)
                writer.add_scalar("train/learning_rate", lr, step)
                writer.add_scalar("train/tokens_per_second", tokens_per_second, step)
                writer.add_scalar("train/grad_norm", float(grad_norm), step)
                interval_start = time.perf_counter()
                interval_tokens = 0

            if step % config.training.eval_every == 0 or step == max_steps:
                validation_loss = evaluate(
                    train_model,
                    validation_corpus,
                    validation_rng,
                    batches=config.training.eval_batches,
                    batch_size=config.training.micro_batch_size,
                    device=device,
                    use_bf16=use_bf16,
                    torch=torch,
                )
                # Clamp only the exponential to keep pathological early-run losses
                # from overflowing; the logged loss remains exact.
                perplexity = math.exp(min(validation_loss, 20.0))
                print(
                    json.dumps(
                        {
                            "step": step,
                            "validation_loss": validation_loss,
                            "validation_perplexity": perplexity,
                        }
                    )
                )
                writer.add_scalar("validation/loss", validation_loss, step)
                writer.add_scalar("validation/perplexity", perplexity, step)

            if config.training.save_checkpoints and (
                step % config.training.save_every == 0 or step == max_steps
            ):
                checkpoint = save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    batch_provider=batch_provider,
                    step=step,
                    output_dir=config.output_dir,
                    config=config,
                    train_rng=train_rng,
                    validation_rng=validation_rng,
                    torch=torch,
                )
                print(f"Saved {checkpoint}")
                prune_checkpoints(config.output_dir, config.training.keep_last_checkpoints)

            if (
                device.type == "cuda"
                and config.training.empty_cache_every
                and step % config.training.empty_cache_every == 0
            ):
                # This can tame allocator fragmentation, but synchronizes the device;
                # therefore zero (disabled) is the faster default.
                torch.cuda.empty_cache()
    except torch.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            "CUDA OOM: lower sequence_length first, then micro_batch_size; raise "
            "gradient_accumulation_steps to retain the effective token batch."
        )
        raise exc
    finally:
        writer.close()

    final_dir = config.output_dir / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True)
    model.save_pretrained(
        final_dir,
        safe_serialization=True,
        max_shard_size=config.training.max_shard_size,
    )
    tokenizer.save_pretrained(final_dir)
    (final_dir / "run-config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return final_dir


def main() -> None:
    """Parse CLI arguments and launch one pretraining experiment."""

    parser = argparse.ArgumentParser(description="Pretrain a Llama-style causal LM from scratch.")
    parser.add_argument("config")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_typed(args.config, PretrainConfig)
    output = train(
        config,
        resume=args.resume,
        max_steps_override=args.max_steps,
        requested_device=args.device,
    )
    print(f"Final model saved to {output}")


if __name__ == "__main__":
    main()
