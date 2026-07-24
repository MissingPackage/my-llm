"""QLoRA/PEFT helpers shared by SFT, DPO, GRPO and local inference.

The recommended laptop path freezes a 4-bit NF4 base model and trains small BF16
LoRA matrices.  Centralising that recipe prevents subtle stage-to-stage drift (for
example, forgetting double quantization in DPO or targeting fewer layers in GRPO).
Heavy imports stay inside functions so the core config/tests work without CUDA.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

DEFAULT_ADAPTER_SETTINGS: dict[str, Any] = {
    "enabled": False,
    "quantization": "nf4",
    "double_quant": True,
    "compute_dtype": "bf16",
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",
    "use_rslora": True,
    "use_dora": False,
    "loftq": False,
    "ephemeral_gpu_offload": False,
}


def adapter_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return validated PEFT settings with documented defaults filled in."""

    settings = {**DEFAULT_ADAPTER_SETTINGS, **config.get("peft", {})}
    allowed = set(DEFAULT_ADAPTER_SETTINGS)
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"Unknown PEFT options: {sorted(unknown)}")
    if settings["quantization"] not in {"nf4", "none"}:
        raise ValueError("peft.quantization must be 'nf4' or 'none'")
    if settings["compute_dtype"] not in {"bf16", "fp32"}:
        raise ValueError("peft.compute_dtype must be 'bf16' or 'fp32'")
    if int(settings["lora_r"]) < 1 or int(settings["lora_alpha"]) < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0.0 <= float(settings["lora_dropout"]) < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")
    if settings["loftq"] and settings["quantization"] != "nf4":
        raise ValueError("The one-step LoftQ initializer requires NF4 QLoRA")
    if settings["loftq"] and config.get("adapter_path"):
        raise ValueError("LoftQ initializes a new adapter; it cannot replace a loaded adapter")
    return settings


def lora_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    """Translate our small YAML vocabulary into ``peft.LoraConfig`` arguments."""

    return {
        "r": int(settings["lora_r"]),
        "lora_alpha": int(settings["lora_alpha"]),
        "lora_dropout": float(settings["lora_dropout"]),
        "target_modules": settings["target_modules"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        # rsLoRA scales by alpha/sqrt(rank), which remains stable at the useful
        # rank-32/64 regime better than original alpha/r scaling.
        "use_rslora": bool(settings["use_rslora"]),
        # DoRA can improve low-rank quality but adds magnitude parameters and
        # runtime overhead; it is exposed, not silently combined with every preset.
        "use_dora": bool(settings["use_dora"]),
    }


def _dtype(name: str, torch: Any) -> Any:
    """Map a YAML dtype name to a torch dtype without importing torch globally."""

    return torch.bfloat16 if name == "bf16" else torch.float32


def build_quantization_config(settings: dict[str, Any], torch: Any) -> Any | None:
    """Build the canonical QLoRA NF4 + double-quantization configuration."""

    if settings["quantization"] == "none":
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - exercised by installed CLI
        raise RuntimeError("Install the training extra before using QLoRA") from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        # NF4 places quantization levels according to a normal weight distribution
        # and is the QLoRA-recommended datatype for normally initialized weights.
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_dtype(settings["compute_dtype"], torch),
        # Nested quantization compresses the quantization constants as well, saving
        # roughly another 0.4 bits/parameter without changing compute precision.
        bnb_4bit_use_double_quant=bool(settings["double_quant"]),
    )


def _check_optional_kernels(training: dict[str, Any]) -> None:
    """Fail early with an actionable message for opt-in fast-kernel presets."""

    attention_backend = str(training.get("attention_backend", "sdpa"))
    if attention_backend.startswith("kernels-community/") and not importlib.util.find_spec(
        "kernels"
    ):
        raise RuntimeError(
            "This attention backend uses Hugging Face Hub kernels. "
            "Run: uv sync --extra train --extra fast"
        )
    if training.get("use_liger_kernel") and not importlib.util.find_spec("liger_kernel"):
        raise RuntimeError(
            "This preset enables Liger kernels. Run: uv sync --extra train --extra fast"
        )


def loftq_model_path(model_path: str) -> str | None:
    """Return the explicit base location LoftQ needs when the model is local.

    peft's ``replace_lora_weights_loftq`` resolves a bare ``model_path=None``
    through ``snapshot_download`` on the model's name, which treats a local
    directory as a Hub repo id and fails before ever touching the disk.  Hub
    ids keep the default resolution (their snapshot is already cached).
    """

    return model_path if Path(model_path).is_dir() else None


def load_training_model(config: dict[str, Any], torch: Any) -> Any:
    """Load either a normal model or a trainable QLoRA adapter on one GPU.

    For a new stage, ``model_path`` points at a base/merged checkpoint and no
    ``adapter_path`` is set.  For an experimental continuation, ``adapter_path``
    reloads an existing adapter as trainable.  The recommended DPO pipeline merges
    between stages so disabling the new adapter yields the correct frozen reference.
    """

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install training dependencies: uv sync --extra train") from exc

    training = config["training"]
    _check_optional_kernels(training)
    settings = adapter_settings(config)
    cuda = torch.cuda.is_available()
    attention_backend = training.get("attention_backend", "sdpa")
    gradient_checkpointing = bool(training.get("gradient_checkpointing", True))

    if not settings["enabled"]:
        dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            dtype=dtype,
            attn_implementation=attention_backend,
            low_cpu_mem_usage=True,
        )
        if cuda:
            model.to(torch.device("cuda"))
        return model

    if not cuda or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The QLoRA presets require a CUDA GPU with BF16 support")

    try:
        from peft import (
            LoraConfig,
            LoraRuntimeConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
            replace_lora_weights_loftq,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install PEFT/bitsandbytes: uv sync --extra train") from exc

    quantization_config = build_quantization_config(settings, torch)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        dtype=_dtype(settings["compute_dtype"], torch),
        quantization_config=quantization_config,
        device_map={"": torch.cuda.current_device()},
        attn_implementation=attention_backend,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if quantization_config is not None:
        # This freezes/casts the right layers, enables input gradients, and prepares
        # quantized modules for stable k-bit backpropagation.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    if config.get("adapter_path"):
        model = PeftModel.from_pretrained(
            model,
            config["adapter_path"],
            is_trainable=True,
        )
    else:
        kwargs = lora_kwargs(settings)
        if settings["use_dora"] or settings["ephemeral_gpu_offload"]:
            # Ephemeral offload is particularly useful for DoRA's temporary
            # magnitude computations; it trades PCIe traffic for a lower peak.
            kwargs["runtime_config"] = LoraRuntimeConfig(
                ephemeral_gpu_offload=bool(settings["ephemeral_gpu_offload"])
            )
        model = get_peft_model(model, LoraConfig(**kwargs))
        if settings["loftq"]:
            # One-step LoftQ adjusts the fresh LoRA matrices to compensate for NF4
            # quantization error.  The base must be available as safetensors.
            replace_lora_weights_loftq(
                model, model_path=loftq_model_path(str(config["model_path"]))
            )

    model.print_trainable_parameters()
    return model


def load_inference_model(
    model_path: str,
    *,
    adapter_path: str | None,
    load_in_4bit: bool,
    attention_backend: str,
    torch: Any,
) -> Any:
    """Load a merged model or a base+adapter pair for evaluation/chat."""

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "attn_implementation": attention_backend,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        if not cuda:
            raise RuntimeError("4-bit inference requires CUDA in this project")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if not load_in_4bit:
        model.to(torch.device("cuda" if cuda else "cpu"))
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.config.use_cache = True
    model.eval()
    return model
