"""Merge a LoRA adapter into its unquantized base checkpoint.

Merging between stages costs disk space but makes DPO semantics unambiguous: the
merged SFT model is the frozen reference, while a new DPO adapter is the policy.
The operation defaults to CPU so it does not compete with a 16 GiB GPU; expect a
temporary host-RAM footprint larger than the final BF16 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def merge_adapter(
    base_model: str,
    adapter_path: str,
    output_dir: Path,
    *,
    dtype_name: str,
    device_name: str,
    max_shard_size: str,
) -> Path:
    """Merge ``adapter_path`` into ``base_model`` and save portable safetensors."""

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install training dependencies: uv sync --extra train") from exc

    if output_dir.resolve() in {Path(adapter_path).resolve(), Path(base_model).resolve()}:
        raise ValueError("The merge output must not overwrite its base or adapter")
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    device = torch.device(device_name)
    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = {"": device.index or 0}

    # Merge into an ordinary BF16/FP32 model.  Merging directly into NF4 weights
    # would bake quantization error into every subsequent training stage.
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    if device.type != "cuda":
        model.to(device)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    merged = model.merge_and_unload(progressbar=True, safe_merge=True)

    temporary = output_dir.with_name(f".{output_dir.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    merged.save_pretrained(
        temporary,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    # Post-training may patch a base chat template or EOS token.  Prefer the
    # tokenizer saved beside the adapter, then fall back to the unchanged base.
    adapter_tokenizer = Path(adapter_path) / "tokenizer_config.json"
    tokenizer_source = adapter_path if adapter_tokenizer.exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    tokenizer.save_pretrained(temporary)
    (temporary / "merge-manifest.json").write_text(
        json.dumps(
            {
                "base_model": base_model,
                "adapter_path": adapter_path,
                "dtype": dtype_name,
                "device": device_name,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    temporary.replace(output_dir)
    return output_dir


def main() -> None:
    """Parse merge arguments and create a standalone stage checkpoint."""

    parser = argparse.ArgumentParser(description="Merge a PEFT adapter into its base model.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-shard-size", default="2GB")
    args = parser.parse_args()
    output = merge_adapter(
        args.base_model,
        args.adapter,
        args.output,
        dtype_name=args.dtype,
        device_name=args.device,
        max_shard_size=args.max_shard_size,
    )
    print(f"Merged model saved to {output}")


if __name__ == "__main__":
    main()
