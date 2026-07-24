"""Environment doctor for the single-GPU training presets."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import sys


def _installed_version(package: str) -> str | None:
    """Return a distribution version without importing its CUDA extensions."""

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_hardware() -> tuple[dict[str, object], list[str]]:
    """Collect actionable CUDA/VRAM/package facts and human-readable warnings."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing. Run: uv sync --extra train") from exc

    report: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "packages": {
            package: _installed_version(package)
            for package in (
                "transformers",
                "trl",
                "peft",
                "bitsandbytes",
                "liger-kernel",
                "kernels",
            )
        },
    }
    warnings = []
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        total_gib = properties.total_memory / 1024**3
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        report.update(
            {
                "device": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "vram_gib": round(total_gib, 2),
                "currently_free_vram_gib": round(free_bytes / 1024**3, 2),
                "cuda_runtime": torch.version.cuda,
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "tf32_supported": properties.major >= 8,
                "total_vram_bytes": total_bytes,
            }
        )
        if total_gib < 15.0:
            warnings.append("Less than 15 GiB usable VRAM detected; start QLoRA at 1024 tokens.")
        if not torch.cuda.is_bf16_supported():
            warnings.append("BF16 is unavailable; the 2B presets cannot run unchanged.")
        if properties.major < 8:
            warnings.append(
                "Compute capability < 8.0: BF16/modern flash kernels may be unavailable."
            )
    else:
        warnings.append("CUDA is unavailable. Only debug FP32 tests can run on CPU.")

    if not importlib.util.find_spec("bitsandbytes"):
        warnings.append("bitsandbytes is missing: QLoRA and the 2B fit optimizer are unavailable.")
    if not importlib.util.find_spec("peft"):
        warnings.append("PEFT is missing: LoRA adapters cannot be trained or merged.")
    if not importlib.util.find_spec("kernels") or not importlib.util.find_spec("liger_kernel"):
        warnings.append(
            "Optional fast kernels are missing; install with: uv sync --extra train --extra fast"
        )
    return report, warnings


def main() -> None:
    """Print the machine report as JSON and warnings on stderr."""

    try:
        report, warnings = inspect_hardware()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(report, indent=2))
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
