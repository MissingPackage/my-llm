"""Minimal local chat loop for merged checkpoints or base+LoRA pairs."""

from __future__ import annotations

import argparse

from my_llm.adapters import load_inference_model


def main() -> None:
    """Load one local/Hugging Face model and answer interactively or once."""

    parser = argparse.ArgumentParser(description="Chat with a local checkpoint or adapter.")
    parser.add_argument("model_path", help="Merged checkpoint or original base model")
    parser.add_argument("--adapter", help="Optional PEFT adapter loaded on top of model_path")
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Quantize the base for low-VRAM chat"
    )
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--prompt")
    parser.add_argument("--system", default="You are a concise and helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install training dependencies: uv sync --extra train") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_inference_model(
        args.model_path,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
        attention_backend=args.attention_backend,
        torch=torch,
    )
    device = next(model.parameters()).device

    def answer(prompt: str) -> str:
        """Render the model's chat template, generate, and decode only new tokens."""

        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": prompt},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        generation = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation.update({"temperature": args.temperature, "top_p": 0.95})
        with torch.inference_mode():
            output = model.generate(**encoded, **generation)[0]
        input_width = encoded["input_ids"].shape[1]
        return tokenizer.decode(output[input_width:], skip_special_tokens=True)

    if args.prompt is not None:
        print(answer(args.prompt))
        return
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt or prompt.lower() in {"exit", "quit"}:
            return
        print(f"model> {answer(prompt)}")


if __name__ == "__main__":
    main()
