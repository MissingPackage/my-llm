"""Reasoning prompt formats, answer extraction and deterministic GRPO rewards.

Reward code is intentionally small and auditable.  A learned judge can be gamed and
would consume additional VRAM; GSM8K instead permits an exact final-answer reward.
The weak format reward shapes parseable output but is weighted far below correctness.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

REASONING_SYSTEM_PROMPT = (
    "Solve the problem carefully. Put the reasoning trace between <think> and </think>, "
    "then put only the final result between <answer> and </answer>."
)

# Faithful Italian rendering of REASONING_SYSTEM_PROMPT: same tags, same two-part
# instruction.  Project language rule (revised by ruling E2, 2026-07-17): the ANSWER
# follows the language of the question; the reasoning trace is unconstrained (English
# is expected and fine — forcing Italian thinking costs accuracy per the literature
# reviewed in the bilingual-reasoning recon).  This variant matches the instruction
# language to Italian questions/examples; it does not mandate Italian reasoning.
REASONING_SYSTEM_PROMPT_IT = (
    "Risolvi il problema con attenzione. Metti la traccia di ragionamento tra <think> e "
    "</think>, poi metti solo il risultato finale tra <answer> e </answer>."
)

_REASONING_SYSTEM_PROMPTS = {
    "en": REASONING_SYSTEM_PROMPT,
    "it": REASONING_SYSTEM_PROMPT_IT,
}


def reasoning_system_prompt(lang: str) -> str:
    """Return the reasoning system prompt for a question language (``"en"`` or ``"it"``).

    Implements the revised project language rule (ruling E2, 2026-07-17): the answer
    follows the language of the question, the reasoning trace is unconstrained.
    Callers pass the language of the question and get the instruction in that same
    language.  Pre-existing call sites keep using :data:`REASONING_SYSTEM_PROMPT`
    directly and are unaffected.
    """

    try:
        return _REASONING_SYSTEM_PROMPTS[lang]
    except KeyError:
        valid = ", ".join(sorted(_REASONING_SYSTEM_PROMPTS))
        raise ValueError(
            f"Unsupported reasoning prompt language {lang!r}; valid: {valid}"
        ) from None

ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
STRICT_THINK_FORMAT = re.compile(
    r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)
# Keep accepting the first scaffold's ``<reasoning>`` tag so old checkpoints and
# smoke fixtures remain evaluable while the Qwen path uses its native ``<think>``.
STRICT_REASONING_FORMAT = re.compile(
    r"^\s*<reasoning>.+?</reasoning>\s*<answer>.+?</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?")
LATEX_FRACTION = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def split_gsm8k_solution(solution: str) -> tuple[str, str]:
    """Split GSM8K's ``reasoning #### final`` convention."""

    if "####" not in solution:
        return solution.strip(), extract_final_answer(solution) or ""
    reasoning, answer = solution.rsplit("####", 1)
    return reasoning.strip(), answer.strip()


def math_reasoning_messages(question: str, reasoning: str, answer: str) -> list[dict[str, str]]:
    """Create one consistently tagged conversational reasoning example."""

    return [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
        {
            "role": "assistant",
            "content": f"<think>{reasoning.strip()}</think>\n<answer>{answer.strip()}</answer>",
        },
    ]


def gsm8k_messages(question: str, solution: str) -> list[dict[str, str]]:
    """Convert a GSM8K supervised row to the project's conversational format."""

    reasoning, answer = split_gsm8k_solution(solution)
    return math_reasoning_messages(question, reasoning, answer)


def gsm8k_prompt(question: str) -> list[dict[str, str]]:
    """Create a generation prompt without leaking the reference solution."""

    return [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def completion_to_text(completion: Any) -> str:
    """Normalize TRL's standard or conversational completion representation."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        # Conversational completions may contain several messages; the last content
        # item is the assistant generation scored by the reward function.
        for item in reversed(completion):
            if isinstance(item, dict) and "content" in item:
                return str(item["content"])
        return "".join(str(item) for item in completion)
    return str(completion)


def _last_boxed(text: str) -> str | None:
    """Extract the last balanced ``\\boxed{...}``, including nested LaTeX braces."""

    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    depth = 1
    content_start = start + len(marker)
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index].strip()
    return None


def extract_final_answer(text: str) -> str | None:
    """Extract answer tags, GSM8K separators, boxed LaTeX, or the final number."""

    tagged = ANSWER_TAG.findall(text)
    if tagged:
        return tagged[-1].strip()
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed
    numbers = NUMBER.findall(text)
    return numbers[-1].strip() if numbers else None


def canonical_answer(value: str | None) -> Fraction | str | None:
    """Canonicalize common numeric spellings before an exact comparison."""

    if value is None:
        return None
    cleaned = value.strip().replace(",", "").replace("$", "")
    # Convert a simple LaTeX fraction to the same syntax accepted by Fraction.  The
    # fallback string path still handles symbolic/choice answers deterministically.
    cleaned = LATEX_FRACTION.sub(r"\1/\2", cleaned)
    cleaned = cleaned.replace("\\(", "").replace("\\)", "")
    numeric = NUMBER.fullmatch(cleaned)
    if numeric:
        try:
            if "/" in cleaned:
                numerator, denominator = cleaned.split("/", 1)
                return Fraction(Decimal(numerator.strip())) / Fraction(Decimal(denominator.strip()))
            return Fraction(Decimal(cleaned))
        except (InvalidOperation, ValueError, ZeroDivisionError):
            pass
    return " ".join(cleaned.lower().split())


def answers_equal(candidate: str | None, reference: str | None) -> bool:
    """Compare two non-empty canonical answers exactly."""

    left = canonical_answer(candidate)
    right = canonical_answer(reference)
    return left is not None and right is not None and left == right


def _expand_references(reference: Any, count: int) -> list[str]:
    """Align one reference per GRPO completion without assuming TRL batching shape."""

    if isinstance(reference, str):
        return [reference] * count
    values = list(reference)
    if not values:
        return [""] * count
    if len(values) == count:
        return [str(value) for value in values]
    repeats = (count + len(values) - 1) // len(values)
    return [str(value) for value in (values * repeats)[:count]]


def exact_answer_reward(completions: list[Any], answer: Any, **_: Any) -> list[float]:
    """Award 1 only when the extracted final answer equals the dataset reference."""

    references = _expand_references(answer, len(completions))
    return [
        1.0
        if answers_equal(
            extract_final_answer(completion_to_text(completion)),
            extract_final_answer(reference),
        )
        else 0.0
        for completion, reference in zip(completions, references, strict=True)
    ]


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    """Reward a complete reasoning+answer envelope, accepting both project tags."""

    rewards = []
    for completion in completions:
        text = completion_to_text(completion)
        valid = STRICT_THINK_FORMAT.fullmatch(text) or STRICT_REASONING_FORMAT.fullmatch(text)
        rewards.append(1.0 if valid else 0.0)
    return rewards
