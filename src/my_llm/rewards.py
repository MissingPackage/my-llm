"""Custom GRPO rewards for reasoning style: self-verification and concise answers.

Like the rewards in :mod:`my_llm.reasoning`, these are pure, deterministic and small
enough to audit line by line — no judge model, no I/O, no state.  Both are *shape*
rewards: a regex can demand that the surface form of a behaviour is present, but it
cannot certify that the behaviour is genuine (a syntactically valid verification may
still be arithmetically wrong or copied verbatim from the prompt).  That residual
hacking risk is contained structurally, not here: the exact-answer reward stays
dominant and these rewards enter the mix with weight <= 0.2 (goal contract), so a
policy that games the form without solving the problem still loses.

Each heuristic is designed so that the *cheapest* hacks pay nothing: bare markers,
empty answers, markers outside the reasoning block, or a short answer without the
full envelope all score 0.  The remaining exploits require producing text that is
at least superficially indistinguishable from the desired behaviour.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from my_llm.reasoning import (
    ANSWER_TAG,
    STRICT_REASONING_FORMAT,
    STRICT_THINK_FORMAT,
    completion_to_text,
    exact_answer_reward,
    format_reward,
)

# Both project reasoning tags are accepted, mirroring ``format_reward``: Qwen paths
# emit ``<think>`` while old checkpoints and smoke fixtures still use ``<reasoning>``.
THINK_BLOCK = re.compile(r"<(think|reasoning)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)

# A deliberate, colon-terminated marker.  The colon is required so that incidental
# prose ("I check the fridge") does not count as announcing a verification step.
VERIFICATION_MARKER = re.compile(r"\b(?:verifica|controllo|check|verify)\s*:", re.IGNORECASE)

# How far past the marker the computational evidence must appear.  A narrow window
# forces the substance to sit next to the announcement instead of anywhere later.
VERIFICATION_WINDOW_CHARS = 100

_WINDOW_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
# ASCII-only on purpose (ruff RUF001): a real numeric check virtually always carries
# ``=`` or an ASCII operator even when the model also writes Unicode multiply/divide.
_WINDOW_OPERATION = re.compile(r"[=+*/<>%-]")

DEFAULT_MAX_ANSWER_CHARS = 80


def _has_genuine_verification(think_text: str) -> bool:
    """True if any marker is followed closely by numbers plus an operation/comparison."""

    for marker in VERIFICATION_MARKER.finditer(think_text):
        window = think_text[marker.end() : marker.end() + VERIFICATION_WINDOW_CHARS]
        if len(_WINDOW_NUMBER.findall(window)) >= 2 and _WINDOW_OPERATION.search(window):
            return True
    return False


def verification_step_reward(completions: list[Any], **_: Any) -> list[float]:
    """Award 1 when the reasoning block contains a genuine-looking self-verification step.

    Rationale: on GSM8K-style tasks, models that re-derive or cross-check their result
    before answering are more reliable, but the exact-answer reward alone never pays for
    the checking behaviour itself.  This reward shapes it explicitly: a verification
    marker (``verifica:``, ``controllo:``, ``check:``, ``verify:``) inside the
    ``<think>``/``<reasoning>`` block must be followed, within a narrow window, by real
    computational content — at least two numbers and an arithmetic or comparison symbol.

    Anti-hacking design (cheap exploits score 0):
    - bare marker without substance (``verifica: ok``) — no numbers in the window;
    - repeated markers with no content — each window is checked independently;
    - a lone restated result (``check: 42``) — a single number is not a comparison;
    - marker outside the reasoning block, or no block at all — only tagged blocks
      are scanned.

    Honest limits: this is a *form* check, not a semantic one.  A regex cannot verify
    that the arithmetic is correct, relevant to the problem, or not copied from the
    prompt — ``verifica: 1+1=2`` appended to every trace would pay.  The reward is
    binary (repetition does not stack) and the goal contract caps its weight at 0.2
    under a dominant exact-answer reward, so faking the form without solving the task
    remains a losing strategy.
    """

    rewards = []
    for completion in completions:
        text = completion_to_text(completion)
        genuine = any(
            _has_genuine_verification(block.group(2)) for block in THINK_BLOCK.finditer(text)
        )
        rewards.append(1.0 if genuine else 0.0)
    return rewards


def _concise_scores(completions: list[Any], max_chars: int) -> list[float]:
    """Score the full envelope plus a non-empty answer within the length budget."""

    rewards = []
    for completion in completions:
        text = completion_to_text(completion)
        if not (STRICT_THINK_FORMAT.fullmatch(text) or STRICT_REASONING_FORMAT.fullmatch(text)):
            rewards.append(0.0)
            continue
        answers = ANSWER_TAG.findall(text)
        content = answers[-1].strip() if answers else ""
        rewards.append(1.0 if 0 < len(content) <= max_chars else 0.0)
    return rewards


def concise_answer_reward(completions: list[Any], **_: Any) -> list[float]:
    """Award 1 when the final answer is present, non-empty and at most 80 characters.

    Rationale: the ``<answer>`` block should carry only the final result — verbose
    answers duplicate the reasoning trace, hurt exact-answer extraction, and waste
    completion tokens.  This reward pays for a tight answer, but only inside the full
    ``<think>``/``<reasoning>`` + ``<answer>`` envelope: brevity alone must not pay,
    otherwise the degenerate policy "skip reasoning, emit a short guess" would collect
    it for free.

    Anti-hacking design (cheap exploits score 0):
    - empty or whitespace-only answer — trivially concise, worth nothing;
    - missing ``<answer>`` tag — nothing to score;
    - short answer without the reasoning block — malformed envelope is rejected.

    Honest limits: length is a proxy, not a measure of quality — a wrong or vacuous
    short answer still collects this reward, and 80 characters is a heuristic budget
    (parametrizable via :func:`make_concise_answer_reward`).  Correctness is the
    dominant exact-answer reward's job; this one only shapes the output surface and
    is capped at weight <= 0.2 by the goal contract.
    """

    return _concise_scores(completions, DEFAULT_MAX_ANSWER_CHARS)


def make_concise_answer_reward(
    max_chars: int = DEFAULT_MAX_ANSWER_CHARS,
) -> Callable[..., list[float]]:
    """Build a TRL-compatible concise-answer reward with a custom length budget.

    The default export keeps the standard signature; this factory exists so a YAML
    config (phase 2) can tune the budget without a new function per threshold.  The
    returned closure gets a distinctive ``__name__`` because TRL logs reward metrics
    under the function name.
    """

    def reward(completions: list[Any], **_: Any) -> list[float]:
        return _concise_scores(completions, max_chars)

    reward.__name__ = f"concise_answer_reward_{max_chars}"
    return reward


# --- YAML reward selection (grpo-rewards, phase 2) --------------------------------

REWARD_REGISTRY: dict[str, Callable[..., list[float]]] = {
    "exact_answer": exact_answer_reward,
    "format": format_reward,
    "verification_step": verification_step_reward,
    "concise_answer": concise_answer_reward,
}

# The goal contract caps the custom shape rewards at weight 0.2 so the exact-answer
# reward stays dominant.  Raising the cap requires a recorded decision, never a config
# change, which is why there is deliberately no bypass flag here.
CUSTOM_REWARD_NAMES = frozenset({"verification_step", "concise_answer"})
MAX_CUSTOM_REWARD_WEIGHT = 0.2


def resolve_reward_config(
    training: dict[str, Any],
) -> tuple[list[str], list[Callable[..., list[float]]], list[float]]:
    """Resolve GRPO reward functions and weights from the ``training`` config dict.

    Without a ``rewards`` key the historical behaviour is preserved exactly:
    ``[exact_answer, format]`` with weights taken verbatim from
    ``training["reward_weights"]`` — existing configs run byte-identically.

    With ``rewards: [{name: str, weight: float}, ...]`` the functions and weights
    derive from the list, validated against the goal contract:

    - names must come from :data:`REWARD_REGISTRY`, without duplicates;
    - custom shape rewards (:data:`CUSTOM_REWARD_NAMES`) are capped at
      :data:`MAX_CUSTOM_REWARD_WEIGHT`;
    - ``exact_answer`` must be present and strictly heavier than every other
      reward, so correctness always dominates shape;
    - ``reward_weights`` must be absent — two sources of truth for weights would
      let one silently override the other.

    Pure and importable without the training stack, so core CI can test it.
    """

    entries = training.get("rewards")
    if entries is None:
        return (
            ["exact_answer", "format"],
            [exact_answer_reward, format_reward],
            training["reward_weights"],
        )
    if "reward_weights" in training:
        raise ValueError(
            "training.rewards and training.reward_weights are mutually exclusive: "
            "with a rewards list, each entry carries its own weight — remove reward_weights"
        )
    if not isinstance(entries, list) or not entries:
        raise ValueError("training.rewards must be a non-empty list of {name, weight} entries")

    names: list[str] = []
    weights: list[float] = []
    for entry in entries:
        if not isinstance(entry, dict) or "name" not in entry or "weight" not in entry:
            raise ValueError(
                f"Each training.rewards entry needs 'name' and 'weight' keys, got: {entry!r}"
            )
        name = entry["name"]
        if name not in REWARD_REGISTRY:
            known = ", ".join(sorted(REWARD_REGISTRY))
            raise ValueError(f"Unknown reward {name!r}; valid names: {known}")
        if name in names:
            raise ValueError(f"Duplicate reward {name!r} in training.rewards")
        weight = float(entry["weight"])
        if weight <= 0:
            raise ValueError(f"Reward {name!r} weight must be positive, got {weight}")
        if name in CUSTOM_REWARD_NAMES and weight > MAX_CUSTOM_REWARD_WEIGHT:
            raise ValueError(
                f"Reward {name!r} weight {weight} exceeds the goal-contract cap of "
                f"{MAX_CUSTOM_REWARD_WEIGHT} for custom shape rewards; raising it takes a "
                "recorded decision, not a config change"
            )
        names.append(name)
        weights.append(weight)

    if "exact_answer" not in names:
        raise ValueError(
            "training.rewards must include 'exact_answer': the goal contract keeps the "
            "correctness reward dominant over all shape rewards"
        )
    exact_weight = weights[names.index("exact_answer")]
    for name, weight in zip(names, weights, strict=True):
        if name != "exact_answer" and weight >= exact_weight:
            raise ValueError(
                f"Reward {name!r} weight {weight} is not strictly below exact_answer's "
                f"{exact_weight}: the goal contract requires exact_answer to dominate"
            )
    return names, [REWARD_REGISTRY[name] for name in names], weights
