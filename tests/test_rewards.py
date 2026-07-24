"""Anti-hacking tests for the custom GRPO shape rewards (grpo-rewards, phases 1-2)."""

from pathlib import Path

import pytest

from my_llm.config import load_yaml
from my_llm.reasoning import exact_answer_reward, format_reward
from my_llm.rewards import (
    concise_answer_reward,
    make_concise_answer_reward,
    resolve_reward_config,
    verification_step_reward,
)

# --- verification_step_reward -----------------------------------------------------

VERIFIED_IT = (
    "<think>12 per 3 fa 36. verifica: 36 / 3 = 12, il conto torna.</think>"
    "<answer>36</answer>"
)
VERIFIED_EN = (
    "<think>20 plus 21 gives 41. check: 20 + 21 = 41, matches the result.</think>"
    "<answer>41</answer>"
)


def test_verification_positive_it_and_en() -> None:
    assert verification_step_reward([VERIFIED_IT, VERIFIED_EN]) == [1.0, 1.0]


def test_verification_accepts_legacy_reasoning_tag_and_controllo() -> None:
    completion = (
        "<reasoning>7 * 6 = 42. controllo: 42 / 6 = 7, coerente.</reasoning>"
        "<answer>42</answer>"
    )
    assert verification_step_reward([completion]) == [1.0]


def test_verification_negative_no_marker() -> None:
    # Correct reasoning but no announced verification step: nothing to reward.
    completion = "<think>2 + 2 = 4, so the answer is 4.</think><answer>4</answer>"
    assert verification_step_reward([completion]) == [0.0]


def test_verification_negative_marker_without_computation_nearby() -> None:
    # Marker followed only by prose within the window: no numbers, no comparison.
    completion = (
        "<think>The total is 36. verify: everything looks consistent to me.</think>"
        "<answer>36</answer>"
    )
    assert verification_step_reward([completion]) == [0.0]


def test_verification_hack_bare_marker_pays_nothing() -> None:
    # HACKING CASE: bare "verifica: ok" — the announced check has no substance.
    completion = "<think>12 per 3 fa 36. verifica: ok</think><answer>36</answer>"
    assert verification_step_reward([completion]) == [0.0]


def test_verification_hack_repeated_empty_markers_pay_nothing() -> None:
    # HACKING CASE: spamming markers without content — each window is empty of math.
    completion = (
        "<think>verifica: verifica: verifica: fatto, tutto ok davvero</think>"
        "<answer>36</answer>"
    )
    assert verification_step_reward([completion]) == [0.0]


def test_verification_hack_marker_outside_think_pays_nothing() -> None:
    # HACKING CASE: the verification lives outside the reasoning block (or there is
    # no block at all) — only tagged blocks are scanned.
    outside = "<think>12 * 3 = 36</think><answer>36</answer> verifica: 36 / 3 = 12"
    no_block = "verifica: 36 / 3 = 12 <answer>36</answer>"
    assert verification_step_reward([outside, no_block]) == [0.0, 0.0]


def test_verification_hack_restated_result_is_not_a_check() -> None:
    # HACKING CASE: "check: 42" restates one number without any actual comparison.
    completion = "<think>7 * 6 = 42 indeed. check: 42</think><answer>42</answer>"
    assert verification_step_reward([completion]) == [0.0]


# --- concise_answer_reward --------------------------------------------------------

CONCISE_IT = "<think>Sei più sei fa dodici.</think><answer>12</answer>"
CONCISE_EN = "<think>Seven times six is forty-two.</think><answer>42</answer>"


def test_concise_positive_it_and_en() -> None:
    assert concise_answer_reward([CONCISE_IT, CONCISE_EN]) == [1.0, 1.0]


def test_concise_negative_long_answer() -> None:
    completion = f"<think>Some work.</think><answer>{'4' * 81}</answer>"
    assert concise_answer_reward([completion]) == [0.0]


def test_concise_negative_missing_answer_tag() -> None:
    assert concise_answer_reward(["<think>Some work, then nothing.</think>"]) == [0.0]


def test_concise_boundary_at_80_chars() -> None:
    at_limit = f"<think>w</think><answer>{'4' * 80}</answer>"
    assert concise_answer_reward([at_limit]) == [1.0]


def test_concise_hack_empty_or_whitespace_answer_pays_nothing() -> None:
    # HACKING CASE: an empty answer is trivially "concise" and must be worth 0.
    empty = "<think>Some work.</think><answer></answer>"
    whitespace = "<think>Some work.</think><answer>   </answer>"
    assert concise_answer_reward([empty, whitespace]) == [0.0, 0.0]


def test_concise_hack_short_answer_without_think_pays_nothing() -> None:
    # HACKING CASE: brevity without the full envelope would reward the degenerate
    # policy "skip reasoning, emit a short guess".
    assert concise_answer_reward(["<answer>4</answer>"]) == [0.0]


def test_concise_hack_bare_short_text_pays_nothing() -> None:
    # HACKING CASE: a short completion with no tags at all.
    assert concise_answer_reward(["4"]) == [0.0]


def test_concise_factory_custom_budget_and_trl_name() -> None:
    tight = make_concise_answer_reward(max_chars=3)
    long_answer = "<think>w</think><answer>1234</answer>"
    short_answer = "<think>w</think><answer>123</answer>"
    assert tight([long_answer, short_answer]) == [0.0, 1.0]
    # TRL logs per-reward metrics under the function name: it must be distinctive.
    assert tight.__name__ == "concise_answer_reward_3"


# --- shared contracts: determinism and TRL completion formats ---------------------


def test_rewards_are_deterministic() -> None:
    batch = [VERIFIED_IT, CONCISE_EN, "<think>x</think><answer></answer>", "no tags"]
    assert verification_step_reward(batch) == verification_step_reward(batch)
    assert concise_answer_reward(batch) == concise_answer_reward(batch)


def test_rewards_accept_all_trl_completion_shapes() -> None:
    # str, single message dict, and conversational list — all via completion_to_text.
    as_str = VERIFIED_IT
    as_dict = {"role": "assistant", "content": VERIFIED_IT}
    as_conversation = [{"role": "assistant", "content": VERIFIED_IT}]
    assert verification_step_reward([as_str, as_dict, as_conversation]) == [1.0, 1.0, 1.0]
    assert concise_answer_reward([as_str, as_dict, as_conversation]) == [1.0, 1.0, 1.0]


def test_rewards_ignore_extra_trl_kwargs() -> None:
    # TRL passes dataset columns (e.g. the reference answer) to every reward function.
    assert verification_step_reward([VERIFIED_IT], answer=["36"], prompts=["q"]) == [1.0]
    assert concise_answer_reward([CONCISE_IT], answer=["12"], prompts=["q"]) == [1.0]


# --- Language-agnostic contract on Italian completions (bilingual-reasoning, 3) ---
#
# The bilingual-reasoning goal requires the GRPO rewards to stay language-agnostic:
# realistic Italian completions (think in Italian, "verifica:" self-check, numeric
# answer) must score exactly like their English structural twins — same 1.0/0.0.

# The Unicode multiply sign is deliberate (see the ASCII-only note in rewards.py):
# real traces write it, and the window check still passes via the ASCII "=".
BILINGUAL_IT = (
    "<think>Ogni scatola ha 4 matite e le scatole sono 3, quindi 3 * 4 = 12. "
    "verifica: 3 × 4 = 12, il conto torna.</think>\n<answer>12</answer>"  # noqa: RUF001
)
BILINGUAL_EN = (
    "<think>Each box holds 4 pencils and there are 3 boxes, so 3 * 4 = 12. "
    "check: 3 × 4 = 12, the count matches.</think>\n<answer>12</answer>"  # noqa: RUF001
)


def test_bilingual_exact_answer_reward_is_language_agnostic() -> None:
    assert exact_answer_reward([BILINGUAL_IT, BILINGUAL_EN], answer=["12", "12"]) == [1.0, 1.0]
    # A wrong reference zeroes both languages alike: correctness, not wording.
    assert exact_answer_reward([BILINGUAL_IT, BILINGUAL_EN], answer=["13", "13"]) == [0.0, 0.0]


def test_bilingual_format_reward_is_language_agnostic() -> None:
    assert format_reward([BILINGUAL_IT, BILINGUAL_EN]) == [1.0, 1.0]
    # A missing envelope scores 0 in Italian exactly as in English.
    assert format_reward(["Il risultato è 12.", "The result is 12."]) == [0.0, 0.0]


def test_bilingual_verification_reward_is_language_agnostic() -> None:
    # Italian "verifica:" marker, two numbers, Unicode multiply plus ASCII "=" in
    # the window — accepted identically to the English "check:" twin.
    assert verification_step_reward([BILINGUAL_IT, BILINGUAL_EN]) == [1.0, 1.0]
    no_check_it = "<think>3 * 4 = 12, quindi la risposta è 12.</think><answer>12</answer>"
    no_check_en = "<think>3 * 4 = 12, so the answer is 12.</think><answer>12</answer>"
    assert verification_step_reward([no_check_it, no_check_en]) == [0.0, 0.0]


def test_bilingual_concise_reward_is_language_agnostic() -> None:
    assert concise_answer_reward([BILINGUAL_IT, BILINGUAL_EN]) == [1.0, 1.0]
    # Verbose prose answers (105 chars) blow the budget in both languages.
    long_it = f"<think>Un po' di lavoro.</think><answer>{'la risposta è dodici ' * 5}</answer>"
    long_en = f"<think>Some work here.</think><answer>{'the answer is twelve ' * 5}</answer>"
    assert concise_answer_reward([long_it, long_en]) == [0.0, 0.0]


# --- YAML wiring: resolve_reward_config (phase 2) ---------------------------------


def test_resolve_default_is_unchanged_without_rewards_key() -> None:
    # Non-regression: no `rewards` key means the historical pair with the weights
    # taken verbatim (same object) from the existing reward_weights field.
    weights = [1.0, 0.2]
    names, funcs, resolved = resolve_reward_config({"reward_weights": weights})
    assert names == ["exact_answer", "format"]
    assert funcs == [exact_answer_reward, format_reward]
    assert resolved is weights


@pytest.mark.parametrize("name", ["grpo", "grpo-smoke", "qwen3-1.7b-grpo"])
def test_resolve_existing_grpo_configs_keep_default_rewards(name: str) -> None:
    # Non-regression over the real config files (goal contract): the pre-existing
    # GRPO YAMLs have no `rewards` key and must resolve to the historical pair.
    config = load_yaml(Path(__file__).parents[1] / f"configs/posttrain/{name}.yaml")
    names, funcs, weights = resolve_reward_config(config["training"])
    assert names == ["exact_answer", "format"]
    assert funcs == [exact_answer_reward, format_reward]
    assert weights == config["training"]["reward_weights"]


def test_resolve_valid_rewards_list() -> None:
    training = {
        "rewards": [
            {"name": "exact_answer", "weight": 1.0},
            {"name": "format", "weight": 0.1},
            {"name": "verification_step", "weight": 0.15},
            {"name": "concise_answer", "weight": 0.1},
        ]
    }
    names, funcs, weights = resolve_reward_config(training)
    assert names == ["exact_answer", "format", "verification_step", "concise_answer"]
    assert funcs == [
        exact_answer_reward,
        format_reward,
        verification_step_reward,
        concise_answer_reward,
    ]
    assert weights == [1.0, 0.1, 0.15, 0.1]


def test_resolve_rejects_unknown_reward_name() -> None:
    training = {
        "rewards": [{"name": "exact_answer", "weight": 1.0}, {"name": "bogus", "weight": 0.1}]
    }
    with pytest.raises(ValueError, match=r"Unknown reward 'bogus'.*valid names"):
        resolve_reward_config(training)


def test_resolve_rejects_custom_weight_above_contract_cap() -> None:
    # 0.25 > 0.2: the goal contract caps custom shape rewards; no bypass flag exists,
    # only a recorded decision can raise it.
    training = {
        "rewards": [
            {"name": "exact_answer", "weight": 1.0},
            {"name": "verification_step", "weight": 0.25},
        ]
    }
    with pytest.raises(ValueError, match=r"goal-contract cap.*recorded decision"):
        resolve_reward_config(training)


def test_resolve_rejects_missing_exact_answer() -> None:
    training = {
        "rewards": [{"name": "format", "weight": 1.0}, {"name": "concise_answer", "weight": 0.1}]
    }
    with pytest.raises(ValueError, match="must include 'exact_answer'"):
        resolve_reward_config(training)


def test_resolve_rejects_non_dominant_exact_answer() -> None:
    # A tie is not dominance: correctness must be strictly heavier than any shaping.
    heavier = {
        "rewards": [{"name": "exact_answer", "weight": 0.1}, {"name": "format", "weight": 0.2}]
    }
    tied = {
        "rewards": [{"name": "exact_answer", "weight": 1.0}, {"name": "format", "weight": 1.0}]
    }
    for training in (heavier, tied):
        with pytest.raises(ValueError, match="requires exact_answer to dominate"):
            resolve_reward_config(training)


def test_resolve_rejects_rewards_alongside_reward_weights() -> None:
    # Two sources of truth for weights would let one silently override the other.
    training = {
        "rewards": [{"name": "exact_answer", "weight": 1.0}],
        "reward_weights": [1.0, 0.2],
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_reward_config(training)


def test_resolve_rejects_duplicates_and_malformed_entries() -> None:
    duplicated = {
        "rewards": [{"name": "exact_answer", "weight": 1.0}, {"name": "exact_answer", "weight": 0.5}]
    }
    with pytest.raises(ValueError, match="Duplicate reward"):
        resolve_reward_config(duplicated)
    with pytest.raises(ValueError, match="non-empty list"):
        resolve_reward_config({"rewards": []})
    with pytest.raises(ValueError, match="'name' and 'weight'"):
        resolve_reward_config({"rewards": [{"name": "exact_answer"}]})
    with pytest.raises(ValueError, match="must be positive"):
        resolve_reward_config(
            {"rewards": [{"name": "exact_answer", "weight": 1.0}, {"name": "format", "weight": 0}]}
        )
