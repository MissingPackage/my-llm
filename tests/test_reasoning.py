"""Tests for auditable answer parsing and exact GRPO rewards."""

from my_llm.reasoning import (
    answers_equal,
    exact_answer_reward,
    extract_final_answer,
    format_reward,
    gsm8k_messages,
    split_gsm8k_solution,
)


def test_extracts_tagged_and_gsm8k_answers() -> None:
    assert extract_final_answer("work <answer>1,024</answer>") == "1,024"
    assert extract_final_answer("some work\n#### 42") == "42"
    assert extract_final_answer("the result is -3.5") == "-3.5"
    assert extract_final_answer(r"therefore \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_numeric_answers_are_canonicalized() -> None:
    assert answers_equal("1,024.0", "1024")
    assert answers_equal("1/2", "0.5")
    assert answers_equal(r"\frac{1}{2}", "0.5")
    assert not answers_equal("2", "3")


def test_rewards_handle_text_and_conversational_completions() -> None:
    completions = [
        "<reasoning>2 + 2 is 4.</reasoning><answer>4</answer>",
        [{"role": "assistant", "content": "<reasoning>x</reasoning><answer>5</answer>"}],
    ]
    assert exact_answer_reward(completions, ["#### 4", "#### 4"]) == [1.0, 0.0]
    assert format_reward(completions) == [1.0, 1.0]


def test_native_think_format_is_rewarded() -> None:
    completion = "<think>2 + 2 = 4</think><answer>4</answer>"
    assert format_reward([completion]) == [1.0]


def test_gsm8k_sft_format() -> None:
    reasoning, answer = split_gsm8k_solution("First add.\n#### 12")
    assert (reasoning, answer) == ("First add.", "12")
    messages = gsm8k_messages("What is six plus six?", "First add.\n#### 12")
    assert messages[-1]["content"].endswith("<answer>12</answer>")
