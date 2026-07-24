"""TrainingArguments plumbing regression tests; they need the training stack."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from my_llm.config import load_yaml
from my_llm.posttrain import common_training_args

pytestmark = pytest.mark.training

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compile_mode_is_gated_by_the_compile_flag() -> None:
    # TrainingArguments force-enables compile whenever mode/backend is non-null,
    # so forwarding the mode with torch_compile: false silently inverts the flag.
    config = load_yaml(REPO_ROOT / "configs" / "posttrain" / "sft-smoke.yaml")
    training = dict(config["training"])

    training["torch_compile"] = False
    training["torch_compile_mode"] = "reduce-overhead"
    args = common_training_args(config, training, has_eval=False, train_examples=8)
    assert args["torch_compile"] is False
    assert args["torch_compile_mode"] is None

    training["torch_compile"] = True
    args = common_training_args(config, training, has_eval=False, train_examples=8)
    assert args["torch_compile_mode"] == "reduce-overhead"
