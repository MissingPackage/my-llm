"""Learning-rate schedule regression tests."""

import pytest

from my_llm.pretrain import lr_multiplier


def test_warmup_and_cosine_schedule() -> None:
    assert lr_multiplier(0, warmup_steps=10, max_steps=100, min_ratio=0.1) == pytest.approx(0.1)
    assert lr_multiplier(9, warmup_steps=10, max_steps=100, min_ratio=0.1) == pytest.approx(1.0)
    assert lr_multiplier(100, warmup_steps=10, max_steps=100, min_ratio=0.1) == pytest.approx(0.1)
