import os

import pytest
import torch


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)


def test_arange_elems() -> None:
    arr = torch.arange(0, 10, dtype=torch.float32)
    assert torch.allclose(arr[-1], torch.tensor([9.0]))


def test_div_zero() -> None:
    a = torch.zeros(1, dtype=torch.long)
    b = torch.ones(1, dtype=torch.long)

    assert torch.isinf(b / a)


def test_div_zero_python() -> None:
    with pytest.raises(ZeroDivisionError):
        1 / 0
