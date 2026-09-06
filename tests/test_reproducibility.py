"""Tests for pipeline random seed control."""

from __future__ import annotations

import random

import numpy
import torch

from helpers.reproducibility import configure_random_seed


def test_configure_random_seed_repeats_python_numpy_and_torch_values() -> None:
    configure_random_seed(123)
    first = (
        random.random(),
        float(numpy.random.random()),
        torch.rand(3),
    )

    configure_random_seed(123)
    second = (
        random.random(),
        float(numpy.random.random()),
        torch.rand(3),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
