"""Random-number generator setup for reproducible experiment runs."""

from __future__ import annotations

import random

from helpers.logging_config import get_logger

logger = get_logger(__name__)


def configure_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch before pipeline/model construction.

    This controls stochastic model initialization and training operations such
    as dropout. It intentionally does not force deterministic accelerator
    kernels, which can reject or significantly slow scatter-based GNN code.
    """
    if seed < 0:
        raise ValueError("Random seed must be greater than or equal to zero.")

    random.seed(seed)

    import numpy

    numpy.random.seed(seed % (2**32))

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(
        "Configured random seed=%s for Python, NumPy, and PyTorch "
        "(deterministic accelerator kernels not forced)",
        seed,
    )
