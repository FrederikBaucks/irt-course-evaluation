"""Miscellaneous helper functions."""
import random
import numpy as np


def set_random_seeds(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def sigmoid(x):
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-x))
