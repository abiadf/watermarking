""""Module containing shared functions to be used across modules"""

import numpy as np

def get_mae(sequence1, sequence2) -> float:
    """Calculates the Mean Absolute Error (not %) between 2 sequences"""
    return np.mean(np.abs(sequence1 - sequence2))
