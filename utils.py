""""Module containing shared functions to be used across modules"""

import numpy as np

def get_mae(sequence1, sequence2) -> float:
    """Calculates the Mean Absolute Error (not %) between 2 sequences"""

    if len(sequence1) != len(sequence2):
        print("Sequences are of different lengths")       
    return np.mean(np.abs(sequence1 - sequence2))

def normalize_dataset(input_data):
    """Normalize the input data to make its mean 0 and stdev 1"""
    return (input_data - np.mean(input_data)) / np.std(input_data)
