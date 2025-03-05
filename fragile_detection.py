"""Module concerned with detecting the fragile watermark"""

import numpy as np
import hashlib

# import ECG_robust as robust
import ECG_fragile as fragile
import parameters as param

def _store_each_lsb_of_series(input_signal: np.ndarray) -> np.ndarray:
    """Given an input function, this function takes the Least Significant
    Bit (LSB) of each element and stores it
    - input_signal (list): input signal to be processed
    - output (list): list of LSBs of each element in input signal"""

    return input_signal.astype(int) & 1 # Get LSBs

def generate_seed(binary_str: str) -> int:
    """Hashes a binary string to produce a valid seed for np.random.seed()."""
    hash_digest = hashlib.sha256(binary_str.encode()).hexdigest()
    return int(hash_digest, 16) % (param.PYTHON_SEED_LIMIT)  # Fit within seed range

def detect_tampering(received_ecg: np.ndarray, seed_k: int, segment_indices_list: list):
    """Detects if the watermarked signal has been tampered with
    - original_ecg (np.ndarray): original ECG signal
    - received_ecg (np.ndarray): received ECG signal
    - fs (float): sampling frequency of ECG signal
    - seed_k (int): kappa value for the watermark
    - segment_indices_list (list): list of indices of the segments in the original ECG signal that are watermarked
    - output (dict): Tampered segments and their degree of alteration"""

    tampered_segments = {}
    extracted_lsb = _store_each_lsb_of_series(fragile.watermarked_signal)

    for i, segment_indices in enumerate(segment_indices_list):
        received_segment = received_ecg[segment_indices]
        hash_values_of_signal   = fragile.compute_segment_hashes(fragile.watermarked_signal, fragile.window_indices_for_all_segments)
        quantized_segment_hashes= fragile.quantize_hash_values_for_all_segments(hash_values_of_signal, param.BIT_LENGTH)
        seeded_hash_segments    = fragile.prepend_seed_to_every_hash(quantized_segment_hashes, seed_k, param.BIT_LENGTH)

        recomputed_segment_watermarks = fragile.convert_hash_to_int_and_generate_watermark(segment_indices, seeded_hash_segments)

        extracted_watermark = extracted_lsb[segment_indices]
        if not np.array_equal(recomputed_segment_watermarks, extracted_watermark):
            tampering_amount = np.sum(recomputed_segment_watermarks != extracted_watermark) / len(received_segment)
            tampered_segments[i] = tampering_amount

    return tampered_segments

# NOTE: segment_indices_list in the function is fragile.segments_list
# NOTE: make sure we are not skipping the seed scaling step

signal_lsb = _store_each_lsb_of_series(fragile.watermarked_signal)
hash_values_of_signal   = fragile.compute_segment_hashes(fragile.watermarked_signal, fragile.window_indices_for_all_segments)
quantized_segment_hashes= fragile.quantize_hash_values_for_all_segments(hash_values_of_signal, param.BIT_LENGTH)
seeded_hash_segments    = fragile.prepend_seed_to_every_hash(quantized_segment_hashes, param.SEED_K, param.BIT_LENGTH)
tampered_segments = detect_tampering(fragile.watermarked_signal, param.SEED_K, segment_indices_list)
