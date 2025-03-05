"""Module about detecting the fragile watermark"""

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

    original_window_indices = fragile.window_indices_for_all_segments
    extracted_lsb     = _store_each_lsb_of_series(received_ecg)
    tampered_segments = {}

    for i, segment_indices in enumerate(segment_indices_list):
        received_segment = received_ecg[segment_indices]

        # Step 2: Recompute hash values using same window indices
        print('ooooooooooooooooo')
        recomputed_hash_matrix = fragile.compute_segment_hashes(received_segment, original_window_indices[i])

        # Step 3: Quantize hash values (same bit length as embedding)
        quantized_hash = fragile._quantize_hash_values_for_1_segment(recomputed_hash_matrix, param.BIT_LENGTH)

        # Step 4: Prepend kappa (seed)
        kappa_binary = format(seed_k, f'0{param.BIT_LENGTH}b')
        seeded_binary = kappa_binary + quantized_hash

        # Step 5: Generate fragile watermark
        seed_int = int(seeded_binary, 2)  
        scaled_seed = fragile._scale_all_seeds(np.array([seed_int]))[0]  
        np.random.seed(int(scaled_seed))
        recomputed_watermark = np.random.randint(0, 2, size=len(received_segment))

        # Step 6: Compare extracted vs. recomputed watermark
        extracted_watermark = extracted_lsb[segment_indices]
        if not np.array_equal(recomputed_watermark, extracted_watermark):
            tampering_amount = np.sum(recomputed_watermark != extracted_watermark) / len(received_segment)
            tampered_segments[i] = tampering_amount

    return tampered_segments

tampered_segments = detect_tampering(fragile.unshifted_watermarked_ecg_signal,
                                     param.SEED_K, fragile.segments_list)

# NOTE: segment_indices_list in the function is fragile.segments_list
# NOTE: make sure we are not skipping the seed scaling step
