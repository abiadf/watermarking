"""Module about detecting the fragile watermark
For this module's input signal, we use the shifter/scaled signal (NOT the original)"""

import numpy as np
import hashlib
from typing import Dict
import parameters as param
import ECG_fragile as fragile

def store_each_lsb_of_series(input_signal: np.ndarray) -> np.ndarray:
    """Takes the Least Significant Bit (LSB) of each element of an input
    signal and stores it
    - input_signal (list): input signal to be processed
    - output (list): list of LSBs of each element in input signal"""

    return input_signal.astype(int) & 1 # Get LSBs

def segment_extracted_watermark(extracted_lsb: np.ndarray, segments_list: list) -> list:
    """Extracts LSB of each segment in the watermarked signal
    - extracted_lsb (np.ndarray): LSB of watermarked signal
    - segments_list (list): list of indices of the segments in the original ECG signal that are watermarked
    - output (list): list of LSBs of each segment of the watermarked signal"""
    return [extracted_lsb[segment] for segment in segments_list]

def generate_sha256_seed(binary_str: str) -> int:
    """Generates a valid random seed from a binary string using SHA-256.
    binary_str (str): Binary string to hash
    int: Seed value within Python's seed limit"""

    hash_digest = hashlib.sha256(binary_str.encode()).hexdigest()
    return int(hash_digest, 16) % (param.PYTHON_SEED_LIMIT)  # Fit within seed range

def _bit_accuracy_rate(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """Returns bit accuracy rate between two signals
    - signal1/2 (np.ndarray): signal
    - output (float): bit accuracy rate (0 to 1)"""
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have equal length")
    return np.mean(signal1 == signal2)

def get_bit_accuracy(recomputed_watermarks_for_all_segments: list, extracted_watermark_segments: list) -> Dict[int, float]:
    """Detects if the watermarked signal has been tampered with
    - recomputed_watermarks_for_all_segments (list): list of recomputed watermarks for each segment
    - extracted_watermark_segments (list): list of extracted watermarks for each segment
    - output (dict): dict of segment index to tampering ratio for tampered segments"""
    
    all_bit_accuracy  = {}
    for i, (recomputed_wm, extracted_wm) in enumerate(zip(recomputed_watermarks_for_all_segments, extracted_watermark_segments)):
        if not np.array_equal(recomputed_wm, extracted_wm):
            bit_accuracy = _bit_accuracy_rate(recomputed_watermarks_for_all_segments[i],
                                              extracted_watermark_segments[i])
            all_bit_accuracy[i] = bit_accuracy

    return all_bit_accuracy

# recomputes watermark
segment_hashes = fragile.compute_segment_hashes(fragile.watermarked_signal,# fragile.unshifted_watermarked_ecg_signal,
                                                fragile.window_indices_for_all_segments,
                                                fragile.num_segments_in_signal)
quantized_segment_hashes = fragile.quantize_hash_values_for_all_segments(segment_hashes,
                                                                         param.BIT_LENGTH)
seeded_hash_segments = fragile.prepend_seed_to_every_hash(quantized_segment_hashes, param.SEED_K, param.BIT_LENGTH)
recomputed_watermarks_for_all_segments = fragile.convert_hash_to_int_and_generate_watermark(fragile.segments_list,
                                                                                            seeded_hash_segments)

# deals with existing watermarks
extracted_lsb = store_each_lsb_of_series(fragile.watermarked_signal)
extracted_watermark_segments = segment_extracted_watermark(extracted_lsb, fragile.segments_list)

all_bit_accuracy = get_bit_accuracy(recomputed_watermarks_for_all_segments, extracted_watermark_segments)
# print(all_bit_accuracy)

print("Extracted Watermark First Segment:", extracted_watermark_segments[0])
print("Original Watermark First Segment:", recomputed_watermarks_for_all_segments[0])
