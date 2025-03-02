"""This module implements the fragile watermark from the ECG paper"""

from typing import List, Union

import numpy as np
import scipy.signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

import parameters as param

def shift_signal_up_to_remove_negative_values(ecg_signal):
    """Shift signal up to remove negative values, as algo can't handle them"""
    min_value = np.min(ecg_signal)
    if min_value <0:
        shifted_ecg_signal = ecg_signal - min_value
    return shifted_ecg_signal, min_value

def scale_signal_and_remove_decimals(ecg_signal, scale_factor):
    """The ECG contains decimals, which are not well handled by the LSB algo, so we scale the signal to 
    minimize the damage, then remove decilams, and make it of type int. Ex: ECG signal y-value of 3.51 needs to be int first, so becomes 3, then applying
    LSB gives 3-1=2. We lost too much info, so let's scale 3.5 by 100 (for ex)= 351, then do LSB -> 350, we
    lost less info"""
    return np.floor(ecg_signal * scale_factor).astype(int)

def remove_lsb_from_each_element_in_signal(scaled_signal):
    """The least significant bit (LSB) is the last bit in a number's binary form, ex: 19 (dec) = 10011 (bin)
    The LSB here is 1. This function intakes an array, clears the LSB to 0, equivalent to subtracting 1 when
    the number is odd, or leaving it unchanged it even"""
    scaled_signal_no_lsb = scaled_signal & ~1
    return scaled_signal_no_lsb

def _get_signal_peaks(ecg_signal_no_lsb, distance_between_points, peak_as_fraction_of_max):
    """Given a signal, get the peak indices and heights"""
    min_peak_height               = np.max(ecg_signal_no_lsb) * peak_as_fraction_of_max
    peak_indices, peak_properties = scipy.signal.find_peaks(ecg_signal_no_lsb, height = min_peak_height, distance= distance_between_points)
    peak_heights                  = peak_properties['peak_heights']  # Extract peak heights
    return peak_indices, peak_heights

def split_signal_to_heartbeat_segments(ecg_signal_no_lsb) -> list:
    """Divide ECG signal to roughly equal segments so each contains a heartbeat (no strict requirement)
    np.array_split makes the split as equal as possible. No overlap between segments"""

    peak_indices, peak_heights= _get_signal_peaks(ecg_signal_no_lsb, param.MIN_DIST_BETWEEN_PEAKS, param.PEAK_AS_FRACTION_OF_MAX)
    avg_heartbeat_length      = np.floor(np.mean(np.diff(peak_indices)))
    num_segments_in_signal    = int(np.floor(len(ecg_signal_no_lsb)/avg_heartbeat_length))
    ecg_signal_indices        = np.arange(len(ecg_signal_no_lsb))
    segments_indices_list     = np.array_split(ecg_signal_indices, num_segments_in_signal)

    return segments_indices_list, num_segments_in_signal

def _get_window_indices_for_1_segment(segment_array, num_of_intervals, window_length) -> list:
    """Given a segment, generate random intervals of size 'window_length'
    within segment. Can overlap, but not exceed segment boundaries
    - num_of_intervals: # intervals we will have in segment (= p from paper)
    - window_length, (= w from paper)
    - intervals_array: array of all interval indices in a segment"""
    segment_length = len(segment_array)
    if segment_length < window_length:
        print("Segment is smaller than window, window size will be adjusted!")
        window_length = segment_length//2

    starting_points = np.random.randint(0, segment_length - window_length, size = num_of_intervals)
    intervals_array = [segment_array[start: start + window_length] for start in starting_points]
    return intervals_array

def get_window_indices_for_all_segments(segments_indices_list: list, seed_k: int) -> list:
    """Generate interval windows for all segments by looping function that does
    it for each segment
    - segments_list: list of ECG segments
    - seed_k: random seed (= k from paper)
    - output (list): list of interval arrays for each segment"""
    np.random.seed(seed_k)

    window_indices_for_all_segments = []
    for segment in segments_indices_list:
        window_indices_for_1_segment = _get_window_indices_for_1_segment(segment, param.NUM_WINDOWS_PER_SEGMENT, param.WINDOW_LEN)
        window_indices_for_all_segments.append(window_indices_for_1_segment)
    return window_indices_for_all_segments

def _apply_butterworth_filter_to_1_window(signal: np.ndarray, fs: float, cutoff: Union[float, List[float]], order: int, filter_type: str) -> np.ndarray:
    """Applies Butterworth filter to input signal
    - signal (np.ndarray): The ECG signal to filter.
    - fs (float): Sampling frequency in Hz
    - cutoff (float | List[float]): Cutoff frequency/frequencies in Hz.
    - order (int): order of the filter, 4 is recommended
    - filter_type (str): Type of filter - 'low', 'band', or 'high'.
    - output (np.ndarray: filtered signal), same length as signal"""
    nyquist_freq      = 0.5 * fs  # Nyquist frequency
    normalized_cutoff = np.array(cutoff) / nyquist_freq  # Normalize cutoff frequency
    b_num, a_denom    = butter(order, normalized_cutoff, btype=filter_type, analog=False)
    return filtfilt(b_num, a_denom, signal)

def _compute_signal_power_of_1_window(signal: np.ndarray) -> float:
    """Computes a signal's power
    - signal (np.ndarray): input signal
    - output (float): input signal's computed power"""
    return np.mean(signal ** 2)

def _compute_hash_values_of_1_window(ecg_window: np.ndarray, fs: float) -> List[float]:
    """Computes 3 hash values for a given ECG window; for each of low+band+high freq filters
    - ecg_window (np.ndarray): The interval of ECG signal
    - fs (float): Sampling frequency in Hz
    - output (List[float]): 3 power values for low-, band-, andhigh-pass filtered signals"""

    # Cutoff frequencies (denominator from internet)
    low_cutoff  = fs / 6
    high_cutoff = fs / 3
    band_cutoff = [low_cutoff, high_cutoff]
    ORDER = 4 # recommended value

    # Apply filters
    low_pass_signal  = _apply_butterworth_filter_to_1_window(ecg_window, fs, cutoff=low_cutoff, order=ORDER, filter_type='low')
    band_pass_signal = _apply_butterworth_filter_to_1_window(ecg_window, fs, cutoff=band_cutoff, order=ORDER, filter_type='band')
    high_pass_signal = _apply_butterworth_filter_to_1_window(ecg_window, fs, cutoff=high_cutoff, order=ORDER, filter_type='high')

    # Compute power values
    power_low  = _compute_signal_power_of_1_window(low_pass_signal)
    power_band = _compute_signal_power_of_1_window(band_pass_signal)
    power_high = _compute_signal_power_of_1_window(high_pass_signal)

    hash_values_of_window = [power_low, power_band, power_high] # 3 hash values
    return hash_values_of_window

def compute_segment_hashes(ecg_signal_no_lsb, window_indices_for_all_segments: list) -> np.ndarray:
    """Apply hash computation to each window in an ECG signal segmented into 
    multiple segments and windows, then store results in 3D array. It then reshapes
    the hash array to concatenate hash values for each segment.
    - window_indices_for_all_segments (list): nested list where each segment 
      contains multiple windows, and each window is represented by its indices
    - np.ndarray: 3D array of shape (num_segments, num_windows_per_segment, 3) 
      containing the hash values for each window"""

    ecg_hash_matrix = np.full((num_segments_in_signal,
                               param.NUM_WINDOWS_PER_SEGMENT,
                               param.NUM_HASH_VALUES_PER_WINDOW), np.nan)

    for i, segment in enumerate(window_indices_for_all_segments):
        for j, window in enumerate(segment):
            hash_values = _compute_hash_values_of_1_window(ecg_signal_no_lsb[window], param.fs)
            ecg_hash_matrix[i, j, :] = hash_values

    segment_hashes = ecg_hash_matrix.reshape(num_segments_in_signal, -1)
    return segment_hashes

def _quantize_hash_values_for_1_segment(segment_hashes: np.ndarray, bit_length: int) -> str:
    """Converts decimal hash values to fixed-length binary sequence
    - hash_values (np.ndarray): hash values for 1 segment
    - bit_length (int): # of bits for each quantized value (default 8-bit)
    - str: concatenated binary string"""

    scaled_values = (segment_hashes * (2**bit_length - 1)).astype(int)  # Scale to [0, 255] for 8-bit
    binary_strings = [format(val, f'0{bit_length}b') for val in scaled_values]  # Convert to binary
    return ''.join(binary_strings) # Concatenate into a single binary string

def quantize_hash_values_for_all_segments(segment_hashes: np.ndarray, bit_length: int) -> list:
    """Converts decimal hash values to fixed-length binary sequence for all segments
    - segment_hashes (np.ndarray): hash values for all segments
    - bit_length (int): # of bits for each quantized value (default 8-bit)
    - str: concatenated binary string"""

    quantized_segment_hashes = []
    for segment in segment_hashes:
        quantized_segment_hash = _quantize_hash_values_for_1_segment(segment, bit_length)
        quantized_segment_hashes.append(quantized_segment_hash)

    return quantized_segment_hashes

def prepend_seed_to_every_hash(quantized_segment_hashes: list, seed: int, bit_length: int) -> list:
    """Prepends the seed (kappa) to the beginning of each hash string
    - seed (int): kappa value
    - bit_length (int): # of bits for seed, equal to the bit_length of hash values
    - quantized_segment_hash (str): concatenated binary string of hash values
    - str: concatenated binary string of kappa and hash values"""
    seed_binary          = format(seed, f'0{bit_length}b')
    seeded_hash_segments = [seed_binary + binary_hash for binary_hash in quantized_segment_hashes]
    return seeded_hash_segments

def _scale_all_seeds(all_seeds_int: np.ndarray) -> np.ndarray:
    """Scales all seeds to be within 32-bit integer range    
    - all_seeds_int (np.ndarray): array of seeds to be scaled
    - np.ndarray: array of scaled seeds"""	

    PYTHON_SEED_LIMIT = 2**32 - 1
    max_seed          = np.max(all_seeds_int)

    if max_seed > PYTHON_SEED_LIMIT:
        scale_factor  = max_seed / PYTHON_SEED_LIMIT  # Compute relative scaling factor
        all_seeds_int = all_seeds_int / scale_factor  # Scale seed down proportionally
        print(f"Max seed exceeds Python limits, will downscale all seeds")
    all_seeds_int = np.floor(all_seeds_int)#.astype(int)
    return all_seeds_int

def convert_hash_to_int_and_generate_watermark(ecg_segments: list, seeded_hash_segments: list) -> list:
    """Converts the seeded hash segments to integers
    - ecg_segments (list): list of ECG segments
    - seeded_hash_segments (list): list of seeded hash segments
    - np.ndarray: array of integers"""

    all_seeds_int = np.array([int(seeded_binary, 2) for seeded_binary in seeded_hash_segments])
    scaled_seeds  = _scale_all_seeds(all_seeds_int) # Scale based on largest seed

    watermarks_for_all_segments = []
    for i, seed in enumerate(scaled_seeds):
        np.random.seed(int(seed))
        watermark_for_segment = np.random.randint(0, 2, size=len(ecg_segments[i]))
        watermarks_for_all_segments.append(watermark_for_segment)

    return watermarks_for_all_segments

def embed_watermark_into_ecg(ecg_signal: list, ecg_segments: list, watermarks_for_all_segments: list) -> list:
    """Embeds watermarks into ECG segments
    - ecg_segments (list): list of ECG segments
    - watermarks_for_all_segments (list): list of watermarks for each segment
    - np.ndarray: array of watermarked ECG segments"""

    watermarked_ecg = ecg_signal.copy()  # Make a copy of the original ECG signal

    for i, segment_idx in enumerate(ecg_segments):
        watermark = watermarks_for_all_segments[i] # Get binary watermark for segment
        for j, idx in enumerate(segment_idx):
            watermarked_ecg[idx] = (watermarked_ecg[idx] & ~1) | watermark[j]  # Set LSB with the watermark bit
    return watermarked_ecg

def unscale_signal(scaled_signal_no_lsb, scale_factor):
    """Function that unscaled the modified signal"""
    ecg_signal_no_lsb = scaled_signal_no_lsb / scale_factor
    return ecg_signal_no_lsb

def unshift_signal_back_to_original(ecg_signal, min_value):
    """Shift signal down to bring signal back to original
    - ecg_signal (np.ndarray): array of watermarked ECG segments
    - min_value (int): minimum value of the signal"""

    if min_value <0:
        unshifted_ecg_signal = ecg_signal + min_value
    return unshifted_ecg_signal


shifted_ecg_signal, min_value  = shift_signal_up_to_remove_negative_values(param.ecg_signal)
scaled_signal                  = scale_signal_and_remove_decimals(shifted_ecg_signal, param.ECG_SCALE_FACTOR)
scaled_signal_no_lsb           = remove_lsb_from_each_element_in_signal(scaled_signal)
segments_list, num_segments_in_signal= split_signal_to_heartbeat_segments(scaled_signal_no_lsb)
window_indices_for_all_segments= get_window_indices_for_all_segments(segments_list, 2)
segment_hashes                 = compute_segment_hashes(scaled_signal_no_lsb, window_indices_for_all_segments)
quantized_segment_hashes       = quantize_hash_values_for_all_segments(segment_hashes, param.BIT_LENGTH)
seeded_hash_segments           = prepend_seed_to_every_hash(quantized_segment_hashes, param.SEED_K, param.BIT_LENGTH)
watermarks_for_all_segments    = convert_hash_to_int_and_generate_watermark(segments_list, seeded_hash_segments)
watermarked_signal             = embed_watermark_into_ecg(scaled_signal_no_lsb, segments_list, watermarks_for_all_segments)
watermarked_ecg_signal_unscaled= unscale_signal(watermarked_signal, param.ECG_SCALE_FACTOR)
unshifted_ecg_signal           = unshift_signal_back_to_original(watermarked_ecg_signal_unscaled, min_value)

print(np.mean(np.abs(param.ecg_signal - unshifted_ecg_signal)/param.ecg_signal) * 100)


plt.figure(figsize=(13,6))
plt.plot(param.ecg_signal, label="Original ECG")
plt.plot(unshifted_ecg_signal, label="ECG+fragile WM")
plt.title("ECG Signal")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.legend()
plt.show()

# TODO: make input of fragile watermark be the robust watermarked signal
