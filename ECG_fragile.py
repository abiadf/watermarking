"""This module implements the fragile watermark from the ECG paper.
The input signal of the fragile watermark is the signal + robust watermark (from paper)"""

from typing import List, Union

import numpy as np
import scipy.signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

import ECG_robust as robust
import ECG_parameters as param
from ECG_robust import SignalAnalysis

class SignalProcessing():
    """Handles pre- and post-processing of signal"""

    @staticmethod
    def shift_signal_up_to_remove_negative_values(ecg_signal: np.ndarray) -> np.ndarray:
        """Shift signal up to remove negative values, as algo can't handle them"""
        min_value = np.min(ecg_signal)
        if min_value <0:
            shifted_ecg_signal = ecg_signal - min_value
        return shifted_ecg_signal, min_value

    @staticmethod
    def scale_signal_and_remove_decimals(ecg_signal: np.ndarray, scale_factor: int) -> np.ndarray:
        """The ECG contains decimals, which are not well handled by the LSB algo, so we scale the signal to 
        minimize the damage, then remove decilams, and make it of type int. Ex: ECG signal y-value of 3.51 needs to be int first, so becomes 3, then applying
        LSB gives 3-1=2. We lost too much info, so let's scale 3.5 by 100 (for ex)= 351, then do LSB -> 350, we
        lost less info"""
        return np.floor(ecg_signal * scale_factor).astype(int)

    @staticmethod
    def remove_lsb_from_each_element_in_signal(scaled_signal: np.ndarray) -> np.ndarray:
        """The least significant bit (LSB) is the last bit in a number's binary form, ex: 19 (dec) = 10011 (bin)
        The LSB here is 1. This function intakes an array, clears the LSB to 0, equivalent to subtracting 1 when
        the number is odd, or leaving it unchanged it even"""
        scaled_signal_no_lsb = scaled_signal & ~1
        return scaled_signal_no_lsb

    @staticmethod
    def unscale_signal(scaled_signal_no_lsb: np.ndarray, scale_factor: int) -> np.ndarray:
        """Function that unscaled the modified signal"""
        ecg_signal_no_lsb = scaled_signal_no_lsb / scale_factor
        return ecg_signal_no_lsb

    @staticmethod
    def unshift_signal_back_to_original(ecg_signal: np.ndarray, min_value) -> np.ndarray:
        """Shift signal down to bring signal back to original
        - ecg_signal (np.ndarray): array of watermarked ECG segments
        - min_value (int): minimum value of the signal"""

        if min_value <0:
            unshifted_ecg_signal = ecg_signal + min_value
        return unshifted_ecg_signal

class WatermarkGenerator():
    """Handles watermark generation process"""

    @staticmethod
    def _get_signal_peaks(ecg_signal_no_lsb: np.ndarray, distance_between_points: int, peak_as_fraction_of_max: float) -> tuple:
        """Given a signal, get the peak indices and heights"""
        min_peak_height               = np.max(ecg_signal_no_lsb) * peak_as_fraction_of_max
        peak_indices, peak_properties = scipy.signal.find_peaks(ecg_signal_no_lsb, height = min_peak_height, distance= distance_between_points)
        peak_heights: np.ndarray      = peak_properties['peak_heights']  # Extract peak heights
        return peak_indices, peak_heights

    @staticmethod
    def split_signal_to_heartbeat_segments(ecg_signal_no_lsb: np.ndarray) -> tuple:
        """Divide ECG signal to roughly equal segments so each contains a heartbeat (no strict requirement)
        np.array_split makes the split as equal as possible. No overlap between segments"""

        peak_indices, peak_heights= WatermarkGenerator._get_signal_peaks(ecg_signal_no_lsb, param.MIN_DIST_BETWEEN_PEAKS, param.PEAK_AS_FRACTION_OF_MAX)
        avg_heartbeat_length      = np.floor(np.mean(np.diff(peak_indices)))
        num_segments_in_signal    = int(np.floor(len(ecg_signal_no_lsb)/avg_heartbeat_length))
        ecg_signal_indices        = np.arange(len(ecg_signal_no_lsb))
        segments_indices_list     = np.array_split(ecg_signal_indices, num_segments_in_signal)

        return segments_indices_list, num_segments_in_signal

    @staticmethod
    def _get_window_indices_for_1_segment(segment_array: np.ndarray, num_of_intervals: int, window_length: int) -> list:
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

    @staticmethod
    def get_window_indices_for_all_segments(segments_indices_list: list, seed_k: int) -> list:
        """Generate interval windows for all segments by looping function that does
        it for each segment
        - segments_list: list of ECG segments
        - seed_k: random seed (= k from paper)
        - output (list): list of interval arrays for each segment"""
        np.random.seed(seed_k)

        window_indices_for_all_segments = []
        for segment in segments_indices_list:
            window_indices_for_1_segment = WatermarkGenerator._get_window_indices_for_1_segment(segment, param.NUM_WINDOWS_PER_SEGMENT, param.WINDOW_LEN)
            window_indices_for_all_segments.append(window_indices_for_1_segment)
        return window_indices_for_all_segments

    @staticmethod
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
        # print(len(b_num), len(a_denom), type(signal), signal)
        return filtfilt(b_num, a_denom, signal)

    @staticmethod
    def _compute_signal_power_of_1_array(signal: np.ndarray) -> float:
        """Computes an array's power
        - signal (np.ndarray): input signal
        - output (float): input signal's computed power"""
        return np.mean(signal ** 2)

    @staticmethod
    def _compute_power_hash_values_of_1_window(ecg_window: np.ndarray, fs: float) -> List[float]:
        """Computes 3 hash values for a given ECG window; for each of low+band+high freq filters
        - ecg_window (np.ndarray): The interval of ECG signal
        - fs (float): Sampling frequency in Hz
        - output (List[float]): 3 power values for low-, band-, andhigh-pass filtered signals"""

        # Cutoff frequencies (denominator from internet)
        low_cutoff  = fs / 6
        high_cutoff = fs / 3
        band_cutoff = [low_cutoff, high_cutoff]

        # Apply filters
        low_pass_signal  = WatermarkGenerator._apply_butterworth_filter_to_1_window(ecg_window, fs, cutoff=low_cutoff, order=param.BUTTER_ORDER, filter_type='low')
        band_pass_signal = WatermarkGenerator._apply_butterworth_filter_to_1_window(ecg_window, fs, cutoff=band_cutoff, order=param.BUTTER_ORDER, filter_type='band')
        high_pass_signal = WatermarkGenerator._apply_butterworth_filter_to_1_window(ecg_window, fs, cutoff=high_cutoff, order=param.BUTTER_ORDER, filter_type='high')

        # Compute power values
        power_low  = WatermarkGenerator._compute_signal_power_of_1_array(low_pass_signal)
        power_band = WatermarkGenerator._compute_signal_power_of_1_array(band_pass_signal)
        power_high = WatermarkGenerator._compute_signal_power_of_1_array(high_pass_signal)

        hash_values_of_window = [power_low, power_band, power_high] # 3 hash values
        return hash_values_of_window

class FragileWatermark():
    """Main class for fragile watermarking"""

    def compute_segment_power_hashes(ecg_signal_no_lsb: np.ndarray, window_indices_for_all_segments: list, num_segments_in_signal: int) -> np.ndarray:
        """Apply hash computation to each window in an ECG signal segmented into 
        multiple segments and windows, then store results in 3D array. Then reshapes
        the hash array to concatenate hash values for each segment.
        Inputting into the function a signal of S segments and each of W windows, returns 3 items 
        per window, or 3*W items for EACH SEGMENT
        - window_indices_for_all_segments (list): nested list where each segment 
        contains multiple windows, and each window is represented by its indices
        - np.ndarray: 3D array of shape (num_segments, num_windows_per_segment, 3) 
        containing the hash values for each window"""

        ecg_hash_matrix = np.full((num_segments_in_signal,
                                   param.NUM_WINDOWS_PER_SEGMENT,
                                   param.NUM_HASH_VALUES_PER_WINDOW), np.nan)

        for i, segment_indices in enumerate(window_indices_for_all_segments):
            for j, window_indices in enumerate(segment_indices):
                hash_values = WatermarkGenerator._compute_power_hash_values_of_1_window(ecg_signal_no_lsb[window_indices], param.fs)
                ecg_hash_matrix[i, j, :] = hash_values

        segment_hashes = ecg_hash_matrix.reshape(num_segments_in_signal, -1)
        return segment_hashes

    def _quantize_hash_values_for_1_segment(segment_hashes: np.ndarray, bit_length: int) -> str:
        """Converts decimal hash values to fixed-length binary sequence
        - hash_values (np.ndarray): hash values for 1 segment
        - bit_length (int): # of bits for each quantized value (default 8-bit)
        - str: concatenated binary string"""

        if type(segment_hashes) != np.ndarray:
            segment_hashes = np.array(segment_hashes)  # Convert list to NumPy array

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
            quantized_segment_hash = FragileWatermark._quantize_hash_values_for_1_segment(segment, bit_length)
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

    def _normalize_all_seeds(all_seeds_int: np.ndarray) -> np.ndarray:
        """Scales all seeds to be within 32-bit integer range (Python limit)
        - all_seeds_int (np.ndarray): array of seeds to be scaled
        - np.ndarray: array of scaled seeds"""

        min_seed = np.min(all_seeds_int)
        max_seed = np.max(all_seeds_int)

        if max_seed > param.PYTHON_SEED_LIMIT:
            # Only rescale if max_seed is too large
            normalized_seeds = (all_seeds_int - min_seed)/(max_seed - min_seed)# * param.PYTHON_SEED_LIMIT
            return np.floor(normalized_seeds)        
        return all_seeds_int # Return unchanged if within limits


    # outdated, replaced by _normalize... method
    def _scale_all_seeds(all_seeds_int: np.ndarray) -> np.ndarray:
        """Scales all seeds to be within 32-bit integer range    
        - all_seeds_int (np.ndarray): array of seeds to be scaled
        - np.ndarray: array of scaled seeds"""	

        max_seed = np.max(all_seeds_int)

        if max_seed > param.PYTHON_SEED_LIMIT:
            scale_factor  = max_seed / param.PYTHON_SEED_LIMIT  # Compute relative scaling factor
            all_seeds_int = np.floor(all_seeds_int / scale_factor)  # Scale seed down proportionally
            print(f"Max seed exceeds Python limits, will downscale all seeds")
        # all_seeds_int = np.floor(all_seeds_int)#.astype(int)
        return all_seeds_int

    def convert_hash_to_int_and_generate_watermark(ecg_segments: list, seeded_hash_segments: list) -> list:
        """Converts the seeded hash segments to integers
        - ecg_segments (list): list of ECG segments
        - seeded_hash_segments (list): list of seeded hash segments
        - np.ndarray: array of integers"""

        all_seeds_int = np.array([int(seeded_binary, 2) for seeded_binary in seeded_hash_segments])
        scaled_seeds  = FragileWatermark._normalize_all_seeds(all_seeds_int) # Scale based on largest seed

        watermarks_for_all_segments = []
        for i, seed in enumerate(scaled_seeds):
            np.random.seed(int(seed))
            watermark_for_segment = np.random.randint(0, 2, size=len(ecg_segments[i]))
            watermarks_for_all_segments.append(watermark_for_segment)

        return watermarks_for_all_segments

    # outdated
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

    def apply_lsb_watermark_to_ecg_segments(ecg_signal: np.ndarray, ecg_segments: list, watermarks_for_all_segments: list) -> list:
        """Embeds watermarks into ECG segments and returns a segmented output.
        We keep the output segmented (instead of concatenated) as it will be used in the detection
        - ecg_segments (list): list of ECG segments (each a list/array of indices)
        - watermarks_for_all_segments (list): list of watermarks for each segment
        - returns: list of watermarked ECG segments"""

        watermarked_ecg_segments = []
        for segment_indices, watermark_segment in zip(ecg_segments, watermarks_for_all_segments):
            if len(segment_indices) != len(watermark_segment):
                raise ValueError(f"Segment and watermark length mismatch: {len(segment_indices)} vs {len(watermark_segment)}")
            watermarked_segment = ecg_signal[segment_indices].copy()  # Get actual signal values

            watermarked_segment &= ~1  # Clear LSBs
            watermarked_segment |= watermark_segment  # Set new LSBs
            watermarked_ecg_segments.append(watermarked_segment)

        return watermarked_ecg_segments

    def concat_watermarked_segments(watermarked_segments: list) -> np.ndarray:
        """Concatenates the watermarked segments into a single array
        - watermarked_segments (list): list of watermarked ECG segments
        - np.ndarray: concatenated array of watermarked ECG segments"""
        return np.concatenate(watermarked_segments)


# shifted_ecg_signal, min_value   = SignalProcessing.shift_signal_up_to_remove_negative_values(robust.ecg_signal)
shifted_ecg_signal, min_value   = SignalProcessing.shift_signal_up_to_remove_negative_values(robust.watermarked_ecg_signal)
scaled_signal                   = SignalProcessing.scale_signal_and_remove_decimals(shifted_ecg_signal, param.ECG_SCALE_FACTOR)
scaled_signal_no_lsb            = SignalProcessing.remove_lsb_from_each_element_in_signal(scaled_signal)
segments_list, num_segments_in_signal= WatermarkGenerator.split_signal_to_heartbeat_segments(scaled_signal_no_lsb)
window_indices_for_all_segments = WatermarkGenerator.get_window_indices_for_all_segments(segments_list, param.SEED_K)
segment_hashes                  = FragileWatermark.compute_segment_power_hashes(scaled_signal_no_lsb, window_indices_for_all_segments, num_segments_in_signal)
quantized_segment_hashes        = FragileWatermark.quantize_hash_values_for_all_segments(segment_hashes, param.BIT_LENGTH)
seeded_hash_segments            = FragileWatermark.prepend_seed_to_every_hash(quantized_segment_hashes, param.SEED_K, param.BIT_LENGTH)
watermarks_for_all_segments     = FragileWatermark.convert_hash_to_int_and_generate_watermark(segments_list, seeded_hash_segments)
# watermarked_signal              = embed_watermark_into_ecg(scaled_signal_no_lsb, segments_list, watermarks_for_all_segments)
watermarked_ecg_segments        = FragileWatermark.apply_lsb_watermark_to_ecg_segments(scaled_signal_no_lsb, segments_list, watermarks_for_all_segments)
watermarked_signal              = FragileWatermark.concat_watermarked_segments(watermarked_ecg_segments)
watermarked_ecg_signal_unscaled = SignalProcessing.unscale_signal(watermarked_signal, param.ECG_SCALE_FACTOR)
watermarked_ecg_signal_unshifted= SignalProcessing.unshift_signal_back_to_original(watermarked_ecg_signal_unscaled, min_value)

fragile_mae = SignalAnalysis.get_mae(robust.ecg_signal, watermarked_ecg_signal_unshifted)
print(f"Fragile MAE: {fragile_mae}")


def plot_fragile_results(should_we_plot):
    """Plots the results of the fragile watermarking"""
    if should_we_plot: 
        plt.figure(figsize=(13,6))
        plt.plot(robust.ecg_signal, label="Original ECG")
        plt.plot(watermarked_ecg_signal_unshifted, label="ECG+fragile WM")
        plt.title("ECG Signal")
        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.legend()
        plt.show()

should_we_plot = 0
plot_fragile_results(should_we_plot)

# NOTE: quantized_segment_hashes bit length may be inconsistent,
# ensure bit-length normalization before binary conversion
