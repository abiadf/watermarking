"""This module implements the fragile watermark from the ECG paper"""

import numpy as np
import matplotlib.pyplot as plt
import hashlib
import scipy.signal

n_timesteps= 400 # from me
fs         = 20 # sampling freq, from me (make sure its >2x the highest dominant signal freq)
x_values   = np.arange(0, n_timesteps/fs, 1/fs)
ecg_signal = 1.2 * np.sin(2 * np.pi * 1 * x_values) + \
             0.3 * np.sin(2 * np.pi * 2 * x_values) + \
             0.08* np.sin(2 * np.pi * 6 * x_values) + \
             0.1 * np.cos(2 * np.pi * 5 * x_values) + \
             0.08* np.cos(2 * np.pi * 9 * x_values) + \
             -1.5* np.exp(-((x_values - 0.3)/0.025)**2) +  \
             2.5 * np.exp(-((x_values - 0.37)/0.018)**2) + \
             -1.2* np.exp(-((x_values - 0.42)/0.025)**2) + \
             0.6 * np.exp(-((x_values - 0.65)/0.04)**2)

SHA256_OUTPUT_LEN = 256
HEX_BASE          = 16
ECG_SCALE_FACTOR  = 1000 #inversely proportional to watermark strength

def scale_signal_and_remove_decimals(ecg_signal, ECG_SCALE_FACTOR):
    """The ECG contains decimals, which are not well handled by the LSB algo, so we scale the signal to 
    minimize the damage, then remove decilams, and make it of type int. Ex: ECG signal y-value of 3.51 needs to be int first, so becomes 3, then applying
    LSB gives 3-1=2. We lost too much info, so let's scale 3.5 by 100 (for ex)= 351, then do LSB -> 350, we
    lost less info"""
    return np.floor(ecg_signal * ECG_SCALE_FACTOR).astype(int)

def remove_lsb_from_each_element_in_signal(scaled_signal):
    """The least significant bit (LSB) is the last bit in a number's binary form, ex: 19 (dec) = 10011 (bin)
    The LSB here is 1. This function intakes an array, clears the LSB to 0, equivalent to subtracting 1 when
    the number is odd, or leaving it unchanged it even"""
    modified_scaled_signal = scaled_signal & ~1
    return modified_scaled_signal

def unscale_signal(modified_scaled_signal, ECG_SCALE_FACTOR):
    """Function that unscaled the modified signal"""
    ecg_signal_no_lsb = modified_scaled_signal / ECG_SCALE_FACTOR
    return ecg_signal_no_lsb

def get_signal_peaks(ecg_signal_no_lsb, distance_between_points = 30, peak_as_fraction_of_max = 0.8):
    """Given a signal, get the peak indices and heights"""
    min_peak_height               = np.max(ecg_signal_no_lsb) * peak_as_fraction_of_max
    peak_indices, peak_properties = scipy.signal.find_peaks(ecg_signal_no_lsb, height = min_peak_height, distance= distance_between_points)
    peak_heights                  = peak_properties['peak_heights']  # Extract peak heights
    return peak_indices, peak_heights

# below function doesnt necessarily produce equally sized intervals, should fix
def split_signal_to_heartbeat_segments(ecg_signal_no_lsb, peak_indices) -> list:
    """Divide ECG signal to equal segments so each contains a heartbeat (no strict requirement). 
    We do this by dividing on indices"""
    segments_list = []
    for i in range(len(peak_indices) - 1):
        start = peak_indices[i]
        end   = peak_indices[i+1]
        segments_list.append(ecg_signal_no_lsb[start:end])
    return segments_list

def _set_intervals_in_1_segment(segment_array, num_of_intervals = 5, window_length = 20) -> list:
    """Given a segment, generate random interval windows within segment
    INPUT:
    - num_of_intervals, how many intervals we will have in segment (= p from paper)
    - window_length, (= w from paper)
    OUTPUT:
    - intervals_start_end: tuple of (start_point, end_point) of interval"""

    segment_length = len(segment_array) # take 1st segment
    if segment_length < window_length:
        print("Segment is too small/window too long")
        return[]

    starting_points  = np.random.randint(0, segment_length - window_length, size = num_of_intervals)

    intervals_start_end = [(start, start + window_length) for start in starting_points]
    return intervals_start_end

def set_interval_bounds_for_all_segments(segments_list: list, seed_k: int = 2) -> list:
    """Generate interval windows for all segments by looping function that does
    it for each segment
    - segments_list: list of ECG segments
    - seed_k: random seed (= k from paper)"""
    np.random.seed(seed_k)

    segment_intervals_start_end = []
    for segment in segments_list:
        intervals_start_end = _set_intervals_in_1_segment(segment)
        segment_intervals_start_end.append(intervals_start_end)
    return segment_intervals_start_end


scaled_signal              = scale_signal_and_remove_decimals(ecg_signal, ECG_SCALE_FACTOR)
modified_scaled_signal     = remove_lsb_from_each_element_in_signal(scaled_signal)
ecg_signal_no_lsb          = unscale_signal(modified_scaled_signal, ECG_SCALE_FACTOR)
peak_indices, peak_heights = get_signal_peaks(ecg_signal_no_lsb)
segments_list              = split_signal_to_heartbeat_segments(ecg_signal_no_lsb, peak_indices)
segment_intervals_start_end= set_interval_bounds_for_all_segments(segments_list, 2)


