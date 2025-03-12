"""This module implements the Discrete Wavelet Transform (DWT) from the
paper by Raave (2024)"""

import numpy as np
import pywt
import DWT_parameters as DWT
from ECG_robust import ecg_signal

watermark_stream= np.array([1,0,1,1,0,0,0,0,1,0,0,1,0]) # key, user-given
fibonacci_seq   = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987])
dataset         = ecg_signal
wavelet_type    = 'db4' # Daubechies 4, wavelet choice
LEVEL_1         = 1

window_len                  = 5 # Barker len, from paper
split_dataset_into_sections = np.array_split(dataset, len(dataset)//(10*window_len))
dataset_split_sections      = [np.array_split(frame, DWT.NUM_LISTS_PER_SECTION) for frame in split_dataset_into_sections]

if any(len(inner_sublist) <= window_len for section in dataset_split_sections for inner_sublist in section):
    raise ValueError(f"Error: A sublist has length <= {window_len}")


def append_barker_code_to_array_beginning(array: np.array) -> np.array:
    """Prepend the 5th Barker code to the array"""
    return np.concatenate((DWT.BARKER_CODE_5TH, array), axis=0)

def add_barker_code_values_to_array(array: np.array) -> np.array:
    """Add the 5th Barker code values to the first few values in the array"""
    array[:len(DWT.BARKER_CODE_5TH)] += DWT.BARKER_CODE_5TH
    return array

def get_wavelet_approx_coeffs(array: np.array, wavelet: str, level: int) -> np.array:
    """Get the wavelet Approximation coefficients. 0 is index of approx coeff"""
    coeffs = pywt.wavedec(array, wavelet, level)
    return coeffs[0]

def get_closest_fibonacci_entry(num: int, fibonacci_seq: np.array):
    """Get the closest Fibonacci entry to input num"""
    closest_index = np.abs(fibonacci_seq - np.abs(num)).argmin()
    return fibonacci_seq[closest_index], closest_index

def split_n2_into_frames(array: np.array, frame_len: int) -> list:
    """Split n2 into frames of length frame_len"""
    return np.array_split(array, len(array)//frame_len)

# def change_fibonacci_number(closest_index, fibonacci_seq, watermark_stream, n2_frames):
#     """Change the Fibonacci number based on the watermark stream and n2 frames"""
#     for i, _ in enumerate(n2_frames):
#         if closest_index %2 == watermark_stream[i]:
#             pass # do nothing
#         else:
#             fib_number = fibonacci_seq[closest_index + 1] # increase by 1
#     return fib_number

def change_fibonacci_number(closest_index: int, fib_seq: np.array, watermark_bit: int):
    """
    If closest_index mod 2 equals watermark_bit, return fib_seq[closest_index];
    otherwise, return the next Fibonacci number.
    """
    if closest_index % 2 == watermark_bit:
        return fib_seq[closest_index]
    else:
        return fib_seq[min(closest_index + 1, len(fib_seq) - 1)]



def concat_n2_frames_into_n2(n2_frames):
    """Concatenate the n2 frames into a single array"""
    return np.concatenate(n2_frames)

def concat_n1_and_n2(n1, n2):
    """Concatenate n1 and n2 into a single array"""
    return np.concatenate((n1, n2))

# TODO; recheck change_fibonacci_number, order in paper is confusing



# c_approx   = get_wavelet_approx_coeffs(dataset, wavelet_type, LEVEL_1)
# n2_frames  = split_n2_into_frames(array, DWT.FRAME_SIZE)
# fib_number, closest_index = get_closest_fibonacci_entry(num, fibonacci_seq)
# fib_number = change_fibonacci_number(closest_index, fibonacci_seq, watermark_stream, n2_frames)
# n2 = concat_n2_frames_into_n2(n2_frames)
# n1_n2 = concat_n1_and_n2(n1, n2)

watermarked_sections = []

for i, section in enumerate(dataset_split_sections):
    processed_subsections = []
    for j, sublist in enumerate(section):
        if j%2 == 0: # n1
            # Only keep one of the 2 functions below
            # n1_with_barker = append_barker_code_to_array_beginning(sublist)
            n1_with_barker = add_barker_code_values_to_array(sublist)
            processed_subsections.append(n1_with_barker)
        else: # n2
            c_approx = get_wavelet_approx_coeffs(sublist, wavelet_type, LEVEL_1)
            n2_frames= split_n2_into_frames(c_approx, DWT.FRAME_SIZE) #split c_approx, right??
            for k, frame in enumerate(n2_frames):
                for l in range(len(frame)):
                    fib_number, closest_index = get_closest_fibonacci_entry(frame[l], fibonacci_seq)
                    new_fib_val = change_fibonacci_number(closest_index, fibonacci_seq, watermark_stream[k])
                    frame[l] = new_fib_val
                n2_frames[k] = frame
            n2_modified = concat_n2_frames_into_n2(n2_frames)
    if len(processed_subsections) == 2:
        combined = concat_n1_and_n2(processed_subsections[0], processed_subsections[1])
    else:
        combined = np.concatenate(processed_subsections)
    watermarked_sections.append(combined)

final_watermarked_data = np.concatenate(watermarked_sections)
print("Final watermarked data shape:", final_watermarked_data.shape)

                    # # TODO: see how we can change c_approx to fib_number
                    # # c_approx[i] = fib_number    OR maybe change n2_frames[i] ???
                    # n2 = concat_n2_frames_into_n2(n2_frames)  # is this even needed?
                    # n1_n2 = concat_n1_and_n2(n1_with_barker, n2)
