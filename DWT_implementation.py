"""This module implements the Discrete Wavelet Transform (DWT) embedding from the
paper by Raave (2024)"""

import matplotlib.pyplot as plt
import numpy as np
import pywt
import DWT_parameters as DWT
from ECG_robust import ecg_signal, SignalAnalysis

# options:
# ['haar']
# ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38']
# ['sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']
# ['coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17']
# ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8']
# ['rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8']
# ['dmey']
# ['gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8']
# ['mexh']
# ['morl']
# ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8']
# ['shan']
# ['fbsp']
# ['cmor']


def normalize_dataset(input_data):
    """Normalize the input data to make its mean 0 and stdev 1"""
    return (input_data - np.mean(input_data)) / np.std(input_data)

def add_barker_code_values_to_array(array: np.array) -> np.array:
    """Add the 5th Barker code values (in imaginary space!) to the first
    few array values"""
    multiplier = 0.5
    array      = array.astype(np.complex128)
    array[:len(DWT.BARKER_CODE_5TH)] += multiplier* 1j * DWT.BARKER_CODE_5TH
    return array

def get_wavelet_approx_coeffs(array: np.array, wavelet: str, level: int) -> np.array:
    """Get the wavelet Approximation coefficients. 0 is index of approx coeff"""
    coeffs = pywt.wavedec(array, wavelet, level)
    return coeffs[0], coeffs[1]

def get_closest_fibonacci_entry_idx(num: int, fibonacci_seq: np.array):
    """Get the closest Fibonacci entry to input num"""
    closest_index = np.abs(fibonacci_seq - np.abs(num)).argmin()
    return closest_index

def split_n2_into_frames(array: np.array, frame_len: int) -> list:
    """Split n2 into frames of length frame_len"""
    num_frames = len(array) // frame_len
    if num_frames > 0:
        return np.array_split(array, num_frames)
    else:
        # if array is too small for the split
        return array  # Or some other logic to handle small arrays

def compute_watermark_bit_index(k, idx) -> int:
    """Compute paper formula (p4): l = floor(i/f) + 1"""
    return (k * DWT.FRAME_SIZE + idx) // DWT.FRAME_SIZE  # Compute bit index

def change_fibonacci_number(closest_index: int, fib_seq: np.array, watermark_bit: int) -> int:
    """If closest_index mod 2 equals watermark_bit, return fib_seq[closest_index];
    otherwise, return the next Fibonacci number"""
    if closest_index % 2 == watermark_bit:
        return fib_seq[closest_index]
    else:
        return fib_seq[min(closest_index + 1, len(fib_seq) - 1)]

def apply_inverse_wavelet(c_approx, c_detail, wavelet: str, mode: str) -> np.array:
    """Apply inverse wavelet transformation, outputs an array"""
    return pywt.idwt(c_approx, c_detail, wavelet, mode)

def concat_n2_frames_into_n2(n2_frames) -> np.array:
    """Concatenate the n2 frames into a single array"""
    return np.concatenate(n2_frames)

def concat_n1_and_n2(n1: np.array, n2: np.array) -> np.array:
    """Concatenate n1 and n2 into a single array"""
    return np.concatenate((n1, n2))


def watermark_section(section, watermark_stream, fibonacci_seq, wavelet_type, LEVEL_1):
    """Process each section with watermarking"""
    processed_subsections = []

    for j, sublist in enumerate(section):
        if j % 2 == 0:  # n1
            n1_with_barker = add_barker_code_values_to_array(sublist)
            processed_subsections.append(n1_with_barker)
        else:  # n2
            c_approx, c_detail = get_wavelet_approx_coeffs(sublist, wavelet_type, LEVEL_1)
            n2_frames = split_n2_into_frames(c_approx, DWT.FRAME_SIZE)

            # Embed watermark into coefficients
            for k, frame in enumerate(n2_frames):
                for idx, coeff in enumerate(frame):
                    l = compute_watermark_bit_index(k, idx)
                    if l < len(watermark_stream):  # Ensure within bounds
                        closest_index = get_closest_fibonacci_entry_idx(coeff, fibonacci_seq)
                        frame[idx] = change_fibonacci_number(closest_index, fibonacci_seq, watermark_stream[l])
                n2_frames[k] = frame  # Update frame

            # Reconstruct n2
            n2_modified = concat_n2_frames_into_n2(n2_frames)
            n2_reconstructed = apply_inverse_wavelet(n2_modified, c_detail, wavelet_type, 'symmetric')
            processed_subsections.append(n2_reconstructed)

    return processed_subsections

def watermark_data(dataset, watermark_stream, fibonacci_seq, wavelet_type, LEVEL_1, window_len):
    """Full watermarking logic"""
    # Split the dataset into sections
    split_dataset_into_sections = np.array_split(dataset, len(dataset) // (10 * window_len))
    dataset_split_sections      = [np.array_split(frame, DWT.NUM_LISTS_PER_SECTION) for frame in split_dataset_into_sections]

    # Process each section
    watermarked_sections = []
    for section in dataset_split_sections:
        processed_subsections = watermark_section(section, watermark_stream, fibonacci_seq, wavelet_type, LEVEL_1)
        
        # Concat n1 + n2 frames
        if len(processed_subsections) == 2:  # ensures n1 and n2 are present
            combined_n1_n2 = concat_n1_and_n2(processed_subsections[0], processed_subsections[1])
        else:
            combined_n1_n2 = np.concatenate(processed_subsections)
        watermarked_sections.append(combined_n1_n2)

    # Combine all sections to form the final watermarked data
    return np.concatenate(watermarked_sections)

def plot_DWT_results(should_we_plot, signal, watermarked_signal):
    """Plots the results of the DWT watermarking"""

    if should_we_plot:
        plt.figure(figsize=(11,8))
        plt.plot(signal, label="Original")
        plt.plot(watermarked_signal, label="DWT Watermark")
        plt.title("Normalized Signal vs time")
        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.legend()
        plt.show()

# =======================
# Watermark stream and Fibonacci sequence
watermark_stream = np.array([1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0])  # user-given key
fibonacci_seq    = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987])

wavelet_type     = 'db4'  # Daubechies 4 wavelet choice
LEVEL_1          = 1
window_len       = 5  # Barker code length from paper

# Split dataset into sections
dataset                     = normalize_dataset(ecg_signal)
split_dataset_into_sections = np.array_split(dataset, len(dataset) // (10 * window_len))
dataset_split_sections      = [np.array_split(frame, DWT.NUM_LISTS_PER_SECTION) for frame in split_dataset_into_sections]

# Check validity of dataset split
if any(len(inner_sublist) <= window_len for section in dataset_split_sections for inner_sublist in section):
    raise ValueError(f"Error: A sublist has length <= {window_len}")

# Apply watermark to dataset
DWT_watermarked_data = watermark_data(dataset, watermark_stream, fibonacci_seq, wavelet_type, LEVEL_1, window_len)

# Output results
print(f"DWT Watermarked dataset {len(DWT_watermarked_data)}, input dataset {len(dataset)}")

if len(DWT_watermarked_data) < len(dataset):
    print(SignalAnalysis.get_mae(dataset[len(DWT_watermarked_data)], DWT_watermarked_data))
else:
    print(SignalAnalysis.get_mae(dataset, DWT_watermarked_data[:len(dataset)]))

print(DWT_watermarked_data[0:60])



should_we_plot = 0
plot_DWT_results(should_we_plot, dataset, DWT_watermarked_data)
