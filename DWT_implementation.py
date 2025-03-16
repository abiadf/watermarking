"""This module implements the Discrete Wavelet Transform (DWT) embedding from the
paper by Raave (2024)"""

import matplotlib.pyplot as plt
import numpy as np
import pywt
import DWT_parameters as DWT
from ECG_robust import ecg_signal
from utils import get_mae, normalize_dataset

class SignalProcessor:
    """Handle signal processing and calculations agnostic to the problem"""

    def get_wavelet_approx_coeffs(n2_array: np.array, wavelet: str, level: int) -> np.array:
        """Get the wavelet Approximation (idx 0) and Detail (idx 1) coefficients
        If n2_array is smaller than the required size to apply DWT, pad with 0.
        Throws a warning if output consists of 1 array (should be at least 2 arrays)
        n2_array: np.array, wavelet: str, level: int
        output: c_approx: np.array, c_detail: np.array"""

        if len(n2_array) < DWT.MIN_REQUIRED_SIZE:
            print(f"Warning: Array length {len(n2_array)} is smaller than required. Padding to {DWT.MIN_REQUIRED_SIZE}")
            # Pad with 0 to the required length
            padding_length= DWT.MIN_REQUIRED_SIZE - len(n2_array)
            n2_array      = np.pad(n2_array, (0, padding_length), mode='constant', constant_values=0)

        coeffs = pywt.wavedec(n2_array, wavelet, level)

        if len(coeffs) < 2:
            raise ValueError(f"Wavelet decomposition returned too few coefficients for level {level}. Ensure input is sufficiently large.")
        
        return coeffs[0], coeffs[1]
    
    def apply_inverse_wavelet(c_approx, c_detail, wavelet: str, mode: str) -> np.array:
        """Apply inverse wavelet transformation, outputs an array"""

        # if len(c_approx) != len(c_detail):
        #     # Pad the smaller one to match the size of the larger one
        #     max_len = max(len(c_approx), len(c_detail))
        #     c_approx = np.pad(c_approx, (0, max_len - len(c_approx)), mode='constant', constant_values=0)
        #     c_detail = np.pad(c_detail, (0, max_len - len(c_detail)), mode='constant', constant_values=0)
        
        return pywt.idwt(c_approx, c_detail, wavelet, mode)

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


class DWTImplementation:
    """Handles the problem-specific implementation like n1/n2 calculations from paper"""

    def add_barker_code_values_to_array(array: np.array, multiplier) -> np.array:
        """Add the 5th Barker code values (in imaginary space!) to the first
        few array values"""
        array = array.astype(np.complex128)
        array[:len(DWT.BARKER_CODE_5TH)] += multiplier * 1j * DWT.BARKER_CODE_5TH
        return array

    def get_closest_fibonacci_entry_idx(num: int, fibonacci_seq: np.array):
        """Get the closest Fibonacci entry to |input num|"""
        closest_index = np.abs(fibonacci_seq - np.abs(num)).argmin()
        return closest_index

    def split_n2_approx_into_frames(n2_approx: np.array, frame_len: int) -> list:
        """Split n2 Approximations into frames of length frame_len"""

        # if len(n2_approx) % frame_len != 0:
        #     padding_length= frame_len - (len(n2_approx) % frame_len)
        #     n2_approx     = np.pad(n2_approx,
        #                            (0, padding_length),
        #                            mode = 'constant',
        #                            constant_values = np.mean(n2_approx))
                                #    constant_values = 0)
        return np.array_split(n2_approx, len(n2_approx) // frame_len)
    
    def compute_watermark_bit_index(f_idx, idx, frame_len) -> int:
        """Compute paper formula (p4): l = floor(i/f) + 1
        i = index of the coefficient in the list of approximation coefficients
        f = Approx[n2] frame length
        l = index of the bit in the watermark bit stream.
        In reality, we need to change the formula to account for 
        the fact that we are moving along frames, so we *f_idx"""
        # return 1 + idx//frame_len 
        return (f_idx * frame_len + idx) // frame_len  # Compute bit index

    def change_fibonacci_number(closest_index: int, fib_seq: np.array, watermark_bit: int) -> int:
        """If closest_index mod 2 equals watermark_bit, return fib_seq[closest_index];
        otherwise, return the next Fibonacci number"""
        if closest_index % 2 == watermark_bit:
            return fib_seq[closest_index]
        else:
            return fib_seq[min(closest_index + 1, len(fib_seq) - 1)]

    def concat_n2_frames_into_n2(n2_frames) -> np.array:
        """Concatenate the n2 frames into a single array"""
        return np.concatenate(n2_frames)

    def concat_n1_and_n2(n1: np.array, n2: np.array) -> np.array:
        """Concatenate n1 and n2 into a single array"""
        return np.concatenate((n1, n2))


class Watermarking:
    """Class to handle watermarking logic"""

    def _watermark_1_section(section, watermark_stream, fibonacci_seq, wavelet_type, LEVEL_1):
        """Process each section with watermarking"""

        for j, subsection in enumerate(section):
            if j % 2 == 0: # n1
                n1_with_barker    = DWTImplementation.add_barker_code_values_to_array(subsection, DWT.MULTIPLIER)
            else: # n2
                c_approx, c_detail= SignalProcessor.get_wavelet_approx_coeffs(subsection, wavelet_type, LEVEL_1)
                n2_dwt_frames     = DWTImplementation.split_n2_approx_into_frames(c_approx, DWT.N2_FRAME_SIZE)

                # Embed watermark into coefficients
                for f_idx, frame in enumerate(n2_dwt_frames):
                    for idx, coeff in enumerate(frame):
                        l = DWTImplementation.compute_watermark_bit_index(f_idx, idx, DWT.N2_FRAME_SIZE)
                        if l < len(watermark_stream): # Ensure within bounds
                            closest_fib_index= DWTImplementation.get_closest_fibonacci_entry_idx(coeff, fibonacci_seq)
                            frame[idx]       = DWTImplementation.change_fibonacci_number(closest_fib_index, fibonacci_seq, watermark_stream[l])
                    n2_dwt_frames[f_idx] = frame # Update frame

                # Reconstruct n2
                n2_dwt_modified = DWTImplementation.concat_n2_frames_into_n2(n2_dwt_frames)
                n2_reconstructed= SignalProcessor.apply_inverse_wavelet(n2_dwt_modified, c_detail, wavelet_type, 'symmetric')

        return np.concatenate([n1_with_barker, n2_reconstructed])

    def watermark_data(dataset, watermark_stream, fibonacci_seq, wavelet_type, LEVEL_1):
        """Watermark the entire dataset while keeping sections intact"""

        dataset_split_into_sections    = np.array_split(dataset, len(dataset) // DWT.DATASET_SECTION_LEN)
        dataset_sections_split_to_n1_n2= [np.array_split(section, DWT.NUM_LISTS_PER_SECTION) for section in dataset_split_into_sections]

        watermarked_sections = []
        for section in dataset_sections_split_to_n1_n2:
            processed_section = Watermarking._watermark_1_section(section, watermark_stream,
                                                                  fibonacci_seq, wavelet_type,
                                                                  LEVEL_1)
            watermarked_sections.append(processed_section) # Preserves section structure
        
        return np.concatenate(watermarked_sections) #watermarked_sections

# ===============================================

# Watermark dataset
dataset             = normalize_dataset(np.tile(ecg_signal, 2)) #normalize_dataset(ecg_signal)
dwt_watermarked_data= Watermarking.watermark_data(dataset,
                                                  DWT.watermark_stream,
                                                  DWT.fibonacci_seq,
                                                  DWT.wavelet_type,
                                                  DWT.LEVEL_1)
# Output results
print(f"Input dataset {len(dataset)}, DWT Watermarked dataset {len(dwt_watermarked_data)}")
if len(dwt_watermarked_data) > len(dataset):
    dwt_watermarked_data = dwt_watermarked_data[:len(dataset)]
else:
    dataset = dataset[:len(dwt_watermarked_data)]
print("DWT MAE:", get_mae(dataset, dwt_watermarked_data))

should_we_plot = 0
SignalProcessor.plot_DWT_results(should_we_plot, dataset, dwt_watermarked_data)
