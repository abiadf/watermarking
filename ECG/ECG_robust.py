"""This module implements the robust watermarking from the ECG paper.
Given data with x and y axes, this module watermarks the y-axis only"""

import numpy as np
import matplotlib.pyplot as plt
 
import ECG.ECG_parameters as param

class Preprocessing:
    @staticmethod
    def convert_ssn_to_binary(user_ssn: int) -> str:
        """Converts a Social Security Number (SSN) from decimal to binary format
        Removes the first 2 binary characters ('0b') added by the bin() function
        - user_ssn (int): The SSN in decimal format
        - str: Binary representation of the SSN (excluding '0b' prefix)"""
        return bin(user_ssn)[2:]
    
    @staticmethod
    def split_and_pad_binary_ssn(binary_ssn: str, binary_ssn_chunk_len: int = 4) -> list:
        """Splits binary SSN into chunks of certain length + right-pads the last chunk
        with zeros if it's not the desired length
        INPUT:
        binary_ssn (str): Binary SSN
        binary_ssn_chunk_len (int): Desired length of output chunks, by default 4
        OUTPUT: binary_ssn_split (list): list of binary strings, each of length binary_ssn_chunk_len"""
    
        number_to_pad_with = '0'
        binary_ssn_split   = [binary_ssn[i: i + binary_ssn_chunk_len] for i in range(0, len(binary_ssn), binary_ssn_chunk_len)]
        if len(binary_ssn_split[-1]) != binary_ssn_chunk_len: #righ pad
            binary_ssn_split[-1] = binary_ssn_split[-1].ljust(binary_ssn_chunk_len, number_to_pad_with)
        return binary_ssn_split
        
    @staticmethod
    def _apply_hamming_for_1_chunk(binary_chunk: str) -> str:
        """Encodes a 4-bit binary sequence using Hamming(7,4) error correction, by adding
        3 parity bits (so the total length is 7). Parity bits are calculated as follows:
        INPUT: 4-bit binary_chunk (str) like 'd1 d2 d3 d4' = '1110'
        OUTPUT: 7-bit Hamming(7,4) encoded str of item, like 'p1 p2 d1 p3 d2 d3 d4'
        p1 checks if (d1 + d2 + d4) is even, then p1=0, else p1=1
        p2 checks if (d1 + d3 + d4) is even, then p2=0, else p2=1
        p3 checks if (d2 + d3 + d4) is even, then p3=0, else p3=1"""

        p1 = (param.chunk_length - (int(binary_chunk[0])+ int(binary_chunk[1])+ int(binary_chunk[3])) )%2
        p2 = (param.chunk_length - (int(binary_chunk[0])+ int(binary_chunk[2])+ int(binary_chunk[3])) )%2
        p3 = (param.chunk_length - (int(binary_chunk[1])+ int(binary_chunk[2])+ int(binary_chunk[3])) )%2
    
        binary_item_with_hamming = str(p1) + str(p2) + binary_chunk[0] + str(p3) + \
                                binary_chunk[1] + binary_chunk[2] + binary_chunk[3]
        return binary_item_with_hamming
    
    @staticmethod
    def apply_hamming_to_all_ssn_chunks(binary_ssn_split: list) -> str:
        """Loops the Hamming function to all chunks, resulting in watermark W
        INPUT: binary_ssn_split (list): list of binary strings, each of length binary_ssn_chunk_len
        OUTPUT: ssn_with_hamming (str): combined sequence of all Hamming chunks (each 7digits)
        total length 7*8=56"""
    
        ssn_with_hamming = ''
        for ssn_chunk in binary_ssn_split:
            ssn_with_hamming += Preprocessing._apply_hamming_for_1_chunk(ssn_chunk)
        return ssn_with_hamming


class WatermarkEmbedding:
    def _turn_watermark_to_nonbinary_sequence(ssn_with_hamming: str) -> np.array:
        """This function turns the binary watermark info of [0,1] to [-1,1];
        first it x2 the sequence so it becomes [2,0] then subtracts 1 to get [-1,1]
        Then we add a [0] at the beginning to make sure that when adding to
        Fourier coeffs, it does not add to the 1st Fourier term ('DC component')
            - ssn_with_hamming (str): Hamming-encoded binary watermark string
            - np.array: Array containing watermark values in the range [-1,1]"""

        watermark = [0] + [2*int(i)-1 for i in ssn_with_hamming] # from [0,1] to [-1,1]
        return np.array(watermark)
    
    def _split_signal_to_subsequences(ecg_signal, subsequence_length: int, n_timesteps: int) -> list:
        """Splits an ECG signal into non-overlapping subsequences. The last subsequence
        might be shorter than others if signal length is not a multiple of `subsequence_length`
            - ecg_signal (array-like): input ECG signal
            - subsequence_length (int): Length of each subsequence
            - n_timesteps (int): Total number of timesteps in the ECG signal
            - output (list): List of ECG subsequences"""

        # ecg_subsequences = [ecg_signal[i:i+subsequence_length]
        #                     for i in range(0, n_timesteps - subsequence_length + 1, subsequence_length)]

        ecg_subsequences = [ecg_signal[i:i+subsequence_length]
                            for i in range(0, n_timesteps, subsequence_length)]
        return ecg_subsequences

    def _get_fourier_terms(ecg_subseq: np.array) -> tuple:
        """Calculates Fourier transform terms of a subsequence
            - ecg_subseq (np.array): A segment of the ECG signal
            - tuple: (FFT coefficients, Magnitudes, Phase angles)"""

        subseq_fft   = np.fft.fft(ecg_subseq)  # Apply FFT
        magnitudes   = np.abs(subseq_fft)
        phase_angles = np.angle(subseq_fft)
        return subseq_fft, magnitudes, phase_angles

    def _apply_watermark_to_subsequences(watermark: np.array, magnitudes: np.array, power) -> list:
        """Embeds watermark to the Fourier magnitude spectrum of an ECG sequence.
        The watermark modifies the first few Fourier magnitudes while ensuring non-negative values
        Note that 1st item in watermark is 0, so the watermark is 1 bit longer than the subsequence/3
            - watermark (np.array): The non-binary watermark sequence
            - magnitudes (np.array): The original Fourier magnitudes of the ECG subsequence
            - power (float): Scaling factor for watermark strength
            - output (np.array): Modified Fourier magnitudes with the embedded watermark"""

        modified_magnitudes                  = magnitudes.copy()
        modified_magnitudes[:len(watermark)] = np.maximum(0, magnitudes[:len(watermark)] + power*watermark)
        return modified_magnitudes

    def get_watermarked_subsequences(ecg_subsequences, watermark_sequence):
        """Given the ECG subsequences, get the whole watermarked series
            - ecg_subsequences (list): List of ECG signal subsequences
            - watermark_sequence (np.array): Watermark sequence to be embedded
            - output (np.array): Watermarked ECG signal"""

        watermarked_subsequences = []
        for ecg_subseq in ecg_subsequences:
            subseq_fft, magnitudes, phase_angles = WatermarkEmbedding._get_fourier_terms(ecg_subseq)
        
            modified_magnitude  = WatermarkEmbedding._apply_watermark_to_subsequences(watermark_sequence, magnitudes, param.power)
            modified_fft_series = modified_magnitude * np.exp(1j * phase_angles)
            watermarked_subseq  = np.fft.ifft(modified_fft_series).real # keep real part
            watermarked_subsequences.append(watermarked_subseq)
            # print(f"Subseq: {get_mae(ecg_subseq, watermarked_subseq)} %")

        watermarked_ecg_signal = np.concatenate(watermarked_subsequences)
        
        return watermarked_ecg_signal


class SignalAnalysis:
    """Handles signal analysis and plotting"""

    @staticmethod
    def calculate_beta(rho_values, num_subsequences):
        """Calculates beta, as defined in the paper"""
        beta = np.sum(rho_values)/num_subsequences
        return beta

    @staticmethod
    def plot_robust_results(should_we_plot, ecg_signal, watermarked_ecg_signal):
        """Plots the results of the robust watermarking"""
        freqs = np.fft.fftfreq(len(ecg_signal), 1/param.fs)
        subseq_fft, magnitudes, phase_angles = WatermarkEmbedding._get_fourier_terms(ecg_signal)
        phases = phase_angles

        if should_we_plot:
            plt.figure(figsize=(13,6))
            plt.subplot(1,2,1)
            plt.plot(ecg_signal, label="Original ECG")
            plt.plot(watermarked_ecg_signal, label="Watermarked ECG")
            plt.title("ECG Signal")
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.legend()

            plt.subplot(1,2,2)
            plt.plot(freqs[:len(freqs)//2], phases[:len(freqs)//2])  # Magnitudes for positive frequencies
            plt.title("ECG Frequency Spectrum")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.show()

