"""This module runs all parts of the ECG algorithm, tailored to the ECG scenario"""

import math
import numpy as np

import ECG.ECG_parameters as param
from ECG.ECG_robust import Preprocessing, WatermarkEmbedding, SignalAnalysis
from utils import get_mae

# %% Robust embedding
binary_ssn       = Preprocessing.convert_ssn_to_binary(param.user_ssn)
binary_ssn_split = Preprocessing.split_and_pad_binary_ssn(binary_ssn, binary_ssn_chunk_len = 4)
ssn_with_hamming = Preprocessing.apply_hamming_to_all_ssn_chunks(binary_ssn_split)

subsequence_length = param.subsequence_len_factor * len(ssn_with_hamming) # m = 3*l (from paper)
n_timesteps        = math.floor(subsequence_length * param.num_subsequences) # from paper
x_values           = np.arange(0, n_timesteps/param.fs, 1/param.fs)

ecg_signal = 1.2 * np.sin(2 * np.pi * 1 * x_values) + \
             0.3 * np.sin(2 * np.pi * 2 * x_values) + \
             0.08* np.sin(2 * np.pi * 6 * x_values) + \
             0.1 * np.cos(2 * np.pi * 5 * x_values) + \
             0.08* np.cos(2 * np.pi * 9 * x_values) + \
             -1.5* np.exp(-((x_values - 0.3)/0.025)**2) +  \
             2.5 * np.exp(-((x_values - 0.37)/0.018)**2) + \
             -1.2* np.exp(-((x_values - 0.42)/0.025)**2) + \
             0.6 * np.exp(-((x_values - 0.65)/0.04)**2)

watermark_sequence     = WatermarkEmbedding._turn_watermark_to_nonbinary_sequence(ssn_with_hamming)
ecg_subsequences       = WatermarkEmbedding._split_signal_to_subsequences(ecg_signal, subsequence_length, n_timesteps)
watermarked_ecg_signal = WatermarkEmbedding.get_watermarked_subsequences(ecg_subsequences, watermark_sequence)

mae  = get_mae(ecg_signal, watermarked_ecg_signal)
mape = np.mean(np.abs((ecg_signal - watermarked_ecg_signal)/ecg_signal)) * 100
# print(f"Robust: MAE {mae}%, MAPE {mape}%")

should_we_plot = 0
SignalAnalysis.plot_robust_results(should_we_plot, ecg_signal, watermarked_ecg_signal)
