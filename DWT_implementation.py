"""This module implements the Discrete Wavelet Transform (DWT) from the
paper by Raave (2024)"""

import numpy as np
import pywt
import DWT_parameters as DWT

dataset = np.array([41, 23, 9, 31, 2, 5, 16, 14, 3, 14, 25,
                    16, 2, 3, 6, 1, 3, 5, 22, 12, 7,15])
wavelet = 'db4' # Daubechies 4, wavelet choice
coeffs  = pywt.wavedec(dataset, wavelet, level= DWT.LEVEL)

frame_size = 11 # example in paper
window_len = 5 # Barker len, from paper
split_dataset_into_frames = np.array_split(dataset, len(dataset)//window_len)


DWT.BARKER_CODE_5TH

# for frame in split_dataset_into_frames:
#     for frame in y: