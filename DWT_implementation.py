"""This module implements the Discrete Wavelet Transform (DWT) from the
paper by Raave (2024)"""

import numpy as np
import pywt
import DWT_parameters as DWT

from ECG_robust import ecg_signal

watermark_stream = np.array([1,0,1,1,0,0,0,0,1,0,0,1,0]) # key, user-given
dataset = ecg_signal
wavelet = 'db4' # Daubechies 4, wavelet choice

window_len = 5 # Barker len, from paper
frame_size = 11 # example in paper
split_dataset_into_sections = np.array_split(dataset, len(dataset)//(10*window_len))
split_sections = [np.array_split(frame, DWT.NUM_LISTS_PER_SECTION) for frame in split_dataset_into_sections]


approx_level_1 = []
detail_level_1 = []
for i, section in enumerate(split_sections):
    for j, sublist in enumerate(section):
        if j != 1:
            # print(sublist)
            sublist[:len(DWT.BARKER_CODE_5TH)] += DWT.BARKER_CODE_5TH
            # print(sublist)
        else:
            LEVEL_1 = 1
            coeffs = pywt.wavedec(dataset, wavelet, level= LEVEL_1)
            approx_level_1.append(coeffs[0])
            detail_level_1.append(coeffs[1])
            print(approx_level_1)
            print(detail_level_1)
    break




coeffs  = pywt.wavedec(dataset, wavelet, level= DWT.LEVEL)


# for frame in split_dataset_into_frames:
#     for frame in y: