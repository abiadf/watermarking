"""Central location for all parameters used in the DWT part"""

import numpy as np

BARKER_CODE_5TH = np.array([1, 1, 1, 0, 1]) # 5th Barker code
MULTIPLIER      = 0.5 # Barker code multipluer, user-defined
window_len      = 5 # Barker code length from paper

NUM_LISTS_PER_SECTION= 2 #from paper, n1 and n2
N2_FRAME_SIZE        = 11 # example in paper
DATASET_SECTION_LEN  = 100 * window_len

wavelet_type     = 'db4' # Daubechies 4 wavelet (not in paper, personal choice)
LEVEL_1          = 1
LEVEL = 5 # how many times DWT is applied

fibonacci_seq    = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987])
