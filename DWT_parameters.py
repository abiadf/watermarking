"""Central location for all parameters used in the DWT part"""

import numpy as np

BARKER_CODE_5TH = np.array([1, 1, 1, 0, 1]) # 5th Barker code
LEVEL = 5 # how many times DWT is applied

NUM_LISTS_PER_SECTION = 2 #from paper, n1 and n2

FRAME_SIZE = 11 # example in paper
