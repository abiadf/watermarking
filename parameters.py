"""Central location for all parameters used in the project"""

# Personal info
birth_date = '10-20-2020'
user_ssn   = 999_999_999

# Robust watermark
power                  = 0.01 # power, use 0.01-0.1 range (paper doesnt give value)
chunk_length           = 4 # chunk to input into Hamming, from paper
num_subsequences       = 4 # from me
subsequence_len_factor = 3 # from paper, where m = 3*l


# Basic timeseries info
# n_timesteps= 400 # from me
fs         = 20 # sampling freq, from me (make sure its >2x the highest dominant signal freq)
# x_values   = np.arange(0, n_timesteps/fs, 1/fs)
# ecg_signal = 1.2 * np.sin(2 * np.pi * 1 * x_values) + \
#              0.3 * np.sin(2 * np.pi * 2 * x_values) + \
#              0.08* np.sin(2 * np.pi * 6 * x_values) + \
#              0.1 * np.cos(2 * np.pi * 5 * x_values) + \
#              0.08* np.cos(2 * np.pi * 9 * x_values) + \
#              -1.5* np.exp(-((x_values - 0.3)/0.025)**2) +  \
#              2.5 * np.exp(-((x_values - 0.37)/0.018)**2) + \
#              -1.2* np.exp(-((x_values - 0.42)/0.025)**2) + \
#              0.6 * np.exp(-((x_values - 0.65)/0.04)**2)

# Peak detection
MIN_DIST_BETWEEN_PEAKS = 32
PEAK_AS_FRACTION_OF_MAX= 0.8

# Scaling/unscaling the signal
ECG_SCALE_FACTOR  = 1000 # increase to make watermark affect the series less

# Dividing the signal
WINDOW_LEN                 = 28 # make >27 OR >max(15, 3 × order) in Butterfield eq, order (4)
NUM_WINDOWS_PER_SEGMENT    = 2
NUM_HASH_VALUES_PER_WINDOW = 3 # set to 3 (low, band, high) from paper

# Hashing and watermarking
BIT_LENGTH        = 8
SEED_K            = 2 # to randomly generate interval start points
PYTHON_SEED_LIMIT = 2**32 - 1 # highest value np.seed can handle
BUTTER_ORDER      = 4 # Butterfield filter order, 4 is recommended
