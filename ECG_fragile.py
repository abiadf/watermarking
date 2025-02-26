import numpy as np
import matplotlib.pyplot as plt
import hashlib

n_timesteps= 400 # from me
fs         = 20 # sampling freq, from me (make sure its >2x the highest dominant signal freq)
x_values   = np.arange(0, n_timesteps/fs, 1/fs)
ecg_signal = 1.2 * np.sin(2 * np.pi * 1 * x_values) + \
             0.3 * np.sin(2 * np.pi * 2 * x_values) + \
             0.08* np.sin(2 * np.pi * 6 * x_values) + \
             0.1 * np.cos(2 * np.pi * 5 * x_values) + \
             0.08* np.cos(2 * np.pi * 9 * x_values) + \
             -1.5* np.exp(-((x_values - 0.3)/0.025)**2) +  \
             2.5 * np.exp(-((x_values - 0.37)/0.018)**2) + \
             -1.2* np.exp(-((x_values - 0.42)/0.025)**2) + \
             0.6 * np.exp(-((x_values - 0.65)/0.04)**2)

SHA256_OUTPUT_LEN = 256
HEX_BASE          = 16
ECG_SCALE_FACTOR  = 1000

def convert_ecg_signal_to_binary(ecg_signal: np.array) -> str:
    """Converts ECG signal > bite array > SHA256 hash > binary watermark
    Note the lengths: sha256 hash (256bit) = 64char, binary watermark = 256char"""

    ecg_in_bytes  = ecg_signal.tobytes()
    sha256_hash   = hashlib.sha256(ecg_in_bytes).hexdigest() # len = 64
    watermark_bin = bin(int(sha256_hash, HEX_BASE))[2:].zfill(SHA256_OUTPUT_LEN)
    return watermark_bin

def embed_watermark_in_ecg(ecg_signal: np.array, watermark_bin: str) -> np.array:
    """Embeds binary watermark in ECG signal, by replacing the least significant 
    bit with the watermark bit of each item. Uses a scaling factor to reduce
    the watermark's effect on the signal"""

    ecg_scaled_int = np.array(ecg_signal*ECG_SCALE_FACTOR, dtype=np.int32) # Scale + convert to int
    for i in range(len(watermark_bin)):
        ecg_scaled_int[i] = (ecg_scaled_int[i] & ~1) | int(watermark_bin[i])
        print(ecg_scaled_int[i]/ECG_SCALE_FACTOR- ecg_signal[i])
    watermarked_ecg = ecg_scaled_int/ECG_SCALE_FACTOR
    # print(watermarked_ecg - ecg_signal)
    return watermarked_ecg


def embed_watermark_my_way(ecg_signal: np.array, watermark_bin: str) -> np.array:
    # what to do with -ve values? the binary becomes -ve too

    # seems the issue is with the sign, when subtracting original-modified
    # i get -1, 0, 1. not sure how to tackle it, but its definitely a problem
    # look into how sign is handled in paper, and in binary code
    # CONFIRMED: the other function does not have the subtracted value as -1,1,0, but some
    # decimals (not sure how is that possible)

    ecg_scaled_int = np.array(ecg_signal*ECG_SCALE_FACTOR, dtype=np.int32) # Scale + convert to int
    modified_ecg_signal = np.copy(ecg_scaled_int)
    for i, bit in enumerate(watermark_bin):
        ecg_binary        = list(bin(ecg_scaled_int[i]))
        ecg_binary[-1]    = bit
        modified_ecg_component = int("".join(ecg_binary), 2)
        # print(ecg_scaled_int[i]- modified_ecg_component)
        modified_ecg_signal[i] = modified_ecg_component

    watermarked_ecg = modified_ecg_signal/ECG_SCALE_FACTOR
    return watermarked_ecg


watermark_bin   = convert_ecg_signal_to_binary(ecg_signal)
watermarked_ecg = embed_watermark_in_ecg(ecg_signal, watermark_bin)
watermarked_ecg2= embed_watermark_my_way(ecg_signal, watermark_bin)

print(np.mean(np.abs(watermarked_ecg - ecg_signal)/ecg_signal) * 100)
print(np.mean(np.abs(watermarked_ecg2 - ecg_signal)/ecg_signal) * 100)


