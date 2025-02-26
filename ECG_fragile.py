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
 
ecg_bytes     = ecg_signal.tobytes()
watermark_hex = hashlib.sha256(ecg_bytes).hexdigest()
watermark_bin = bin(int(watermark_hex, 16))[2:].zfill(256)

ecg_int = np.array(ecg_signal * 1000, dtype=np.int32) # Scale + convert to int
for i in range(len(watermark_bin)):
    ecg_int[i] = (ecg_int[i] & ~1) | int(watermark_bin[i])
watermarked_ecg = ecg_int/1000

print(type(~5))



