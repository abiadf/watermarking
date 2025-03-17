import numpy as np
import stumpy
import matplotlib.pyplot as plt

# Sample time series
ts = np.array([1, 2, 3, 4, 3, 2, 1, 3, 4, 3, 2, 1],  dtype=np.float64)

# Compute Matrix Profile with window size 4
mp = stumpy.stump(ts, m=4)

# Extract Matrix Profile values and indices of nearest subsequences
matrix_profile = mp[:, 0]
nearest_neighbors = mp[:, 1]

# Plot the matrix profile and highlight the subsequences
plt.figure(figsize=(10, 6))
plt.plot(matrix_profile, label="Matrix Profile", color='blue')

# Plot the subsequences and their nearest neighbors
for i in range(len(ts) - 4):
    start_idx = i
    end_idx = start_idx + 4
    plt.plot([start_idx, start_idx], [0, matrix_profile[i]], color='red', linestyle='--')
    plt.plot([end_idx, end_idx], [0, matrix_profile[i]], color='red', linestyle='--')

plt.xlabel("Subsequence Index")
plt.ylabel("Distance")
plt.title("Matrix Profile of Time Series with Highlighted Subsequences")
plt.legend()
plt.show()
