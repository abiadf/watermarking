
## Running
The files in this dir should be run via modules in the parent dir:
- `run_ECG.py`: stays true to the ECG scenario, or
- `run_ECG.ipynb`: tailored to datasets and detection of TimeWak


## Implementing ECG paper
The ECG algorithm [found in this paper](https://www.ee.bilkent.edu.tr/~kozat/papers/5_1.pdf)
embeds robust + fragile watermark, but only detects the fragile one.

This dir is concerned with implementing the work of Kozat et al. (2009), who embedded patients' data into their ECG signals using 2 watermarks, a robust one to resist attacks, and a fragile one to detect tampering. Upon reproducing Kozat et al.'s work, we get an understanding of their method, what went well, and what could be improved.

Robust watermark:
- Get patients' Social Security Number (SSN)
- Convert to binary
- Add Hamming(7,4)
- Get Fourier frequencies
- Split watermark into multiple frequencies (but not the first one)

Fragile watermark:
- Get ECG timeseries
- Remove the Lease Significant Bit from each point
- Split timeseries to individual heartbeat `segments`
- Generate p windows of random starting location, for each segment
- Generate 3 hash values for each interval (low, band and high pass filter power)
- Convert these hash values to binary
- Append seed $\kappa$ to hash
- Generate random binary vector based on this seed value, of same length as segment; this is the watermark
- Replace the LSB of each point in a segment with the respective value of this segment's watermark

The algorithm is outlined in a diagram found in this repo, `diagrams/ECG_diagram`
