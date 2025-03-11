# Watermarking
This repo focuses on a novel way to embed watermarks into timeseries.

## Part 1 - Implementing ECG paper
[Paper link](https://www.ee.bilkent.edu.tr/~kozat/papers/5_1.pdf)

The first part is to mimic the implementation of Kozat et al. (2009), who embedded patients' data into their ECG signals using 2 watermarks, a robust one to resist attacks, and a fragile one to detect tampering. Upon reproducing Kozat et al.'s work, we get an understanding of their method, what went well, and what could be improved.

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

Diagram `ECG_diagram` in this repo outlines the steps to implement Kozat et al.'s algorithm.


## Part 2 - Implementing DWT paper
[Paper link](https://repository.tudelft.nl/file/File_21f57473-c108-46fa-94f0-c196773465b5?preview=1)

The algorithm is outlined in a diagram found in this repo