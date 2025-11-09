# CudaHist: CUDA-Based Saturated Histogram

A CUDA implementation of an efficient histogram algorithm with 8-bit saturated counters (max 255). Developed for the ECE5550 High Performance Computing lab, this project focuses on gradual kernel optimization techniques.

## 1. Problem Statement

The challenge is to implement an efficient histogramming algorithm for a large input array of integers[^72]. The program must be parallelized using CUDA to run on a GPU[^77].

A key constraint is that the histogram bins must use **unsigned 8-bit counters**. These counters must be **saturated at 255**, meaning any attempt to increment a bin count beyond 255 should result in the count remaining at 255, not rolling over to 0[^74]. The solution must also handle varying input sizes and bin counts, paying close attention to boundary conditions[^105].
