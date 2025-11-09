# CudaHist: CUDA-Based Saturated Histogram

A CUDA implementation of an efficient histogram algorithm with 8-bit saturated counters (max 255). Developed for the ECE5550 High Performance Computing lab, this project focuses on gradual kernel optimization techniques.

---

## 1. Problem Statement

The challenge is to implement an efficient histogramming algorithm for a large input array of integers[^72]. The program must be parallelized using CUDA to run on a GPU[^77].

A key constraint is that the histogram bins must use **unsigned 8-bit counters**. These counters must be **saturated at 255**, meaning any attempt to increment a bin count beyond 255 should result in the count remaining at 255, not rolling over to 0[^74]. The solution must also handle varying input sizes and bin counts, paying close attention to boundary conditions[^105].

---

## 2. Objective

The primary objective is to implement and optimize an efficient histogramming algorithm using CUDA, leveraging the GPU's parallel computation capabilities for large datasets[^2, 3].

Secondary goals include:
- Correctly implementing the 8-bit saturated counter logic[^74, 90].
- Optimizing performance by tuning memory management (Host-to-Device/Device-to-Host transfers) and thread/block configurations[^4, 6, 16].
- Verifying the correctness of the GPU-computed results[^4, 27].

---

## 3. Proposed Solution

The solution is a CUDA-based application developed in a Google Colab environment[^75]. It consists of:

- **`main.cu`**: The host driver program that handles problem setup, parsing command-line arguments (for input size and bin count), allocating host/device memory, managing data transfers, and timing each stage of the process.
- **`kernel.cu`**: Contains the core logic.
  1.  **`histogramKernel` (Device Kernel)**: A CUDA kernel where each thread processes one element of the input array. It uses `atomicAdd` to safely increment bin counters in global memory, preventing race conditions.
  2.  **`histogram` (Host Function)**: A wrapper function that launches the kernel and handles the 8-bit saturation. It first computes the histogram into a temporary 32-bit integer array on the device (to prevent atomic operation overflow). It then copies this result to the host, iterates through it, and clamps (saturates) any value greater than 255 down to 255 before storing it in the final `uint8_t` host array.
- **`notebook.ipynb`**: A Google Colab notebook used to set up the CUDA environment, compile the code using `nvcc`, and execute the program with various test cases.
