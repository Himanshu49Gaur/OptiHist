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

  ---

## 4. Proposed Methodology

The development and execution methodology follows these steps:

1.  **Environment Setup**: The project is run in a Google Colab notebook, mounting Google Drive to access project files. The GPU (`!nvidia-smi`) and CUDA compiler (`!nvcc --version`) are verified.
2.  **Compilation**: The source files (`main.cu`, `kernel.cu`, `support.cu`) are compiled into an executable named `histogram` using `nvcc`, targeting the `sm_75` architecture (for the Colab Tesla T4 GPU).
3.  **Host Setup**: `main.cu` parses command-line arguments for input size (`m`) and bin count (`n`). It defaults to 1,000,000 elements and 4,096 bins if no arguments are given.
4.  **Memory Allocation**: Host memory is allocated for the input array and the final 8-bit bin array. Device memory is allocated for the input array.
5.  **Data Transfer (H2D)**: The host input array is copied to the device.
6.  **Kernel Launch**: The `histogram` host function is called. This function allocates device memory for 32-bit *intermediate* bins, initializes them to zero using `cudaMemset`, and then launches the `histogramKernel` with a calculated grid and block size (`BLOCK_SIZE = 256`).
7.  **Data Transfer (D2H)**: After the kernel completes, the 32-bit intermediate bin array is copied from the device back to a temporary host array.
8.  **Post-Processing (Saturation)**: The host CPU iterates through the 32-bit temporary array. For each value, it checks if it is greater than 255. If so, it stores 255 in the final 8-bit bin array; otherwise, it stores the value itself. This efficiently implements the saturation requirement.
9.  **Verification**: A `verify` function is called to compare the GPU results against a CPU-based implementation, ensuring correctness. All test cases passed[^41, 50, 59].
10. **Cleanup**: All allocated host and device memory is freed.

---

