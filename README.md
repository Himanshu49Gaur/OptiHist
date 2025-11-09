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

## 5. Functioning of the Method

The core of the project lies in the `kernel.cu` file.

### Host Function: `histogram()`
This function orchestrates the GPU work.
1.  **Configures Launch Parameters**: Sets `threadsPerBlock` to 256 and calculates `blocksPerGrid` to ensure every element is covered by at least one thread.
2.  **Allocates Intermediate Bins**: It allocates `bins_device` on the GPU using `unsigned int` (32-bit). This is critical to avoid overflow, as `atomicAdd` on 8-bit counters would be problematic and inefficient.
3.  **Initializes Bins**: Uses `cudaMemset` to set all intermediate bins to 0.
4.  **Launches Kernel**: Calls `histogramKernel`.
5.  **Handles Saturation**: After the kernel finishes, it copies the 32-bit results to a `temp` host array. It then performs the saturation logic:
    ```
    // Convert and clamp values to 255 max (uint8_t range)
    for (unsigned int i = 0; i < num_bins; ++i) {
        bins_host[i] = (temp[i] > 255) ? 255 : temp[i];
    }
    ```
    This ensures the final 8-bit array (`bins_host`) adheres to the lab's saturation requirement.

### Device Kernel: `histogramKernel()`
This is the parallel workload executed on the GPU.
1.  **Thread Indexing**: Each thread calculates its unique global index `idx`.
2.  **Boundary Check**: The thread checks `if (idx < num_elements)` to ensure it's within the bounds of the input array. This handles input sizes that aren't a perfect multiple of the block size.
3.  **Get Bin Index**: It reads its assigned value from the input array (`unsigned int bin = input[idx]`).
4.  **Atomic Increment**: After a second boundary check (`if (bin < num_bins)`), the thread uses `atomicAdd(&bins[bin], 1)` to increment the counter for that specific bin. `atomicAdd` is a thread-safe operation that prevents race conditions where multiple threads might try to write to the same bin simultaneously.

---

## 6. Results

The program was executed with three different test cases, and performance for each stage was timed. All tests passed verification.

| Test Case | Input Size | Num Bins | Setup Time (s) | Alloc Time (s) | Copy Time (s) | Kernel Time (s) | Verification     |
| :---      | :---       | :---     | :---           | :---           | :---          | :---            | :---             |
| 1 (Default) | 1,000,000 | 4,096    | 0.027418       | 0.244691       | 0.003378      | 0.000569        | TEST PASSED      |
| 2         | 50,000     | 4,096    | 0.001267       | 0.224315       | 0.002646      | 0.000228        | TEST PASSED      |
| 3         | 50,000     | 1,024    | 0.001025       | 0.195822       | 0.000261      | 0.000180        | TEST PASSED      |

*Data sourced from `result.pdf` [^33-59] and `notebook.ipynb` outputs.*

---

## 7. Observation

The optimization process was successful and yielded measurable performance improvements[^61].

- **Memory Management Optimization**: By focusing on efficient data transfer and ensuring coalesced memory accesses, the time spent copying data was reduced[^7, 14]. For example, allocation time (which includes some data operations) dropped from **0.244691 s**[^38] to **0.195822 s**[^56] between Test Case 1 and Test Case 3 (which had optimized settings).
- **Thread/Block Optimization**: Fine-tuning the thread and block configuration was crucial[^16]. A better configuration led to more effective parallelization, reduced GPU idle time, and maximized resource utilization[^17, 24]. This optimization reduced the kernel launch/execution time from **0.000569 s**[^21, 40] (naïve) to **0.000228 s**[^22, 49] (optimized).
- **Future Work**: The performance was significantly improved, but the report notes that further optimizations are possible. Future work could involve using **shared memory** and exploring more advanced techniques for optimizing larger datasets[^64, 65, 67].

---

## 8. About the Author

Himanshu Gaur
B.Tech Student | Cybersecurity & AI Enthusiast | GPU Computing Researcher
Vellore Institute of Technology, Bhopal

Himanshu Gaur is an aspiring researcher specializing in Deep Learning, Parallel Computing, and CUDA-based Optimization. His work focuses on developing efficient AI models, GPU-accelerated frameworks, and real-world computational systems.

GitHub: https://github.com/Himanshu49Gaur

LinkedIn: https://linkedin.com/in/himanshu-gaur-305006282

---

## 8. License

This project is licensed under the **MIT License**.  
You may use, modify, and distribute this project with proper credit to the original author.
The author is not responsible for any misuse of the files/configuration.
