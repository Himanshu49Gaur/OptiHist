// Include necessary headers for CUDA runtime API and standard integer types
#include <cuda_runtime.h>
#include <stdint.h>

// Define the number of threads per block to use in CUDA kernel execution
#define BLOCK_SIZE 256

// CUDA Kernel Function: Computes a histogram on the GPU
__global__ void histogramKernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {
    // Compute the global index of the thread in the grid
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the thread index is within the valid range
    if (idx < num_elements) {
        // Read the value from the input array at position `idx`
        unsigned int bin = input[idx];

        // Only process if bin index is within bounds
        if (bin < num_bins) {
            // Atomically increment the bin count to avoid race conditions
            atomicAdd(&bins[bin], 1);
        }
    }
}

// Host Function: Sets up and runs the CUDA kernel to compute the histogram
void histogram(unsigned int* input, uint8_t* bins_host, unsigned int num_elements, unsigned int num_bins) {
    // Set number of threads per block
    int threadsPerBlock = BLOCK_SIZE;

    // Compute how many blocks are needed to process all elements
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory on the GPU to store histogram bins (unsigned int type)
    unsigned int* bins_device;
    cudaMalloc((void**)&bins_device, num_bins * sizeof(unsigned int));

    // Initialize GPU memory for bins to zero
    cudaMemset(bins_device, 0, num_bins * sizeof(unsigned int));

    // Launch the histogram kernel on the GPU
    histogramKernel<<<blocksPerGrid, threadsPerBlock>>>(input, bins_device, num_elements, num_bins);

    // Ensure the kernel has completed execution before moving on
    cudaDeviceSynchronize();

    // Allocate temporary memory on the host (CPU) to store the results
    unsigned int* temp = (unsigned int*)malloc(num_bins * sizeof(unsigned int));

    // Copy the histogram data from device (GPU) to host (CPU)
    cudaMemcpy(temp, bins_device, num_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Convert and clamp values to 255 max (uint8_t range), and store in bins_host
    for (unsigned int i = 0; i < num_bins; ++i) {
        bins_host[i] = (temp[i] > 255) ? 255 : temp[i]; // Clamp values to avoid overflow in 8-bit bins
    }

    // Free the temporary memory on host
    free(temp);

    // Free the memory on device
    cudaFree(bins_device);
}
