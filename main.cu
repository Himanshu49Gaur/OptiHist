// Include standard I/O and fixed-width integer types
#include <stdio.h>
#include <stdint.h>

// Include custom headers
#include "support.h"   // Timer and utility functions like initVector, verify, FATAL
#include "kernel.h"    // Declaration for the histogram function

int main(int argc, char* argv[])
{
    // Timer object for performance measurement
    Timer timer;

    // --- Problem Setup ---
    printf("\nSetting up the problem...");
    fflush(stdout);  // Ensure prompt is displayed immediately
    startTime(&timer);  // Start timing

    // Declare pointers for input and histogram bins
    unsigned int *in_h;   // Host input array
    uint8_t* bins_h;      // Host histogram bins (uint8_t)
    unsigned int *in_d;   // Device input array
    unsigned int num_elements;  // Number of elements in input
    unsigned int num_bins;      // Number of histogram bins

    cudaError_t cuda_ret;  // To hold CUDA API return codes for error checking

    // Parse command-line arguments
    if(argc == 1) {
        // No arguments provided → default values
        num_elements = 1000000;
        num_bins = 4096;
    } else if(argc == 2) {
        // One argument → set input size, default bin size
        num_elements = atoi(argv[1]);
        num_bins = 4096;
    } else if(argc == 3) {
        // Two arguments → set both input size and bin size
        num_elements = atoi(argv[1]);
        num_bins = atoi(argv[2]);
    } else {
        // Too many arguments or invalid usage
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./histogram            # Input: 1,000,000, Bins: 4,096"
           "\n    Usage: ./histogram <m>        # Input: m, Bins: 4,096"
           "\n    Usage: ./histogram <m> <n>    # Input: m, Bins: n"
           "\n");
        exit(0);
    }

    // Initialize host input array with random values in range [0, num_bins)
    initVector(&in_h, num_elements, num_bins);

    // Allocate memory for the histogram bins on host
    bins_h = (uint8_t*) malloc(num_bins * sizeof(uint8_t));

    // Done setting up the problem
    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n    Number of bins = %u\n", num_elements, num_bins);

    // --- Allocate Device Memory ---
    printf("Allocating device variables...");
    fflush(stdout);
    startTime(&timer);

    // Allocate input array on device (GPU)
    cuda_ret = cudaMalloc((void**)&in_d, num_elements * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess)
        FATAL("Unable to allocate device memory");

    // Synchronize to ensure allocation is complete
    cudaDeviceSynchronize();
    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));

    // --- Copy Data from Host to Device ---
    printf("Copying data from host to device...");
    fflush(stdout);
    startTime(&timer);

    // Transfer input data to GPU
    cuda_ret = cudaMemcpy(in_d, in_h, num_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
        FATAL("Unable to copy memory to the device");

    // Synchronize to ensure data transfer is complete
    cudaDeviceSynchronize();
    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));

    // --- Launch Kernel ---
    printf("Launching kernel...");
    fflush(stdout);
    startTime(&timer);

    // Call histogram function (runs CUDA kernel internally)
    histogram(in_d, bins_h, num_elements, num_bins);

    // Ensure kernel has finished executing
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess)
        FATAL("Unable to launch/execute kernel");

    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));

    // --- Verify Results ---
    printf("Verifying results...");
    fflush(stdout);

    // Compares GPU result against CPU reference
    verify(in_h, bins_h, num_elements, num_bins);

    // --- Cleanup ---
    // Free GPU memory
    cudaFree(in_d);

    // Free host memory
    free(in_h);
    free(bins_h);

    return 0;
}
// End of main.cu