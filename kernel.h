// Include guard to prevent multiple inclusions of this header file
#ifndef KERNEL_H
#define KERNEL_H

// Include the standard fixed-width integer types (like uint8_t)
#include <stdint.h>

// Declaration of the histogram function
// - input: pointer to the array of input values (each value is treated as a bin index)
// - bins_host: pointer to the array on the host where the final histogram will be stored (as uint8_t values, range 0-255)
// - num_elements: total number of elements in the input array
// - num_bins: total number of bins (size of the histogram)
void histogram(unsigned int* input, uint8_t* bins_host, unsigned int num_elements, unsigned int num_bins);

// End of include guard
#endif
