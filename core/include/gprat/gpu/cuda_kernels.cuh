#ifndef GPRAT_CUDA_KERNELS_HPP
#define GPRAT_CUDA_KERNELS_HPP

#pragma once

#include "gprat/detail/config.hpp"

GPRAT_NS_BEGIN

/**
 * @brief Kernel to transpose a matrix.
 *
 * @param transposed Pointer to the transposed output matrix.
 * @param original Pointer to the original input matrix.
 * @param width Width of the original matrix.
 * @param height Height of the original matrix.
 */
__global__ void transpose(double *transposed, double *original, std::size_t width, std::size_t height);

GPRAT_NS_END

#endif
