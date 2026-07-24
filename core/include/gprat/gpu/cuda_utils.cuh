#ifndef GPRAT_CUDA_UTILS_HPP
#define GPRAT_CUDA_UTILS_HPP

#pragma once

#include "gprat/detail/config.hpp"
#include "gprat/target.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <hpx/algorithm.hpp>
#include <hpx/async_cuda/cuda_exception.hpp>
#include <vector>

GPRAT_NS_BEGIN

#define BLOCK_SIZE 16

using hpx::cuda::experimental::check_cuda_error;

/**
 * @brief Copies a vector from the host to the device using the next CUDA stream
 *        of gpu.
 *
 * Allocates device memory for the vector and synchronizes the stream after
 * copying the data.
 *
 * @param h_vector The vector to copy from the host
 * @param gpu The GPU target for computations
 *
 * @return A pointer to the copied vector on the device
 */
inline double *copy_to_device(const std::vector<double> &h_vector, CUDA_GPU &gpu)
{
    double *d_vector;
    check_cuda_error(cudaMalloc(&d_vector, h_vector.size() * sizeof(double)));
    cudaStream_t stream = gpu.next_stream();
    check_cuda_error(
        cudaMemcpyAsync(d_vector, h_vector.data(), h_vector.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
    check_cuda_error(cudaStreamSynchronize(stream));
    return d_vector;
}

/**
 * @brief Creates and returns a cuSolver handle.
 */
inline cusolverDnHandle_t create_cusolver_handle()
{
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    return handle;
}

/**
 * @brief Destroys the cuSolver handle.
 *
 * @param handle The cuSolver handle to destroy
 */
inline void destroy(cusolverDnHandle_t handle) { cusolverDnDestroy(handle); }

/**
 * @brief Frees the device memory allocated in a vector of shared futures.
 *
 * @param vector The vector of shared futures to free
 */
inline void free(std::vector<hpx::shared_future<double *>> &vector)
{
    for (auto &ptr : vector)
    {
        check_cuda_error(cudaFree(ptr.get()));
    }
}

GPRAT_NS_END

#endif
