#ifndef GPRAT_GPU_GP_UNCERTAINTY_HPP
#define GPRAT_GPU_GP_UNCERTAINTY_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include "gprat/target.hpp"

GPRAT_NS_BEGIN

namespace gpu
{

/**
 * @brief Retrieve diagonal elements of posterior covariance matrix.
 *
 * @param A Diagonal elements matrix A
 * @param B Diagonal elements matrix B
 * @param M Number of rows in the matrix
 *
 * @return Diagonal elements of posterior covariance matrix
 */
hpx::shared_future<double *> diag_posterior(
    const hpx::shared_future<double *> A, const hpx::shared_future<double *> B, std::size_t M, CUDA_GPU &gpu);

/**
 * @brief Retrieve diagonal elements of posterior covariance matrix.
 *
 * @param A Posterior covariance matrix
 * @param M Number of rows in the matrix
 *
 * @return Diagonal elements of posterior covariance matrix
 */
hpx::shared_future<double *> diag_tile(const hpx::shared_future<double *> A, std::size_t M, CUDA_GPU &gpu);

}  // end of namespace gpu

GPRAT_NS_END

#endif
