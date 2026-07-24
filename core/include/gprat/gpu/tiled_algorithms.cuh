#ifndef GPRAT_GPU_TILED_ALGORITHMS_HPP
#define GPRAT_GPU_TILED_ALGORITHMS_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include "gprat/hyperparameters.hpp"
#include "gprat/target.hpp"
#include "gprat/kernels.hpp"

#include <cusolverDn.h>
#include <hpx/modules/async_cuda.hpp>

GPRAT_NS_BEGIN

namespace gpu
{

// Tiled Cholesky Algorithm

/**
 * @brief Perform right-looking Cholesky decomposition.
 *
 * @param n_streams Number of CUDA streams.
 * @param ft_tiles Matrix represented as a vector of tiles, containing the
 *        covariance matrix, afterwards the Cholesky decomposition.
 * @param n_tile_size Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 * @param gpu GPU target for computations.
 * @param cusolver cuSolver handle, already created.
 */
void right_looking_cholesky_tiled(std::vector<hpx::shared_future<double *>> &ft_tiles,
                                  const std::size_t n_tile_size,
                                  const std::size_t n_tiles,
                                  CUDA_GPU &gpu,
                                  const cusolverDnHandle_t &cusolver);

// Tiled Triangular Solve Algorithms

/**
 * @brief Perform tiled forward triangular matrix-vector solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side vector, afterwards containing the tiled solution vector
 * @param n_tile_size Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 * @param gpu GPU target for computations.
 */
void forward_solve_tiled(std::vector<hpx::shared_future<double *>> &ft_tiles,
                         std::vector<hpx::shared_future<double *>> &ft_rhs,
                         const std::size_t n_tile_size,
                         const std::size_t n_tiles,
                         CUDA_GPU &gpu);

/**
 * @brief Perform tiled backward triangular matrix-vector solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side vector, afterwards containing the tiled solution vector
 * @param n_tile_size Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 * @param gpu GPU target for computations.
 */
void backward_solve_tiled(std::vector<hpx::shared_future<double *>> &ft_tiles,
                          std::vector<hpx::shared_future<double *>> &ft_rhs,
                          const std::size_t n_tile_size,
                          const std::size_t n_tiles,
                          CUDA_GPU &gpu);

/**
 * @brief Perform tiled forward triangular matrix-matrix solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side matrix, afterwards containing the tiled solution matrix.
 * @param n_tile_size Tile size of first dimension.
 * @param m_tile_size Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 * @param gpu GPU target for computations.
 */
void forward_solve_tiled_matrix(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    CUDA_GPU &gpu);

/**
 * @brief Perform tiled backward triangular matrix-matrix solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side matrix, afterwards containing the tiled solution matrix.
 * @param n_tile_size Tile size of first dimension.
 * @param m_tile_size Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 * @param gpu GPU target for computations.
 */
void backward_solve_tiled_matrix(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    CUDA_GPU &gpu);

/**
 * @brief Perform tiled matrix-vector multiplication
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_vector Tiled vector represented as a vector of futurized tiles.
 * @param ft_rhsTiled solution represented as a vector of futurized tiles.
 * @param N_row Tile size of first dimension.
 * @param N_col Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 * @param gpu GPU target for computations.
 */
void matrix_vector_tiled(std::vector<hpx::shared_future<double *>> &ft_tiles,
                         std::vector<hpx::shared_future<double *>> &ft_vector,
                         std::vector<hpx::shared_future<double *>> &ft_rhs,
                         const std::size_t N_row,
                         const std::size_t N_col,
                         const std::size_t n_tiles,
                         const std::size_t m_tiles,
                         CUDA_GPU &gpu);

/**
 * @brief Perform tiled symmetric k-rank update on diagonal tiles
 *
 * @param ft_tCC_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_inter_tiles Tiled vector holding the diagonal tile results
 * @param n_tile_size Tile size of first dimension.
 * @param m_tile_size Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 * @param gpu GPU target for computations.
 */
void symmetric_matrix_matrix_diagonal_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<double *>> &ft_inter_tiles,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    CUDA_GPU &gpu);

void compute_gemm_of_invK_y(std::vector<hpx::shared_future<double *>> &ft_invK,
                            std::vector<hpx::shared_future<double *>> &ft_y,
                            std::vector<hpx::shared_future<double *>> &ft_alpha,
                            const std::size_t n_tile_size,
                            const std::size_t n_tiles,
                            CUDA_GPU &gpu);

// Tiled Loss
hpx::shared_future<double> compute_loss_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_alpha,
    std::vector<hpx::shared_future<double *>> &ft_y,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    CUDA_GPU &gpu);

// Tiled Diagonal of Posterior Covariance Matrix
void symmetric_matrix_matrix_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    CUDA_GPU &gpu);

/**
 * @brief Compute the difference between two tiled vectors
 *
 * @param ft_priorK Tiled vector that is being subtracted from.
 * @param ft_inter Tiled vector that is being subtracted.
 * @param ft_vector Tiled vector that is the result of the subtraction.
 * @param m_tile_size Tile size dimension.
 * @param m_tiles Number of tiles.
 * @param gpu GPU target for computations.
 */
void vector_difference_tiled(std::vector<hpx::shared_future<double *>> &ft_priorK,
                             std::vector<hpx::shared_future<double *>> &ft_inter,
                             std::vector<hpx::shared_future<double *>> &ft_vector,
                             const std::size_t m_tile_size,
                             const std::size_t m_tiles,
                             CUDA_GPU &gpu);

// Tiled Prediction Uncertainty
void matrix_diagonal_tiled(std::vector<hpx::shared_future<double *>> &ft_priorK,
                           std::vector<hpx::shared_future<double *>> &ft_vector,
                           const std::size_t m_tile_size,
                           const std::size_t m_tiles,
                           CUDA_GPU &gpu);

// Compute I-y*y^T*inv(K)
void update_grad_K_tiled_mkl(std::vector<hpx::shared_future<double *>> &ft_tiles,
                             const std::vector<hpx::shared_future<double *>> &ft_v1,
                             const std::vector<hpx::shared_future<double *>> &ft_v2,
                             const std::size_t n_tile_size,
                             const std::size_t n_tiles,
                             CUDA_GPU &gpu);

/**
 * @brief Updates the lengthscale hyperparameter of the SEK kernel using Adam.
 *
 * @param ft_invK Tiled inverse of the covariance matrix K represented as a vector of futurized tiles.
 * @param ft_gradparam Tiled gradient of the hyperparameter represented as a vector of futurized tiles.
 * @param ft_alpha Tiled vector containing the precomputed inv(K) * y where y is the training output.
 * @param sek_params Hyperparameters of the SEK kernel
 * @param adam_params Hyperparameter of the Adam optimizer
 * @param n_tile_size Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 * @param m_T Tiled vector containing the first moment of the Adam optimizer.
 * @param v_T Tiled vector containing the second moment of the Adam optimizer.
 * @param beta1_T Tiled vector containing the first moment of the Adam optimizer.
 * @param beta2_T Tiled vector containing the second moment of the Adam optimizer.
 * @param iter Current iteration.
 * @param gpu GPU target for computations.
 *
 * @return The updated hyperparameter
 */
double update_lengthscale(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    SEKParams sek_params,
    AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    CUDA_GPU &gpu);

/**
 * @brief Updates the vertical lengthscale hyperparameter of the SEK kernel
 *        using Adam.
 *
 * @param ft_invK Tiled inverse of the covariance matrix K represented as a vector of futurized tiles.
 * @param ft_gradparam Tiled gradient of the hyperparameter represented as a vector of futurized tiles.
 * @param ft_alpha Tiled vector containing the precomputed inv(K) * y where y is the training output.
 * @param sek_params Hyperparameters of the SEK kernel
 * @param adam_params Hyperparameter of the Adam optimizer
 * @param n_tile_size Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 * @param m_T Tiled vector containing the first moment of the Adam optimizer.
 * @param v_T Tiled vector containing the second moment of the Adam optimizer.
 * @param beta1_T Tiled vector containing the first moment of the Adam optimizer.
 * @param beta2_T Tiled vector containing the second moment of the Adam optimizer.
 * @param iter Current iteration.
 * @param gpu GPU target for computations.
 *
 * @return The updated hyperparameter
 */
double update_vertical_lengthscale(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    SEKParams sek_params,
    AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    CUDA_GPU &gpu);

/**
 * @brief Updates a hyperparameter of the SEK kernel using Adam
 *
 * @param ft_invK Tiled inverse of the covariance matrix K represented as a vector of futurized tiles.
 * @param ft_alpha Tiled vector containing the precomputed inv(K) * y where y is the training output.
 * @param sek_params Hyperparameters of the SEK kernel
 * @param adam_params Hyperparameter of the Adam optimizer
 * @param n_tile_size Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 * @param m_T Tiled vector containing the first moment of the Adam optimizer.
 * @param v_T Tiled vector containing the second moment of the Adam optimizer.
 * @param beta1_T Tiled vector containing the first moment of the Adam optimizer.
 * @param beta2_T Tiled vector containing the second moment of the Adam optimizer.
 * @param iter Current iteration.
 * @param gpu GPU target for computations.
 *
  @return The updated hyperparameter
 */
double update_noise_variance(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    SEKParams sek_params,
    AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    CUDA_GPU &gpu);

}  // end of namespace gpu

GPRAT_NS_END

#endif
