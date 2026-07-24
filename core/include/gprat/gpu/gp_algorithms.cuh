#ifndef GPRAT_GPU_GP_ALGORITHMS_HPP
#define GPRAT_GPU_GP_ALGORITHMS_HPP

#pragma once

#include "gprat/detail/config.hpp"
#include "gprat/kernels.hpp"
#include "gprat/target.hpp"
#include "gprat/tile_data.hpp"

#include <hpx/future.hpp>
#include <vector>

GPRAT_NS_BEGIN

namespace gpu
{

/**
 * @brief Generate a tile of the covariance matrix
 *
 * @param input The input data vector
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 * @param gpu GPU target for computations
 *
 * @return A quadratic tile of the covariance matrix of size N x N
 * @note Does apply noise variance on the diagonal
 */
double *gen_tile_covariance(const double *d_input,
                            const std::size_t tile_row,
                            const std::size_t tile_column,
                            const std::size_t n_tile_size,
                            const std::size_t n_regressors,
                            const SEKParams sek_params,
                            CUDA_GPU &gpu);

/**
 * @brief Generate the diagonal of a diagonal tile in the prior covariance matrix
 *
 * @param input The input data vector
 * @param tile_row The row index of the tile in the tiled matrix
 * @param tile_col The column index of the tile in the tiled matrix
 * @param N The dimension of the tile diagonal
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 * @param gpu GPU target for computations
 *
 * @return The diagonal of size N of a tile of the prior covariance matrix of size N x N
 * @note Does NOT apply noise variance
 */
double *gen_tile_prior_covariance(
    const double *d_input,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const SEKParams sek_params,
    CUDA_GPU &gpu);

/**
 * @brief Generate a tile of the cross-covariance matrix
 *
 * @param d_row_input input data for row, allocated on device
 * @param d_col_input input data for column, allocated on device
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param tile_row The row-wise dimension of the tile
 * @param tile_column The column-wise dimension of the tile
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 *
 * @return A tile of the cross covariance matrix of size N_row x N_col
 * @note Does NOT apply noise variance
 */
double *gen_tile_cross_covariance(
    const double *d_row_input,
    const double *d_col_input,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const std::size_t n_row_tile_size,
    const std::size_t n_column_tile_size,
    const std::size_t n_regressors,
    const SEKParams sek_params,
    CUDA_GPU &gpu);

/**
 * @brief Transpose a tile of size n_row_tile_size x n_column_tile_size
 *
 * @param n_row_tile_size The row-wise dimension of the tile
 * @param n_column_tile_size The column-wise dimension of the tile
 * @param tile The tile to transpose
 *
 * @return The transposed tile of size n_row_tile_size x n_column_tile_size
 */
hpx::shared_future<double *> gen_tile_transpose(std::size_t n_row_tile_size,
                                                std::size_t n_column_tile_size,
                                                const hpx::shared_future<double *> f_tile,
                                                CUDA_GPU &gpu);

/**
 * @brief Generate a tile of the output data
 *
 * @param row The row index of the tile in relation to the tiled matrix
 * @param n_tile_size The size of the tile
 * @param output The output data vector
 *
 * @return A tile of the output data of size n_tile_size
 */
double *
gen_tile_output(const std::size_t row, const std::size_t n_tile_size, const double *d_output, CUDA_GPU &gpu);

/**
 * @brief Compute the L2-error norm over all tiles and elements
 *
 * @param n_tiles The number of tiles per dimension
 * @param n_tile_size The number of elements per tile
 * @param b The ground throuth
 * @param tiles The tiled matrix
 */
double compute_error_norm(const std::size_t n_tiles,
                          const std::size_t n_tile_size,
                          const std::vector<double> &b,
                          const std::vector<std::vector<double>> &tiles);

/**
 * @brief Generate a tile initialized with zero
 *
 * @param n_tile_size The size of the tile
 *
 * @return A tile filled with zeros of size N
 */
double *gen_tile_zeros(std::size_t n_tile_size, CUDA_GPU &gpu);

/**
 * @brief Allocates the tiled covariance matrix on the device given the training
 *        data.
 *
 * @param d_training_input The training input data
 * @param n_tiles The number of tiles per dimension
 * @param n_tile_size The size of the tile
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 * @param gpu GPU target for computations
 */
std::vector<hpx::shared_future<double *>> assemble_tiled_covariance_matrix(
    const double *d_training_input,
    const std::size_t n_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const SEKParams sek_params,
    CUDA_GPU &gpu);

/**
 * @brief Allocates the tiled alpha vector on the device given the training
 *        output.
 *
 * @param d_output The training output data
 * @param n_tiles The number of tiles per dimension
 * @param n_tile_size The size of the tile
 * @param gpu GPU target for computations
 *
 * @return A tiled alpha vector of size n_tiles x n_tile_size
 */
std::vector<hpx::shared_future<double *>> assemble_alpha_tiles(
    const double *d_output, const std::size_t n_tiles, const std::size_t n_tile_size, CUDA_GPU &gpu);

/**
 * @brief Allocates the tiled cross covariance matrix on the device given the
 *        training and test data.
 *
 * @param d_test_input The test input data
 * @param d_training_input The training input data
 * @param m_tiles The number of tiles per dimension for the test data
 * @param n_tiles The number of tiles per dimension for the training data
 * @param m_tile_size The size of the tile for the test data
 * @param n_tile_size The size of the tile for the training data
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 * @param gpu GPU target for computations
 *
 * @return A tiled cross covariance matrix with m_tiles x n_tiles tiles
 */
std::vector<hpx::shared_future<double *>> assemble_cross_covariance_tiles(
    const double *d_test_input,
    const double *d_training_input,
    const std::size_t m_tiles,
    const std::size_t n_tiles,
    const std::size_t m_tile_size,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const SEKParams sek_params,
    CUDA_GPU &gpu);

/**
 * @brief Allocates a tiled vector on the device and initializes it with zeros.
 *
 * @param n_tile_size The size of the tile
 * @param n_tiles The number of tiles per dimensionl
 * @param gpu GPU target for computations
 *
 * @return A tiled vector of size n_tiles x n_tile_size with zeros
 */
std::vector<hpx::shared_future<double *>>
assemble_tiles_with_zeros(std::size_t n_tile_size, std::size_t n_tiles, CUDA_GPU &gpu);

/**
 * @brief Allocates the tiled prior covariance matrix on the device given the
 *        test input data.
 *
 * @param d_test_input The test input data
 * @param m_tiles The number of tiles per dimension for the test data
 * @param m_tile_size The size of the tile for the test data
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 * @param gpu GPU target for computations
 *
 * @return A tiled prior covariance matrix with m_tiles x m_tiles tiles
 */
std::vector<hpx::shared_future<double *>> assemble_prior_K_tiles(
    const double *d_test_input,
    const std::size_t m_tiles,
    const std::size_t m_tile_size,
    const std::size_t n_regressors,
    const SEKParams sek_params,
    CUDA_GPU &gpu);

/**
 * @brief Allocates the posterior covariance matrix.
 *
 * @param d_test_input The test input data
 * @param m_tiles The number of tiles per dimension for the test data
 * @param m_tile_size The size of the tile for the test data
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 * @param gpu GPU target for computations
 *
 * @return A tiled posterior covariance matrix with m_tiles x m_tiles tiles
 */
std::vector<hpx::shared_future<double *>> assemble_prior_K_tiles_full(
    const double *d_test_input,
    const std::size_t m_tiles,
    const std::size_t m_tile_size,
    const std::size_t n_regressors,
    const SEKParams sek_params,
    CUDA_GPU &gpu);

/**
 * @brief Allocates the tiled transpose cross covariance matrix on the device
 *        given the cross covariance matrix.
 *
 * Allocates device memory and copies the tranposed cross covariance matrix into
 * it.
 *
 * @param d_cross_covariance_tiles The cross covariance matrix
 * @param n_tiles The number of tiles per dimension for the training data
 * @param m_tiles The number of tiles per dimension for the test data
 * @param n_tile_size The size of the tile for the training data
 * @param m_tile_size The size of the tile for the test data
 * @param gpu GPU target for computations
 */
std::vector<hpx::shared_future<double *>> assemble_t_cross_covariance_tiles(
    const std::vector<hpx::shared_future<double *>> &d_cross_covariance_tiles,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    CUDA_GPU &gpu);

/**
 * @brief Allocates the output vector on the device given the training output
 *
 * @param d_training_input The training input data
 * @param n_tiles The number of tiles per dimension
 * @param n_tile_size The size of the tile
 * @param gpu GPU target for computations
 */
std::vector<hpx::shared_future<double *>> assemble_y_tiles(
    const double *d_training_output, const std::size_t n_tiles, const std::size_t n_tile_size, CUDA_GPU &gpu);

/**
 * @brief Allocates the tiled covariance matrix on the device given the training
 *        data.
 *
 * @param d_training_input The training input data
 * @param n_tile_size The size of the tile
 * @param n_tiles The number of tiles per dimension
 * @param gpu GPU target for computations
 */
std::vector<double> copy_tiled_vector_to_host_vector(std::vector<hpx::shared_future<double *>> &d_tiles,
                                                     std::size_t n_tile_size,
                                                     std::size_t n_tiles,
                                                     CUDA_GPU &gpu);

/**
 * @brief Moves lower triangular tiles of the covariance matrix to the host.
 *
 * Allocates host memory for the tiles on the host and free the device memory.
 *
 * @param d_tiles The tiles on the device
 * @param n_tile_size The size of the tile
 * @param n_tiles The number of tiles
 * @param gpu GPU target for computations
 */
std::vector<mutable_tile_data<double>> move_lower_tiled_matrix_to_host(
    const std::vector<hpx::shared_future<double *>> &d_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    CUDA_GPU &gpu);

/**
 * @brief Frees the device memory of the lower triangular tiles of the covariance matrix.
 *
 * @param d_tiles The tiles on the device
 * @param n_tiles The number of tiles
 */
void free_lower_tiled_matrix(const std::vector<hpx::shared_future<double *>> &d_tiles, const std::size_t n_tiles);

/**
 * @brief Frees the device memory of all tiles of a fully populated tiled matrix.
 *
 * Unlike free_lower_tiled_matrix, this frees every tile (both triangles), for
 * matrices such as the ones produced by assemble_prior_K_tiles_full where the
 * upper triangle holds distinct device allocations rather than aliases of the
 * lower triangle.
 *
 * @param d_tiles The tiles on the device
 * @param n_tiles The number of tiles
 */
void free_full_tiled_matrix(const std::vector<hpx::shared_future<double *>> &d_tiles, const std::size_t n_tiles);

}  // end of namespace gpu

GPRAT_NS_END

#endif
