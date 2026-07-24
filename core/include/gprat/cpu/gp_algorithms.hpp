#ifndef GPRAT_CPU_GP_ALGORITHMS_HPP
#define GPRAT_CPU_GP_ALGORITHMS_HPP

#pragma once

#include "gprat/detail/config.hpp"
#include "gprat/kernels.hpp"
#include "gprat/tile_data.hpp"

#include <span>
#include <vector>

GPRAT_NS_BEGIN

namespace cpu
{

/**
 * @brief Compute the squared exponential kernel of two feature vectors
 *
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 * @param i_input The first feature vector
 * @param j_input The second feature vector
 *
 * @return The entry of a covariance function
 */
double compute_covariance_function(std::size_t n_regressors,
                                   const SEKParams &sek_params,
                                   std::span<const double> i_input,
                                   std::span<const double> j_input);

/**
 * @brief Generate a tile of the covariance matrix
 *
 * @param input The input data vector
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 *
 * @return A quadratic tile of the covariance matrix of size N x N
 * @note Does apply noise variance on the diagonal
 */
mutable_tile_data<double> gen_tile_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    std::span<const double> input);

/**
 * @brief Generate a tile of the prior covariance matrix
 *
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N The dimension of the quadratic tile
 * @param n_regressors The number of regressors
 * @param hyperparameters The kernel hyperparameters
 * @param input The input data vector
 *
 * @return A quadratic tile of the prior covariance matrix of size N x N
 * @note Does NOT apply noise variance on the diagonal
 */
// NAME: gen_tile_priot_covariance
mutable_tile_data<double> gen_tile_full_prior_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    std::span<const double> input);

/**
 * @brief Generate the diagonal of a diagonal tile in the prior covariance matrix
 *
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N The dimension of the tile diagonal
 * @param n_regressors The number of regressors
 * @param hyperparameters The kernel hyperparameters
 * @param input The input data vector
 *
 * @return The diagonal of size N of a tile of the prior covariance matrix of size N x N
 * @note Does NOT apply noise variance
 */
// NAME: gen_tile_diag_prior_covariance
mutable_tile_data<double> gen_tile_prior_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    std::span<const double> input);

/**
 * @brief Generate a tile of the cross-covariance matrix
 *
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N_row The row-wise dimension of the tile
 * @param N_col The column-wise dimension of the tile
 * @param n_regressors The number of regressors
 * @param hyperparameters The kernel hyperparameters
 * @param input The input data vector
 *
 * @return A tile of the cross covariance matrix of size N_row x N_col
 * @note Does NOT apply noise variance
 */
mutable_tile_data<double> gen_tile_cross_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N_row,
    std::size_t N_col,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    std::span<const double> row_input,
    std::span<const double> col_input);

/**
 * @brief Transpose a tile of size N_row x N_col
 *
 * @param N_row The row-wise dimension of the tile
 * @param N_col The column-wise dimension of the tile
 * @param tile The tile to transpose
 *
 * @return The transposed tile of size N_col x N_row
 */
mutable_tile_data<double> gen_tile_transpose(std::size_t N_row, std::size_t N_col, std::span<const double> tile);

/**
 * @brief Generate a tile of the output data
 *
 * @param row The row index of the tile in relation to the tiled matrix
 * @param N The size of the tile
 * @param output The output data vector
 *
 * @return A tile of the output data of size N
 */
mutable_tile_data<double> gen_tile_output(std::size_t row, std::size_t N, std::span<const double> output);

/**
 * @brief Compute the L2-error norm over all tiles and elements
 *
 * @param n_tiles The number of tiles per dimension
 * @param tile_size The number of elements per tile
 * @param b The ground throuth
 * @param tiles The tiled matrix
 */
// decide to cut. If not use BLAS
double compute_error_norm(std::size_t n_tiles,
                          std::size_t tile_size,
                          const std::vector<double> &b,
                          const std::vector<std::vector<double>> &tiles);

/**
 * @brief Generate a tile initialized with zero
 *
 * @param N The size of the tile
 *
 * @return A tile filled with zeros of size N
 */
mutable_tile_data<double> gen_tile_zeros(std::size_t N);

/**
 * @brief Generate an identity tile (i==j?1:0)
 *
 * @param N The dimension of the quadratic tile
 * @return A NxN identity tile
 */
mutable_tile_data<double> gen_tile_identity(std::size_t N);

}  // end of namespace cpu

GPRAT_NS_END

#endif
