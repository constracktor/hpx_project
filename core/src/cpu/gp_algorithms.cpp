#include "gprat/cpu/gp_algorithms.hpp"

#include "gprat/performance_counters.hpp"
#include "gprat/tile_data.hpp"

#include <cmath>

GPRAT_NS_BEGIN

namespace cpu
{

// Tile generation

double compute_covariance_function(std::size_t n_regressors,
                                   const SEKParams &sek_params,
                                   std::span<const double> i_input,
                                   std::span<const double> j_input)
{
    GPRAT_TIME_FUNCTION(&compute_covariance_function);
    // k(z_i,z_j) = vertical_lengthscale * exp(-0.5 / lengthscale^2 * (z_i - z_j)^2)
    double distance = 0.0;
    for (std::size_t k = 0; k < n_regressors; k++)
    {
        const double z_ik_minus_z_jk = i_input[k] - j_input[k];
        distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
    }

    return sek_params.vertical_lengthscale * exp(-0.5 / (sek_params.lengthscale * sek_params.lengthscale) * distance);
}

mutable_tile_data<double> gen_tile_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    std::span<const double> input)
{
    GPRAT_TIME_FUNCTION(&gen_tile_covariance);
    mutable_tile_data<double> tile(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        const std::size_t i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            const std::size_t j_global = N * col + j;

            // compute covariance function
            auto covariance_function = compute_covariance_function(
                n_regressors, sek_params, input.subspan(i_global, n_regressors), input.subspan(j_global, n_regressors));
            if (i_global == j_global)
            {
                // noise variance on diagonal
                covariance_function += sek_params.noise_variance;
            }

            tile.data()[i * N + j] = covariance_function;
        }
    }
    return tile;
}

mutable_tile_data<double> gen_tile_full_prior_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    std::span<const double> input)
{
    GPRAT_TIME_FUNCTION(&gen_tile_full_prior_covariance);
    mutable_tile_data<double> tile(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        const std::size_t i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            const std::size_t j_global = N * col + j;
            // compute covariance function
            tile.data()[i * N + j] = compute_covariance_function(
                n_regressors, sek_params, input.subspan(i_global, n_regressors), input.subspan(j_global, n_regressors));
        }
    }
    return tile;
}

mutable_tile_data<double> gen_tile_prior_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    std::span<const double> input)
{
    GPRAT_TIME_FUNCTION(&gen_tile_prior_covariance);
    mutable_tile_data<double> tile(N);
    for (std::size_t i = 0; i < N; i++)
    {
        const std::size_t i_global = N * row + i;
        const std::size_t j_global = N * col + i;
        // compute covariance function
        tile.data()[i] = compute_covariance_function(
            n_regressors, sek_params, input.subspan(i_global, n_regressors), input.subspan(j_global, n_regressors));
    }
    return tile;
}

mutable_tile_data<double> gen_tile_cross_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N_row,
    std::size_t N_col,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    std::span<const double> row_input,
    std::span<const double> col_input)
{
    GPRAT_TIME_FUNCTION(&gen_tile_cross_covariance);
    mutable_tile_data<double> tile(N_row * N_col);
    for (std::size_t i = 0; i < N_row; i++)
    {
        std::size_t i_global = N_row * row + i;
        for (std::size_t j = 0; j < N_col; j++)
        {
            std::size_t j_global = N_col * col + j;
            // compute covariance function
            tile.data()[i * N_col + j] = compute_covariance_function(
                n_regressors,
                sek_params,
                row_input.subspan(i_global, n_regressors),
                col_input.subspan(j_global, n_regressors));
        }
    }
    return tile;
}

mutable_tile_data<double> gen_tile_transpose(std::size_t N_row, std::size_t N_col, std::span<const double> tile)
{
    GPRAT_TIME_FUNCTION(&gen_tile_transpose);
    mutable_tile_data<double> transposed(N_row * N_col);
    // Transpose entries
    for (std::size_t j = 0; j < N_col; j++)
    {
        for (std::size_t i = 0; i < N_row; ++i)
        {
            // Mapping (i, j) in the original tile to (j, i) in the transposed tile
            transposed.data()[j * N_row + i] = tile[i * N_col + j];
        }
    }
    return transposed;
}

mutable_tile_data<double> gen_tile_output(std::size_t row, std::size_t N, std::span<const double> output)
{
    GPRAT_TIME_FUNCTION(&gen_tile_output);
    mutable_tile_data<double> tile(N);
    std::copy(output.data() + (N * row), output.data() + (N * (row + 1)), tile.data());
    return tile;
}

mutable_tile_data<double> gen_tile_zeros(std::size_t N)
{
    GPRAT_TIME_FUNCTION(&gen_tile_zeros);
    mutable_tile_data<double> tile(N);
    std::fill_n(tile.data(), N, 0.0);
    return tile;
}

mutable_tile_data<double> gen_tile_identity(std::size_t N)
{
    GPRAT_TIME_FUNCTION(&gen_tile_identity);
    mutable_tile_data<double> tile(N * N);
    // Initialize zero tile
    std::fill_n(tile.data(), N * N, 0.0);
    // Fill diagonal with ones
    for (std::size_t i = 0; i < N; i++)
    {
        tile.data()[i * N + i] = 1.0;
    }
    return tile;
}

// Error

double compute_error_norm(std::size_t n_tiles,
                          std::size_t tile_size,
                          const std::vector<double> &b,
                          const std::vector<std::vector<double>> &tiles)
{
    GPRAT_TIME_FUNCTION(&compute_error_norm);
    double error = 0.0;
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        const std::vector<double> &a = tiles[k];
        for (std::size_t i = 0; i < tile_size; i++)
        {
            std::size_t i_global = tile_size * k + i;
            // ||a - b||_2
            error += (b[i_global] - a[i]) * (b[i_global] - a[i]);
        }
    }
    return sqrt(error);
}

}  // end of namespace cpu

GPRAT_NS_END
