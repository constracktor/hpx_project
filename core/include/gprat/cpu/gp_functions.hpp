#ifndef GPRAT_CPU_GP_FUNCTIONS_HPP
#define GPRAT_CPU_GP_FUNCTIONS_HPP

#pragma once

#include "gprat/cpu/gp_algorithms.hpp"
#include "gprat/cpu/tiled_algorithms.hpp"
#include "gprat/detail/config.hpp"
#include "gprat/hyperparameters.hpp"
#include "gprat/kernels.hpp"
#include "gprat/scheduler.hpp"
#include "gprat/tile_data.hpp"

#include <vector>

GPRAT_NS_BEGIN

namespace cpu
{

/**
 * @brief Perform Cholesky decomposition (+Assembly)
 *
 * @param training_input The training input data
 * @param sek_params The kernel hyperparameters
 *
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @return The tiled Cholesky factor
 */
template <typename Scheduler = tiled_scheduler_local>
std::vector<mutable_tile_data<double>>
cholesky(Scheduler &sched,
         const std::vector<double> &training_input,
         const SEKParams &sek_params,
         std::size_t n_tiles,
         std::size_t n_tile_size,
         std::size_t n_regressors)
{
    // Tiled covariance matrix K_NxN
    auto K_tiles = make_tiled_dataset<double>(
        sched,
        n_tiles * n_tiles,
        [&](std::size_t tile_index)
        { return covariance_tile_on(sched, n_tiles, tile_index / n_tiles, tile_index % n_tiles); });

    for (std::size_t row = 0; row < n_tiles; row++)
    {
        for (std::size_t col = 0; col <= row; col++)
        {
            K_tiles[row * n_tiles + col] = detail::named_make_tile<gen_tile_covariance>(
                sched,
                covariance_tile_on(sched, n_tiles, row, col),
                "assemble_tiled_K",
                K_tiles[row * n_tiles + col],
                row,
                col,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }

    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(sched, K_tiles, n_tile_size, n_tiles);

    ///////////////////////////////////////////////////////////////////////////
    // Synchronize
    std::vector<mutable_tile_data<double>> result(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            result[i * n_tiles + j] = K_tiles[i * n_tiles + j].get();
        }
    }
    return result;
}

/**
 * @brief Compute the predictions without uncertainties.
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param test_input The test input data
 * @param hyperparameters The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param m_tiles The number of test tiles
 * @param m_tile_size The size of each test tile
 * @param n_regressors The number of regressors
 *
 * @return A vector containing the predictions
 */
template <typename Scheduler = tiled_scheduler_local>
std::vector<double>
predict(Scheduler &sched,
        const std::vector<double> &training_input,
        const std::vector<double> &training_output,
        const std::vector<double> &test_input,
        const SEKParams &sek_params,
        std::size_t n_tiles,
        std::size_t n_tile_size,
        std::size_t m_tiles,
        std::size_t m_tile_size,
        std::size_t n_regressors)
{
    /*
     * Prediction: hat(y)_M = cross(K)_MxN * K^-1_NxN * y_N
     * - Covariance matrix K_NxN
     * - Cross-covariance cross(K)_MxN
     * - Training output y_N
     * - Prediction output hat(y)_M
     *
     * Algorithm:
     * 1: Compute lower triangular part of covariance matrix K
     * 2: Compute Cholesky factor L of K
     * 3: Compute prediction hat(y):
     *    - triangular solve L * beta = y
     *    - triangular solve L^T * alpha = beta
     *    - compute hat(y) = cross(K) * alpha
     */

    ///////////////////////////////////////////////////////////////////////////
    // Cholesky

    // Tiled covariance matrix K_NxN
    auto K_tiles = make_tiled_dataset<double>(
        sched,
        n_tiles * n_tiles,
        [&](std::size_t tile_index)
        { return covariance_tile_on(sched, n_tiles, tile_index / n_tiles, tile_index % n_tiles); });

    for (std::size_t row = 0; row < n_tiles; row++)
    {
        for (std::size_t col = 0; col <= row; col++)
        {
            K_tiles[row * n_tiles + col] = detail::named_make_tile<gen_tile_covariance>(
                sched,
                covariance_tile_on(sched, n_tiles, row, col),
                "assemble_tiled_K",
                K_tiles[row * n_tiles + col],
                row,
                col,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }

    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(sched, K_tiles, n_tile_size, n_tiles);

    ///////////////////////////////////////////////////////////////////////////
    // Prediction

    // Tiled cross_covariance matrix K_NxM
    auto cross_covariance_tiles = make_tiled_dataset<double>(
        sched,
        m_tiles * n_tiles,
        [&](std::size_t tile_index)
        { return covariance_tile_on(sched, n_tiles, tile_index / n_tiles, tile_index % n_tiles); });
    // Tiled solution
    auto prediction_tiles = make_tiled_dataset<double>(
        sched, m_tiles, [&](std::size_t tile_index) { return prediction_tile_on(sched, m_tiles, tile_index); });
    // Tiled intermediate solution
    auto alpha_tiles = make_tiled_dataset<double>(
        sched, n_tiles, [&](std::size_t tile_index) { return alpha_tile_on(sched, n_tiles, tile_index); });

    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = detail::named_make_tile<gen_tile_output>(
            sched,
            alpha_tile_on(sched, n_tiles, i),
            "assemble_tiled_alpha",
            alpha_tiles[i],
            i,
            n_tile_size,
            training_output);
    }

    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            cross_covariance_tiles[i * n_tiles + j] = detail::named_make_tile<gen_tile_cross_covariance>(
                sched,
                cross_covariance_tile_on(sched, n_tiles, i, j),
                "assemble_pred",
                cross_covariance_tiles[i * n_tiles + j],
                i,
                j,
                m_tile_size,
                n_tile_size,
                n_regressors,
                sek_params,
                test_input,
                training_input);
        }
    }

    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prediction_tiles[i] = detail::named_make_tile<gen_tile_zeros>(
            sched, prediction_tile_on(sched, m_tiles, i), "assemble_tiled", prediction_tiles[i], m_tile_size);
    }

    // Launch asynchronous triangular solve  L * (L^T * alpha) = y
    forward_solve_tiled(sched, K_tiles, alpha_tiles, n_tile_size, n_tiles);
    backward_solve_tiled(sched, K_tiles, alpha_tiles, n_tile_size, n_tiles);

    // Launch asynchronous prediction computation solve: \hat{y} = K_cross_cov * alpha
    matrix_vector_tiled(
        sched, cross_covariance_tiles, alpha_tiles, prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles);

    ///////////////////////////////////////////////////////////////////////////
    // Synchronize prediction
    // Preallocate memory
    std::vector<double> prediction_result;
    prediction_result.reserve(test_input.size());
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        mutable_tile_data<double> tile = prediction_tiles[i].get();
        std::copy_n(tile.data(), tile.size(), std::back_inserter(prediction_result));
    }
    return prediction_result;
}

/**
 * @brief Compute the predictions with uncertainties.
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param test_input The test input data
 * @param hyperparameters The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param m_tiles The number of test tiles
 * @param m_tile_size The size of each test tile
 * @param n_regressors The number of regressors
 *
 * @return A vector containing the prediction vector and the uncertainty vector
 */
template <typename Scheduler = tiled_scheduler_local>
std::vector<std::vector<double>> predict_with_uncertainty(
    Scheduler &sched,
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    const std::vector<double> &test_input,
    const SEKParams &sek_params,
    std::size_t n_tiles,
    std::size_t n_tile_size,
    std::size_t m_tiles,
    std::size_t m_tile_size,
    std::size_t n_regressors)
{
    /*
     * Prediction: hat(y) = cross(K) * K^-1 * y
     * Uncertainty: diag(Sigma) = diag(prior(K)) * diag(cross(K)^T * K^-1 * cross(K))
     * - Covariance matrix K_NxN
     * - Cross-covariance cross(K)_MxN
     * - Prior covariance prior(K)_MxM
     * - Training output y_N
     * - Prediction output hat(y)_M
     * - Posterior covariance matrix Sigma_MxM
     *
     * Algorithm:
     * 1: Compute lower triangular part of covariance matrix K
     * 2: Compute Cholesky factor L of K
     * 3: Compute prediction hat(y):
     *    - triangular solve L * beta = y
     *    - triangular solve L^T * alpha = beta
     *    - compute hat(y) = cross(K) * alpha
     * 4: Compute uncertainty diag(Sigma):
     *    - triangular solve L * V = cross(K)^T
     *    - compute diag(W) = diag(V^T * V)
     *    - compute diag(Sigma) = diag(prior(K)) - diag(W)
     */

    ///////////////////////////////////////////////////////////////////////////
    // Cholesky

    // Tiled covariance matrix K_NxN
    auto K_tiles = make_tiled_dataset<double>(
        sched,
        n_tiles * n_tiles,
        [&](std::size_t tile_index)
        { return covariance_tile_on(sched, n_tiles, tile_index / n_tiles, tile_index % n_tiles); });

    for (std::size_t row = 0; row < n_tiles; row++)
    {
        for (std::size_t col = 0; col <= row; col++)
        {
            K_tiles[row * n_tiles + col] = detail::named_make_tile<gen_tile_covariance>(
                sched,
                covariance_tile_on(sched, n_tiles, row, col),
                "assemble_tiled_K",
                K_tiles[row * n_tiles + col],
                row,
                col,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }

    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(sched, K_tiles, n_tile_size, n_tiles);

    ///////////////////////////////////////////////////////////////////////////
    // Prediction

    // Tiled intermediate solution
    auto alpha_tiles = make_tiled_dataset<double>(
        sched, n_tiles, [&](std::size_t tile_index) { return alpha_tile_on(sched, n_tiles, tile_index); });
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = detail::named_make_tile<gen_tile_output>(
            sched,
            alpha_tile_on(sched, n_tiles, i),
            "assemble_tiled_alpha",
            alpha_tiles[i],
            i,
            n_tile_size,
            training_output);
    }

    // Tiled cross_covariance matrix K_NxM
    auto cross_covariance_tiles = make_tiled_dataset<double>(
        sched,
        m_tiles * n_tiles,
        [&](std::size_t tile_index)
        { return covariance_tile_on(sched, n_tiles, tile_index / n_tiles, tile_index % n_tiles); });
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            cross_covariance_tiles[i * n_tiles + j] = detail::named_make_tile<gen_tile_cross_covariance>(
                sched,
                cross_covariance_tile_on(sched, n_tiles, i, j),
                "assemble_pred",
                cross_covariance_tiles[i * n_tiles + j],
                i,
                j,
                m_tile_size,
                n_tile_size,
                n_regressors,
                sek_params,
                test_input,
                training_input);
        }
    }

    // Tiled solution
    auto prediction_tiles = make_tiled_dataset<double>(
        sched, m_tiles, [&](std::size_t tile_index) { return prediction_tile_on(sched, m_tiles, tile_index); });
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prediction_tiles[i] = detail::named_make_tile<gen_tile_zeros>(
            sched, prediction_tile_on(sched, m_tiles, i), "assemble_tiled", prediction_tiles[i], m_tile_size);
    }

    // Launch asynchronous triangular solve  L * (L^T * alpha) = y
    forward_solve_tiled(sched, K_tiles, alpha_tiles, n_tile_size, n_tiles);
    backward_solve_tiled(sched, K_tiles, alpha_tiles, n_tile_size, n_tiles);

    // Launch asynchronous prediction computation solve: \hat{y} = K_cross_cov * alpha
    matrix_vector_tiled(
        sched, cross_covariance_tiles, alpha_tiles, prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles);

    ///////////////////////////////////////////////////////////////////////////
    // Uncertainty

    // Tiled transposed cross_covariance matrix K_MxN
    auto t_cross_covariance_tiles = make_tiled_dataset<double>(
        sched,
        n_tiles * m_tiles,
        [&](std::size_t tile_index)
        { return t_cross_covariance_tile_on(sched, m_tiles, tile_index / m_tiles, tile_index % m_tiles); });
    for (std::size_t j = 0; j < n_tiles; j++)
    {
        for (std::size_t i = 0; i < m_tiles; i++)
        {
            t_cross_covariance_tiles[j * m_tiles + i] = detail::named_make_tile<gen_tile_transpose>(
                sched,
                t_cross_covariance_tile_on(sched, m_tiles, j, i),
                "assemble_pred",
                t_cross_covariance_tiles[j * m_tiles + i],
                m_tile_size,
                n_tile_size,
                cross_covariance_tiles[i * n_tiles + j]);
        }
    }

    // Tiled prior covariance matrix diagonal diag(K_MxM)
    auto prior_K_tiles = make_tiled_dataset<double>(
        sched, m_tiles, [&](std::size_t tile_index) { return prior_K_tile_on(sched, n_tiles, 0, tile_index); });
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prior_K_tiles[i] = detail::named_make_tile<gen_tile_prior_covariance>(
            sched,
            prior_K_tile_on(sched, m_tiles, 0, i),
            "assemble_tiled",
            prior_K_tiles[i],
            i,
            i,
            m_tile_size,
            n_regressors,
            sek_params,
            test_input);
    }

    // Tiled uncertainty solution
    auto uncertainty_tiles = make_tiled_dataset<double>(
        sched, m_tiles, [&](std::size_t tile_index) { return uncertainty_tile_on(sched, m_tiles, tile_index); });
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        uncertainty_tiles[i] = detail::named_make_tile<gen_tile_zeros>(
            sched, uncertainty_tile_on(sched, m_tiles, i), "assemble_prior_inter", uncertainty_tiles[i], m_tile_size);
    }

    // Launch asynchronous triangular solve L * V = cross(K)^T
    forward_solve_tiled_matrix(sched, K_tiles, t_cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);

    // Launch asynchronous computation diag(W) = diag(V^T * V)
    symmetric_matrix_matrix_diagonal_tiled(
        sched, t_cross_covariance_tiles, uncertainty_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);

    // Launch asynchronous computation diag(Sigma) = diag(prior(K)) - diag(W)
    vector_difference_tiled(sched, prior_K_tiles, uncertainty_tiles, m_tile_size, m_tiles);

    ///////////////////////////////////////////////////////////////////////////
    // Preallocate memory
    std::vector<double> prediction_result;
    std::vector<double> uncertainty_result;
    prediction_result.reserve(test_input.size());
    uncertainty_result.reserve(test_input.size());

    // Synchronize prediction
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        mutable_tile_data<double> tile = prediction_tiles[i].get();
        std::copy_n(tile.begin(), tile.size(), std::back_inserter(prediction_result));
    }

    // Synchronize uncertainty
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        mutable_tile_data<double> tile = uncertainty_tiles[i].get();
        std::copy_n(tile.begin(), tile.size(), std::back_inserter(uncertainty_result));
    }

    return std::vector<std::vector<double>>{ std::move(prediction_result), std::move(uncertainty_result) };
}

/**
 * @brief Compute the predictions with full covariance matrix.
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param test_input The test input data
 * @param sek_params The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param m_tiles The number of test tiles
 * @param m_tile_size The size of each test tile
 * @param n_regressors The number of regressors
 *
 * @return A vector containing the prediction vector and the full posterior covariance matrix
 */
template <typename Scheduler = tiled_scheduler_local>
std::vector<std::vector<double>> predict_with_full_cov(
    Scheduler &sched,
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    const std::vector<double> &test_input,
    const SEKParams &sek_params,
    std::size_t n_tiles,
    std::size_t n_tile_size,
    std::size_t m_tiles,
    std::size_t m_tile_size,
    std::size_t n_regressors)
{
    /*
     * Prediction: hat(y)_M = cross(K) * K^-1 * y
     * Full covariance: Sigma = prior(K) - cross(K)^T * K^-1 * cross(K)
     * - Covariance matrix K_NxN
     * - Cross-covariance cross(K)_MxN
     * - Prior covariance prior(K)_MxM
     * - Training output y_N
     * - Prediction output hat(y)_M
     * - Posterior covariance matrix Sigma_MxM
     *
     * Algorithm:
     * 1: Compute lower triangular part of covariance matrix K
     * 2: Compute Cholesky factor L of K
     * 3: Compute prediction hat(y):
     *    - triangular solve L * beta = y
     *    - triangular solve L^T * alpha = beta
     *    - compute hat(y) = cross(K) * alpha
     * 4: Compute full covariance matrix Sigma:
     *    - triangular solve L * V = cross(K)^T
     *    - compute W = V^T * V
     *    - compute Sigma = prior(K) - W
     * 5: Compute diag(Sigma)
     */

    ///////////////////////////////////////////////////////////////////////////
    // Cholesky

    // Tiled covariance matrix K_NxN
    auto K_tiles = make_tiled_dataset<double>(
        sched,
        n_tiles * n_tiles,
        [&](std::size_t tile_index)
        { return covariance_tile_on(sched, n_tiles, tile_index / n_tiles, tile_index % n_tiles); });
    for (std::size_t row = 0; row < n_tiles; row++)
    {
        for (std::size_t col = 0; col <= row; col++)
        {
            K_tiles[row * n_tiles + col] = detail::named_make_tile<gen_tile_covariance>(
                sched,
                covariance_tile_on(sched, n_tiles, row, col),
                "assemble_tiled_K",
                K_tiles[row * n_tiles + col],
                row,
                col,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }

    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(sched, K_tiles, n_tile_size, n_tiles);

    ///////////////////////////////////////////////////////////////////////////
    // Prediction

    // Tiled intermediate solution
    auto alpha_tiles = make_tiled_dataset<double>(
        sched, n_tiles, [&](std::size_t tile_index) { return alpha_tile_on(sched, n_tiles, tile_index); });
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = detail::named_make_tile<gen_tile_output>(
            sched,
            alpha_tile_on(sched, n_tiles, i),
            "assemble_tiled_alpha",
            alpha_tiles[i],
            i,
            n_tile_size,
            training_output);
    }

    // Tiled cross_covariance matrix K_NxM
    auto cross_covariance_tiles = make_tiled_dataset<double>(
        sched,
        m_tiles * n_tiles,
        [&](std::size_t tile_index)
        { return covariance_tile_on(sched, n_tiles, tile_index / n_tiles, tile_index % n_tiles); });
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            cross_covariance_tiles[i * n_tiles + j] = detail::named_make_tile<gen_tile_cross_covariance>(
                sched,
                cross_covariance_tile_on(sched, n_tiles, i, j),
                "assemble_pred",
                cross_covariance_tiles[i * n_tiles + j],
                i,
                j,
                m_tile_size,
                n_tile_size,
                n_regressors,
                sek_params,
                test_input,
                training_input);
        }
    }

    // Tiled solution
    auto prediction_tiles = make_tiled_dataset<double>(
        sched, m_tiles, [&](std::size_t tile_index) { return prediction_tile_on(sched, n_tiles, tile_index); });
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        prediction_tiles[i] = detail::named_make_tile<gen_tile_zeros>(
            sched, prediction_tile_on(sched, m_tiles, i), "assemble_tiled", prediction_tiles[i], m_tile_size);
    }

    // Launch asynchronous triangular solve  L * (L^T * alpha) = y
    forward_solve_tiled(sched, K_tiles, alpha_tiles, n_tile_size, n_tiles);
    backward_solve_tiled(sched, K_tiles, alpha_tiles, n_tile_size, n_tiles);

    // Launch asynchronous prediction computation solve: \hat{y} = K_cross_cov * alpha
    matrix_vector_tiled(
        sched, cross_covariance_tiles, alpha_tiles, prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles);

    ///////////////////////////////////////////////////////////////////////////
    // Uncertainty

    // Tiled transposed cross_covariance matrix K_MxN
    auto t_cross_covariance_tiles = make_tiled_dataset<double>(
        sched,
        n_tiles * m_tiles,
        [&](std::size_t tile_index)
        { return t_cross_covariance_tile_on(sched, m_tiles, tile_index / m_tiles, tile_index % m_tiles); });
    for (std::size_t j = 0; j < n_tiles; j++)
    {
        for (std::size_t i = 0; i < m_tiles; i++)
        {
            t_cross_covariance_tiles[j * m_tiles + i] = detail::named_make_tile<gen_tile_transpose>(
                sched,
                t_cross_covariance_tile_on(sched, m_tiles, j, i),
                "assemble_pred",
                t_cross_covariance_tiles[j * m_tiles + i],
                m_tile_size,
                n_tile_size,
                cross_covariance_tiles[i * n_tiles + j]);
        }
    }

    // Tiled prior covariance matrix K_MxM
    auto prior_K_tiles = make_tiled_dataset<double>(
        sched,
        m_tiles * m_tiles,
        [&](std::size_t tile_index)
        { return prior_K_tile_on(sched, n_tiles, tile_index / m_tiles, tile_index % m_tiles); });
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            prior_K_tiles[i * m_tiles + j] = detail::named_make_tile<gen_tile_full_prior_covariance>(
                sched,
                prior_K_tile_on(sched, m_tiles, i, j),
                "assemble_prior_tiled",
                prior_K_tiles[i * m_tiles + j],
                i,
                j,
                m_tile_size,
                n_regressors,
                sek_params,
                test_input);

            if (i != j)
            {
                prior_K_tiles[j * m_tiles + i] = detail::named_make_tile<gen_tile_transpose>(
                    sched,
                    prior_K_tile_on(sched, m_tiles, j, i),
                    "assemble_prior_tiled",
                    prior_K_tiles[j * m_tiles + i],
                    m_tile_size,
                    m_tile_size,
                    prior_K_tiles[i * m_tiles + j]);
            }
        }
    }

    // Tiled uncertainty solution
    auto uncertainty_tiles = make_tiled_dataset<double>(
        sched, m_tiles, [&](std::size_t tile_index) { return uncertainty_tile_on(sched, m_tiles, tile_index); });
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        uncertainty_tiles[i] = detail::named_make_tile<gen_tile_zeros>(
            sched, uncertainty_tile_on(sched, m_tiles, i), "assemble_prior_inter", uncertainty_tiles[i], m_tile_size);
    }

    // Launch asynchronous triangular solve L * V = cross(K)^T
    forward_solve_tiled_matrix(sched, K_tiles, t_cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous computation of full covariance Sigma = prior(K) - V^T * V
    symmetric_matrix_matrix_tiled(
        sched, t_cross_covariance_tiles, prior_K_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles);
    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous computation of uncertainty diag(Sigma)
    matrix_diagonal_tiled(sched, prior_K_tiles, uncertainty_tiles, m_tile_size, m_tiles);

    ///////////////////////////////////////////////////////////////////////////
    // Preallocate memory
    std::vector<double> prediction_result;
    std::vector<double> uncertainty_result;
    prediction_result.reserve(test_input.size());
    uncertainty_result.reserve(test_input.size());

    // Synchronize prediction
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        mutable_tile_data<double> tile = prediction_tiles[i].get();
        std::copy_n(tile.begin(), tile.size(), std::back_inserter(prediction_result));
    }

    // Synchronize uncertainty
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        mutable_tile_data<double> tile = uncertainty_tiles[i].get();
        std::copy_n(tile.begin(), tile.size(), std::back_inserter(uncertainty_result));
    }

    return std::vector<std::vector<double>>{ std::move(prediction_result), std::move(uncertainty_result) };
}

///////////////////////////////////////////////////////////////////////////
// OPTIMIZATION

/**
 * @brief Compute loss for given data and Gaussian process model
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param sek_params The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @return The loss
 */
template <typename Scheduler = tiled_scheduler_local>
double calculate_loss(Scheduler &sched,
                      const std::vector<double> &training_input,
                      const std::vector<double> &training_output,
                      const SEKParams &sek_params,
                      std::size_t n_tiles,
                      std::size_t n_tile_size,
                      std::size_t n_regressors)
{
    /*
     * Negative log likelihood loss:
     * loss(theta) = 0.5 * ( log(det(K)) - y^T * K^-1 * y - N * log(2 * pi) )
     * - Covariance matrix K(theta)_NxN
     * - Training output y_N
     * - Hyperparameters theta ={ v, l, v_n }
     *
     * Algorithm:
     * 1: Compute lower triangular part of covariance matrix K
     * 2: Compute Cholesky factor L of K
     * 3: Compute prediction alpha = K^-1 * y:
     *    - triangular solve L * beta = y
     *    - triangular solve L^T * alpha = beta
     * 5: Compute beta = K^-1 * y
     * 6: Compute negative log likelihood loss
     *    - Calculate sum_i^N log(L_ii^2)
     *    - Calculate y^T * beta
     *    - Add constant N * log (2 * pi)
     */

    // Tiled covariance matrix K_NxN
    auto K_tiles = make_tiled_dataset<double>(
        sched,
        n_tiles * n_tiles,
        [&](std::size_t tile_index)
        { return covariance_tile_on(sched, n_tiles, tile_index / n_tiles, tile_index % n_tiles); });
    for (std::size_t row = 0; row < n_tiles; row++)
    {
        for (std::size_t col = 0; col <= row; col++)
        {
            K_tiles[row * n_tiles + col] = detail::named_make_tile<gen_tile_covariance>(
                sched,
                covariance_tile_on(sched, n_tiles, row, col),
                "assemble_tiled_K",
                K_tiles[row * n_tiles + col],
                row,
                col,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }

    // Tiled intermediate solution
    auto alpha_tiles = make_tiled_dataset<double>(
        sched, n_tiles, [&](std::size_t tile_index) { return alpha_tile_on(sched, n_tiles, tile_index); });
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = detail::named_make_tile<gen_tile_output>(
            sched,
            alpha_tile_on(sched, n_tiles, i),
            "assemble_tiled_alpha",
            alpha_tiles[i],
            i,
            n_tile_size,
            training_output);
    }

    // Tiled output
    auto y_tiles = make_tiled_dataset<double>(
        sched, n_tiles, [&](std::size_t tile_index) { return prediction_tile_on(sched, n_tiles, tile_index); });
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        y_tiles[i] = detail::named_make_tile<gen_tile_output>(
            sched,
            prediction_tile_on(sched, n_tiles, i),
            "assemble_tiled_alpha",
            y_tiles[i],
            i,
            n_tile_size,
            training_output);
    }

    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(sched, K_tiles, n_tile_size, n_tiles);

    // Launch asynchronous triangular solve  L * (L^T * alpha) = y
    forward_solve_tiled(sched, K_tiles, alpha_tiles, n_tile_size, n_tiles);
    backward_solve_tiled(sched, K_tiles, alpha_tiles, n_tile_size, n_tiles);

    // Launch asynchronous loss computation
    return compute_loss_tiled(sched, K_tiles, alpha_tiles, y_tiles, n_tile_size, n_tiles).get();
}

/**
 * @brief Perform optimization for a given number of iterations
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 *
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @param adam_params The Adam optimizer hyperparameters
 * @param sek_params The kernel hyperparameters
 * @param trainable_params The vector containing a bool whether to train a hyperparameter
 *
 * @return A vector containing the loss values of each iteration
 */
template <typename Scheduler = tiled_scheduler_local>
std::vector<double>
optimize(Scheduler &sched,
         const std::vector<double> &training_input,
         const std::vector<double> &training_output,
         std::size_t n_tiles,
         std::size_t n_tile_size,
         std::size_t n_regressors,
         const AdamParams &adam_params,
         SEKParams &sek_params,
         std::vector<bool> trainable_params,
         std::size_t start_iter = 0)
{
    /*
     * - Hyperparameters theta={v, l, v_n}
     * - Covariance matrix K(theta)
     * - Training ouput y
     *
     * Algorithm:
     * for opt_iter:
     *   1: Compute distance for entries of covariance matrix K
     *   2: Compute lower triangular part of K with distance
     *   3: Compute lower triangular gradients for delta(K)/delta(v), and delta(K)/delta(l) with distance
     *
     *   4: Compute Cholesky factor L of K
     *   5: Compute K^-1:
     *       - triangular solve L * {} = I
     *       - triangular solve L^T * K^-1 = {}
     *   6: Compute beta = K^-1 * y
     *
     *   7: Compute negative log likelihood loss
     *       - Calculate 0.5 sum_i^N log(L_ii^2)
     *       - Calculate 0.5 y^T * beta
     *       - Add constant N / 2 * log (2 * pi)
     *
     *   8: Compute delta(loss)/delta(param_i)
     *       - Compute trace(K^-1 * delta(K)/delta(theta_i))
     *       - Compute beta^T *  delta(K)/delta(theta_i) * beta
     *   9: Update hyperparameters theta with Adam optimizer
     *       - m_T = beta1 * m_T-1 + (1 - beta1) * g_T
     *       - w_T = beta2 + w_T-1 + (1 - beta2) * g_T^2
     *       - nu_T = nu * sqrt(1 - beta2_T) / (1 - beta1_T)
     *       - theta_T = theta_T-1 - nu_T * m_T / (sqrt(w_T) + epsilon)
     * endfor
     */

    // data holder for computed loss values
    std::vector<double> losses;
    losses.reserve(static_cast<std::size_t>(adam_params.opt_iter));

    // Tiled output
    auto y_tiles = make_tiled_dataset<double>(
        sched, n_tiles, [&](std::size_t tile_index) { return prediction_tile_on(sched, n_tiles, tile_index); });
    // Launch asynchronous assembly of output y
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        y_tiles[i] = detail::named_make_tile<gen_tile_output>(
            sched, prediction_tile_on(sched, n_tiles, i), "assemble_y", y_tiles[i], i, n_tile_size, training_output);
    }

    //////////////////////////////////////////////////////////////////////////////
    // per-loop tiles

    // Tiled covariance matrix K_NxN
    auto K_tiles = make_tiled_dataset<double>(
        sched,
        n_tiles * n_tiles,
        [&](std::size_t tile_index)
        { return covariance_tile_on(sched, n_tiles, tile_index / n_tiles, tile_index % n_tiles); });

    // Tiled inverse covariance matrix K^-1_NxN
    auto K_inv_tiles = make_tiled_dataset<double>(
        sched,
        n_tiles * n_tiles,
        [&](std::size_t tile_index)
        { return K_inv_tile_on(sched, n_tiles, tile_index / n_tiles, tile_index % n_tiles); });

    // Tiled intermediate solution
    auto alpha_tiles = make_tiled_dataset<double>(
        sched, n_tiles, [&](std::size_t tile_index) { return alpha_tile_on(sched, n_tiles, tile_index); });

    // Tiled future data structures for gradients

    // Tiled covariance with gradient v
    auto grad_v_tiles = make_tiled_dataset<double>(
        sched,
        n_tiles * n_tiles,
        [&](std::size_t tile_index)
        { return K_grad_v_tile_on(sched, n_tiles, tile_index / n_tiles, tile_index % n_tiles); });

    // Tiled covariance with gradient l
    auto grad_l_tiles = make_tiled_dataset<double>(
        sched,
        n_tiles * n_tiles,
        [&](std::size_t tile_index)
        { return K_grad_l_tile_on(sched, n_tiles, tile_index / n_tiles, tile_index % n_tiles); });

    auto inter_alpha = make_tiled_dataset<double>(
        sched, n_tiles, [&](std::size_t tile_index) { return inter_alpha_tile_on(sched, n_tiles, tile_index); });

    auto diag_tiles = make_tiled_dataset<double>(
        sched, n_tiles, [&](std::size_t tile_index) { return diag_tile_on(sched, n_tiles, tile_index); });

    //////////////////////////////////////////////////////////////////////////////
    // Perform optimization
    for (std::size_t iter = start_iter; iter < static_cast<std::size_t>(adam_params.opt_iter); iter++)
    {
        ///////////////////////////////////////////////////////////////////////////
        // Launch asynchronous assembly of tiled covariance matrix, derivative of covariance matrix
        // vector w.r.t. to vertical lengthscale and derivative of covariance
        // matrix vector w.r.t. to lengthscale
        for (std::size_t i = 0; i < n_tiles; i++)
        {
            for (std::size_t j = 0; j <= i; j++)
            {
                // Compute the distance (z_i - z_j) of K entries to reuse
                hpx::shared_future<mutable_tile_data<double>> cov_dists = detail::named_async<gen_tile_distance>(
                    "assemble_cov_dist", i, j, n_tile_size, n_regressors, sek_params, training_input);

                K_tiles[i * n_tiles + j] = detail::named_make_tile<gen_tile_covariance_with_distance>(
                    sched,
                    covariance_tile_on(sched, n_tiles, i, j),
                    "assemble_K",
                    K_tiles[i * n_tiles + j],
                    i,
                    j,
                    n_tile_size,
                    sek_params,
                    cov_dists);
                if (trainable_params[0])
                {
                    grad_l_tiles[i * n_tiles + j] = detail::named_make_tile<gen_tile_grad_l>(
                        sched,
                        K_grad_l_tile_on(sched, n_tiles, i, j),
                        "assemble_gradl",
                        grad_l_tiles[i * n_tiles + j],
                        n_tile_size,
                        sek_params,
                        cov_dists);
                    if (i != j)
                    {
                        grad_l_tiles[j * n_tiles + i] = detail::named_make_tile<gen_tile_transpose>(
                            sched,
                            K_grad_l_tile_on(sched, n_tiles, j, i),
                            "assemble_gradl_t",
                            grad_l_tiles[j * n_tiles + i],
                            n_tile_size,
                            n_tile_size,
                            grad_l_tiles[i * n_tiles + j]);
                    }
                }

                if (trainable_params[1])
                {
                    grad_v_tiles[i * n_tiles + j] = detail::named_make_tile<gen_tile_grad_v>(
                        sched,
                        K_grad_v_tile_on(sched, n_tiles, i, j),
                        "assemble_gradv",
                        grad_v_tiles[i * n_tiles + j],
                        n_tile_size,
                        sek_params,
                        cov_dists);
                    if (i != j)
                    {
                        grad_v_tiles[j * n_tiles + i] = detail::named_make_tile<gen_tile_transpose>(
                            sched,
                            K_grad_v_tile_on(sched, n_tiles, j, i),
                            "assemble_gradv_t",
                            grad_v_tiles[j * n_tiles + i],
                            n_tile_size,
                            n_tile_size,
                            grad_v_tiles[i * n_tiles + j]);
                    }
                }
            }
        }

        // Assembly with reallocation -> optimize to only set existing values
        for (std::size_t i = 0; i < n_tiles; i++)
        {
            alpha_tiles[i] = detail::named_make_tile<gen_tile_zeros>(
                sched, alpha_tile_on(sched, n_tiles, i), "assemble_tiled_alpha", alpha_tiles[i], n_tile_size);
        }

        for (std::size_t i = 0; i < n_tiles; i++)
        {
            for (std::size_t j = 0; j < n_tiles; j++)
            {
                if (i == j)
                {
                    K_inv_tiles[i * n_tiles + j] = detail::named_make_tile<gen_tile_identity>(
                        sched,
                        K_inv_tile_on(sched, n_tiles, i, j),
                        "assemble_identity_matrix",
                        K_inv_tiles[i * n_tiles + j],
                        n_tile_size);
                }
                else
                {
                    K_inv_tiles[i * n_tiles + j] = detail::named_make_tile<gen_tile_zeros>(
                        sched,
                        K_inv_tile_on(sched, n_tiles, i, j),
                        "assemble_identity_matrix",
                        K_inv_tiles[i * n_tiles + j],
                        n_tile_size * n_tile_size);
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Launch asynchronous Cholesky decomposition: K = L * L^T
        right_looking_cholesky_tiled(sched, K_tiles, n_tile_size, n_tiles);

        ///////////////////////////////////////////////////////////////////////////
        // Launch asynchronous compute K^-1 through L* (L^T * X) = I
        forward_solve_tiled_matrix(sched, K_tiles, K_inv_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);
        backward_solve_tiled_matrix(sched, K_tiles, K_inv_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);

        ///////////////////////////////////////////////////////////////////////////
        // Launch asynchronous compute beta = inv(K) * y
        matrix_vector_tiled(sched, K_inv_tiles, y_tiles, alpha_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);

        ///////////////////////////////////////////////////////////////////////////
        // Launch asynchronous loss computation where
        // loss(theta) = 0.5 * ( log(det(K)) - y^T * K^-1 * y - N * log(2 * pi) )
        auto loss_value = compute_loss_tiled(sched, K_tiles, alpha_tiles, y_tiles, n_tile_size, n_tiles);

        ///////////////////////////////////////////////////////////////////////////
        // Launch asynchronous update of the hyperparameters
        if (trainable_params[0])
        {  // lengthscale
            update_hyperparameter_tiled_lengthscale(
                sched,
                K_inv_tiles,
                grad_l_tiles,
                alpha_tiles,
                adam_params,
                diag_tiles,
                inter_alpha,
                sek_params,
                n_tile_size,
                n_tiles,
                iter,
                0);
        }
        if (trainable_params[1])
        {  // vertical_lengthscale
            update_hyperparameter_tiled_lengthscale(
                sched,
                K_inv_tiles,
                grad_v_tiles,
                alpha_tiles,
                adam_params,
                diag_tiles,
                inter_alpha,
                sek_params,
                n_tile_size,
                n_tiles,
                iter,
                1);
        }
        if (trainable_params[2])
        {  // noise_variance
            update_hyperparameter_tiled_noise_variance(
                sched, K_inv_tiles, alpha_tiles, adam_params, sek_params, n_tile_size, n_tiles, iter, 2);
        }
        // Synchronize after iteration
        losses.push_back(loss_value.get());
    }
    return losses;
}

/**
 * @brief Perform a single optimization step
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 *
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @param adam_params The Adam optimizer hyperparameters
 * @param sek_params The kernel hyperparameters
 * @param trainable_params The vector containing a bool whether to train a hyperparameter
 *
 * @param iter The current optimization iteration
 *
 * @return The loss value
 */
template <typename Scheduler = tiled_scheduler_local>
double optimize_step(Scheduler &sched,
                     const std::vector<double> &training_input,
                     const std::vector<double> &training_output,
                     std::size_t n_tiles,
                     std::size_t n_tile_size,
                     std::size_t n_regressors,
                     AdamParams &adam_params,
                     SEKParams &sek_params,
                     std::vector<bool> trainable_params,
                     std::size_t iter)
{
    // No point in copy&pasting everything for this function
    const auto old_opt_iter = adam_params.opt_iter;
    adam_params.opt_iter = iter + 1;
    const auto r = optimize(
        sched,
        training_input,
        training_output,
        n_tiles,
        n_tile_size,
        n_regressors,
        adam_params,
        sek_params,
        trainable_params,
        iter);
    adam_params.opt_iter = old_opt_iter;
    return r[0];
}

}  // end of namespace cpu

GPRAT_NS_END

#endif
