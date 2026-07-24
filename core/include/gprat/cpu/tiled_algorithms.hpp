#ifndef GPRAT_CPU_TILED_ALGORITHMS_H
#define GPRAT_CPU_TILED_ALGORITHMS_H

#pragma once

#include "gprat/cpu/adapter_cblas_fp64.hpp"
#include "gprat/cpu/gp_algorithms.hpp"
#include "gprat/cpu/gp_optimizer.hpp"
#include "gprat/cpu/gp_uncertainty.hpp"
#include "gprat/detail/async_helpers.hpp"
#include "gprat/detail/config.hpp"
#include "gprat/hyperparameters.hpp"
#include "gprat/kernels.hpp"
#include "gprat/scheduler.hpp"

#include <hpx/future.hpp>

GPRAT_NS_BEGIN

namespace cpu
{

namespace impl
{
void update_parameters(
    const AdamParams &adam_params,
    SEKParams &sek_params,
    std::size_t N,
    std::size_t n_tiles,
    std::size_t iter,
    std::size_t param_idx,
    double trace,
    double dot,
    bool jitter,
    double factor);
}

// Tiled Cholesky Algorithm

/**
 * @brief Perform right-looking tiled Cholesky decomposition.
 *
 * @param tiles Tiled matrix represented as a vector of futurized tiles, containing the
 *        covariance matrix, afterwards the Cholesky decomposition.
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 */
template <typename Tiles, typename Scheduler = tiled_scheduler_local>
void right_looking_cholesky_tiled(Scheduler &sched, Tiles &tiles, std::size_t N, std::size_t n_tiles)
{
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // POTRF: Compute Cholesky factor L
        tiles[k * n_tiles + k] = detail::named_dataflow<potrf>(
            sched, cholesky_potrf_on(sched, n_tiles, k), "cholesky_tiled", tiles[k * n_tiles + k], N);
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TRSM:  Solve X * L^T = A
            tiles[m * n_tiles + k] = detail::named_dataflow<trsm>(
                sched,
                cholesky_trsm_on(sched, n_tiles, k, m),
                "cholesky_tiled",
                tiles[k * n_tiles + k],
                tiles[m * n_tiles + k],
                N,
                N,
                Blas_trans,
                Blas_right);
        }
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // SYRK:  A = A - B * B^T
            tiles[m * n_tiles + m] = detail::named_dataflow<syrk>(
                sched,
                cholesky_syrk_on(sched, n_tiles, m),
                "cholesky_tiled",
                tiles[m * n_tiles + m],
                tiles[m * n_tiles + k],
                N);
            for (std::size_t n = k + 1; n < m; n++)
            {
                // GEMM: C = C - A * B^T
                tiles[m * n_tiles + n] = detail::named_dataflow<gemm>(
                    sched,
                    cholesky_gemm_on(sched, n_tiles, k, m, n),
                    "cholesky_tiled",
                    tiles[m * n_tiles + k],
                    tiles[n * n_tiles + k],
                    tiles[m * n_tiles + n],
                    N,
                    N,
                    N,
                    Blas_no_trans,
                    Blas_trans);
            }
        }
    }
}

// Tiled Triangular Solve Algorithms

/**
 * @brief Perform tiled forward triangular matrix-vector solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side vector, afterwards containing the tiled solution vector
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 */
template <typename Tiles, typename Scheduler = tiled_scheduler_local>
void forward_solve_tiled(Scheduler &sched, Tiles &ft_tiles, Tiles &ft_rhs, std::size_t N, std::size_t n_tiles)
{
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // TRSM: Solve L * x = a
        ft_rhs[k] = detail::named_dataflow<trsv>(
            sched,
            solve_trsv_on(sched, n_tiles, k),
            "triangular_solve_tiled",
            ft_tiles[k * n_tiles + k],
            ft_rhs[k],
            N,
            Blas_no_trans);
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // GEMV: b = b - A * a
            ft_rhs[m] = detail::named_dataflow<gemv>(
                sched,
                solve_gemv_on(sched, n_tiles, k, m),
                "triangular_solve_tiled",
                ft_tiles[m * n_tiles + k],
                ft_rhs[k],
                ft_rhs[m],
                N,
                N,
                Blas_substract,
                Blas_no_trans);
        }
    }
}

/**
 * @brief Perform tiled backward triangular matrix-vector solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side vector, afterwards containing the tiled solution vector
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 */
template <typename Tiles, typename Scheduler = tiled_scheduler_local>
void backward_solve_tiled(Scheduler &sched, Tiles &ft_tiles, Tiles &ft_rhs, std::size_t N, std::size_t n_tiles)
{
    for (int k_ = static_cast<int>(n_tiles) - 1; k_ >= 0; k_--)  // int instead of std::size_t for last comparison
    {
        std::size_t k = static_cast<std::size_t>(k_);
        // TRSM: Solve L^T * x = a
        ft_rhs[k] = detail::named_dataflow<trsv>(
            sched,
            solve_trsm_on(sched, n_tiles, k),
            "triangular_solve_tiled",
            ft_tiles[k * n_tiles + k],
            ft_rhs[k],
            N,
            Blas_trans);
        for (int m_ = k_ - 1; m_ >= 0; m_--)  // int instead of std::size_t for last comparison
        {
            std::size_t m = static_cast<std::size_t>(m_);
            // GEMV:b = b - A^T * a
            ft_rhs[m] = detail::named_dataflow<gemv>(
                sched,
                solve_gemv_on(sched, n_tiles, k, m),
                "triangular_solve_tiled",
                ft_tiles[k * n_tiles + m],
                ft_rhs[k],
                ft_rhs[m],
                N,
                N,
                Blas_substract,
                Blas_trans);
        }
    }
}

/**
 * @brief Perform tiled forward triangular matrix-matrix solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side matrix, afterwards containing the tiled solution matrix.
 * @param N Tile size of first dimension.
 * @param M Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
template <typename Tiles, typename Scheduler = tiled_scheduler_local>
void forward_solve_tiled_matrix(
    Scheduler &sched,
    Tiles &ft_tiles,
    Tiles &ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{
    for (std::size_t c = 0; c < m_tiles; c++)
    {
        for (std::size_t k = 0; k < n_tiles; k++)
        {
            // TRSM: solve L * X = A
            ft_rhs[k * m_tiles + c] = detail::named_dataflow<trsm>(
                sched,
                solve_matrix_trsm_on(sched, m_tiles, c, k),
                "triangular_solve_tiled_matrix",
                ft_tiles[k * n_tiles + k],
                ft_rhs[k * m_tiles + c],
                N,
                M,
                Blas_no_trans,
                Blas_left);
            for (std::size_t m = k + 1; m < n_tiles; m++)
            {
                // GEMM: C = C - A * B
                ft_rhs[m * m_tiles + c] = detail::named_dataflow<gemm>(
                    sched,
                    solve_matrix_gemm_on(sched, m_tiles, c, k, m),
                    "triangular_solve_tiled_matrix",
                    ft_tiles[m * n_tiles + k],
                    ft_rhs[k * m_tiles + c],
                    ft_rhs[m * m_tiles + c],
                    N,
                    M,
                    N,
                    Blas_no_trans,
                    Blas_no_trans);
            }
        }
    }
}

/**
 * @brief Perform tiled backward triangular matrix-matrix solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side matrix, afterwards containing the tiled solution matrix.
 * @param N Tile size of first dimension.
 * @param M Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
template <typename Tiles, typename Scheduler = tiled_scheduler_local>
void backward_solve_tiled_matrix(
    Scheduler &sched,
    Tiles &ft_tiles,
    Tiles &ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{
    for (std::size_t c = 0; c < m_tiles; c++)
    {
        for (int k_ = static_cast<int>(n_tiles) - 1; k_ >= 0; k_--)  // int instead of std::size_t for last comparison
        {
            std::size_t k = static_cast<std::size_t>(k_);
            // TRSM: solve L^T * X = A
            ft_rhs[k * m_tiles + c] = detail::named_dataflow<trsm>(
                sched,
                solve_matrix_trsm_on(sched, m_tiles, c, k),
                "triangular_solve_tiled_matrix",
                ft_tiles[k * n_tiles + k],
                ft_rhs[k * m_tiles + c],
                N,
                M,
                Blas_trans,
                Blas_left);
            for (int m_ = k_ - 1; m_ >= 0; m_--)  // int instead of std::size_t for last comparison
            {
                std::size_t m = static_cast<std::size_t>(m_);
                // GEMM: C = C - A^T * B
                ft_rhs[m * m_tiles + c] = detail::named_dataflow<gemm>(
                    sched,
                    solve_matrix_gemm_on(sched, m_tiles, c, k, m),
                    "triangular_solve_tiled_matrix",
                    ft_tiles[k * n_tiles + m],
                    ft_rhs[k * m_tiles + c],
                    ft_rhs[m * m_tiles + c],
                    N,
                    M,
                    N,
                    Blas_trans,
                    Blas_no_trans);
            }
        }
    }
}

/**
 * @brief Perform tiled matrix-vector multiplication
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_vector Tiled vector represented as a vector of futurized tiles.
 * @param ft_rhs Tiled solution represented as a vector of futurized tiles.
 * @param N_row Tile size of first dimension.
 * @param N_col Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
template <typename Tiles, typename Scheduler = tiled_scheduler_local>
void matrix_vector_tiled(Scheduler &sched,
                         Tiles &ft_tiles,
                         Tiles &ft_vector,
                         Tiles &ft_rhs,
                         std::size_t N_row,
                         std::size_t N_col,
                         std::size_t n_tiles,
                         std::size_t m_tiles)
{
    for (std::size_t k = 0; k < m_tiles; k++)
    {
        for (std::size_t m = 0; m < n_tiles; m++)
        {
            ft_rhs[k] = detail::named_dataflow<gemv>(
                sched,
                multiply_gemv_on(sched, n_tiles, k, m),
                "prediction_tiled",
                ft_tiles[k * n_tiles + m],
                ft_vector[m],
                ft_rhs[k],
                N_row,
                N_col,
                Blas_add,
                Blas_no_trans);
        }
    }
}

/**
 * @brief Perform tiled symmetric k-rank update on diagonal tiles
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_vector Tiled vector holding the diagonal tile results
 * @param N Tile size of first dimension.
 * @param M Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
template <typename Tiles, typename Scheduler = tiled_scheduler_local>
void symmetric_matrix_matrix_diagonal_tiled(
    Scheduler &sched,
    Tiles &ft_tiles,
    Tiles &ft_vector,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{
    for (std::size_t i = 0; i < m_tiles; ++i)
    {
        for (std::size_t n = 0; n < n_tiles; ++n)
        {
            // Compute inner product to obtain diagonal elements of
            // V^T * V  <=> cross(K) * K^-1 * cross(K)^T
            ft_vector[i] = detail::named_dataflow<dot_diag_syrk>(
                sched,
                k_rank_dot_diag_syrk_on(sched, m_tiles, i),
                "posterior_tiled",
                ft_tiles[n * m_tiles + i],
                ft_vector[i],
                N,
                M);
        }
    }
}

/**
 * @brief Perform tiled symmetric k-rank update (ft_tiles^T * ft_tiles)
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_result Tiled matrix holding the result of the computationi.
 * @param N Tile size of first dimension.
 * @param M Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
template <typename Tiles, typename Scheduler = tiled_scheduler_local>
void symmetric_matrix_matrix_tiled(
    Scheduler &sched,
    Tiles &ft_tiles,
    Tiles &ft_result,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{
    for (std::size_t c = 0; c < m_tiles; c++)
    {
        for (std::size_t k = 0; k < m_tiles; k++)
        {
            for (std::size_t m = 0; m < n_tiles; m++)
            {
                // (SYRK for (c == k) possible)
                // GEMM:  C = C - A^T * B
                ft_result[c * m_tiles + k] = detail::named_dataflow<gemm>(
                    sched,
                    k_rank_gemm_on(sched, m_tiles, c, k, m),
                    "triangular_solve_tiled_matrix",
                    ft_tiles[m * m_tiles + c],
                    ft_tiles[m * m_tiles + k],
                    ft_result[c * m_tiles + k],
                    N,
                    M,
                    M,
                    Blas_trans,
                    Blas_no_trans);
            }
        }
    }
}

/**
 * @brief Compute the difference between two tiled vectors
 * @param ft_minuend Tiled vector that is being subtracted from.
 * @param ft_subtrahend Tiled vector that is being subtracted.
 * @param M Tile size dimension.
 * @param m_tiles Number of tiles.
 */
template <typename Tiles, typename Scheduler = tiled_scheduler_local>
void vector_difference_tiled(
    Scheduler &sched, Tiles &ft_minuend, Tiles &ft_subtrahend, std::size_t M, std::size_t m_tiles)
{
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_subtrahend[i] = detail::named_dataflow<axpy>(
            sched, vector_axpy_on(sched, m_tiles, i), "uncertainty_tiled", ft_minuend[i], ft_subtrahend[i], M);
    }
}

/**
 * @brief Extract the tiled diagonals of a tiled matrix
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_vector Tiled vector containing the diagonals of the matrix tiles
 * @param M Tile size per dimension.
 * @param m_tiles Number of tiles per dimension.
 */
template <typename Tiles, typename Scheduler = tiled_scheduler_local>
void matrix_diagonal_tiled(Scheduler &sched, Tiles &ft_tiles, Tiles &ft_vector, std::size_t M, std::size_t m_tiles)
{
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_vector[i] = detail::named_dataflow<get_matrix_diagonal>(
            sched, get_diagonal_on(sched, m_tiles, i), "uncertainty_tiled", ft_tiles[i * m_tiles + i], M);
    }
}

/**
 * @brief Compute the negative log likelihood loss with a tiled covariance matrix K.
 *
 *  Computes l = 0.5 * ( log(det(K)) + y^T * K^-1 * y) + const.)
 *
 * @param ft_tiles Tiled Cholesky factor matrix represented as a vector of futurized tiles.
 * @param ft_alpha Tiled vector containing the solution of K^-1 * y
 * @param ft_y Tiled vector containing the training output y
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 * @return The loss value to be computed
 */
template <typename Tiles, typename Scheduler = tiled_scheduler_local>
hpx::future<double>
compute_loss_tiled(Scheduler &sched, Tiles &ft_tiles, Tiles &ft_alpha, Tiles &ft_y, std::size_t N, std::size_t n_tiles)
{
    std::vector<hpx::future<double>> loss_tiled;
    loss_tiled.reserve(n_tiles);
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        loss_tiled.push_back(detail::named_dataflow<compute_loss>(
            sched,
            compute_loss_on(sched, n_tiles, k),
            "loss_tiled",
            ft_tiles[k * n_tiles + k],
            ft_alpha[k],
            ft_y[k],
            N));
    }
    return detail::named_dataflow<add_losses>("loss_tiled", loss_tiled, N, n_tiles);
}

/**
 * @brief Updates a hyperparameter of the SEK kernel using Adam
 *
 * @param ft_invK Tiled inverse of the covariance matrix K represented as a vector of futurized tiles.
 * @param ft_gradK_param Tiled covariance matrix gradient w.r.t. a hyperparameter.
 * @param ft_alpha Tiled vector containing the precomputed inv(K) * y where y is the training output.
 * @param adam_params Hyperparameter of the Adam optimizer
 * @param sek_params Hyperparameters of the SEK kernel
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 * @param iter Current iteration.
 * @param param_idx Index of the hyperparameter to optimize.
 */
template <typename Tiles, typename Scheduler = tiled_scheduler_local>
void update_hyperparameter_tiled_lengthscale(
    Scheduler &sched,
    const Tiles &ft_invK,
    const Tiles &ft_gradK_param,
    const Tiles &ft_alpha,
    const AdamParams &adam_params,
    Tiles &diag_tiles,   // Diagonal tiles
    Tiles &inter_alpha,  // Intermediate result
    SEKParams &sek_params,
    std::size_t N,
    std::size_t n_tiles,
    std::size_t iter,
    std::size_t param_idx)
{
    /*
     * PART 1:
     * Compute gradient = 0.5 * ( trace(inv(K) * grad(K)_param) + y^T * inv(K) * grad(K)_param * inv(K) * y )
     *
     * 1: Compute   trace(inv(K) * grad(K)_param)
     * 2: Compute   y^T * inv(K) * grad(K)_param * inv(K) * y
     *
     * Update parameter:
     * 3: Update moments
     *      - m_T = beta1 * m_T-1 + (1 - beta1) * g_T
     *      - w_T = beta2 + w_T-1 + (1 - beta2) * g_T^2
     * 4: Adam step:
     *      - nu_T = nu * sqrt(1 - beta2_T) / (1 - beta1_T)
     *      - theta_T = theta_T-1 - nu_T * m_T / (sqrt(w_T) + epsilon)
     */
    hpx::shared_future<double> trace = hpx::make_ready_future(0.0);
    hpx::shared_future<double> dot = hpx::make_ready_future(0.0);
    bool jitter = false;
    double factor = 1.0;

    // Reset our helper tiles
    for (std::size_t d = 0; d < n_tiles; d++)
    {
        diag_tiles[d] = detail::named_make_tile<gen_tile_zeros>(
            sched, diag_tile_on(sched, n_tiles, d), "assemble", diag_tiles[d], N);
        inter_alpha[d] = detail::named_make_tile<gen_tile_zeros>(
            sched, inter_alpha_tile_on(sched, n_tiles, d), "assemble", inter_alpha[d], N);
    }

    ////////////////////////////////////
    // PART 1: Compute gradient
    // Step 1: Compute trace(inv(K)*grad_K_param)
    // Compute diagonal tiles of inv(K) * grad(K)_param
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            diag_tiles[i] = detail::named_dataflow<dot_diag_gemm>(
                sched,
                diag_tile_on(sched, n_tiles, i),
                "trace",
                ft_invK[i * n_tiles + j],
                ft_gradK_param[j * n_tiles + i],
                diag_tiles[i],
                N,
                N);
        }
    }
    // Compute the trace of the diagonal tiles
    for (std::size_t j = 0; j < n_tiles; ++j)
    {
        trace = detail::named_dataflow<compute_trace>(
            sched, diag_tile_on(sched, n_tiles, j), "trace", diag_tiles[j], trace);
    }
    // Not sure if can be done this way
    // Step 2: Compute alpha^T * grad(K)_param * alpha (with alpha = inv(K) * y)
    // Compute inter_alpha = grad(K)_param * alpha
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        for (std::size_t m = 0; m < n_tiles; m++)
        {
            inter_alpha[k] = detail::named_dataflow<gemv>(
                sched,
                inter_alpha_tile_on(sched, n_tiles, k),
                "gemv",
                ft_gradK_param[k * n_tiles + m],
                ft_alpha[m],
                inter_alpha[k],
                N,
                N,
                Blas_add,
                Blas_no_trans);
        }
    }
    // Compute alpha^T * inter_alpha
    for (std::size_t j = 0; j < n_tiles; ++j)
    {
        dot = detail::named_dataflow<compute_dot>(
            sched, inter_alpha_tile_on(sched, n_tiles, j), "grad_right_tiled", inter_alpha[j], ft_alpha[j], dot);
    }

    impl::update_parameters(
        adam_params, sek_params, N, n_tiles, iter, param_idx, trace.get(), dot.get(), jitter, factor);
}

template <typename Tiles, typename Scheduler = tiled_scheduler_local>
void update_hyperparameter_tiled_noise_variance(
    Scheduler &sched,
    const Tiles &ft_invK,
    const Tiles &ft_alpha,
    const AdamParams &adam_params,
    SEKParams &sek_params,
    std::size_t N,
    std::size_t n_tiles,
    std::size_t iter,
    std::size_t param_idx)
{
    /*
     * PART 1:
     * Compute gradient = 0.5 * ( trace(inv(K) * grad(K)_param) + y^T * inv(K) * grad(K)_param * inv(K) * y )
     *
     * 1: Compute   trace(inv(K) * grad(K)_param)
     * 2: Compute   y^T * inv(K) * grad(K)_param * inv(K) * y
     *
     * Update parameter:
     * 3: Update moments
     *      - m_T = beta1 * m_T-1 + (1 - beta1) * g_T
     *      - w_T = beta2 + w_T-1 + (1 - beta2) * g_T^2
     * 4: Adam step:
     *      - nu_T = nu * sqrt(1 - beta2_T) / (1 - beta1_T)
     *      - theta_T = theta_T-1 - nu_T * m_T / (sqrt(w_T) + epsilon)
     */
    hpx::shared_future<double> trace = hpx::make_ready_future(0.0);
    hpx::shared_future<double> dot = hpx::make_ready_future(0.0);
    bool jitter = true;
    double factor = 1.0;

    ////////////////////////////////////
    // PART 1: Compute gradient
    // Step 1: Compute the trace of inv(K) * noise_variance
    for (std::size_t j = 0; j < n_tiles; ++j)
    {
        trace = detail::named_dataflow<compute_trace_diag>(
            sched, K_inv_tile_on(sched, n_tiles, j, j), "grad_left_tiled", ft_invK[j * n_tiles + j], trace, N);
    }
    ////////////////////////////////////
    // Step 2: Compute the alpha^T * alpha * noise_variance
    for (std::size_t j = 0; j < n_tiles; ++j)
    {
        dot = detail::named_dataflow<compute_dot>(
            sched, alpha_tile_on(sched, n_tiles, j), "grad_right_tiled", ft_alpha[j], ft_alpha[j], dot);
    }

    factor = compute_sigmoid(to_unconstrained(sek_params.noise_variance, true));

    impl::update_parameters(
        adam_params, sek_params, N, n_tiles, iter, param_idx, trace.get(), dot.get(), jitter, factor);
}

}  // end of namespace cpu

GPRAT_NS_END

#endif
