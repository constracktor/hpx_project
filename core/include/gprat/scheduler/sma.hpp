#ifndef GPRAT_SCHEDULER_SMA_HPP
#define GPRAT_SCHEDULER_SMA_HPP

#pragma once

#include "gprat/detail/actions.hpp"
#include "gprat/detail/config.hpp"
#include "gprat/scheduler.hpp"

GPRAT_NS_BEGIN

struct tiled_scheduler_sma : tiled_scheduler_distributed
{
    using tiled_scheduler_distributed::tiled_scheduler_distributed;

    std::size_t num_localities = localities_.size();
};

constexpr std::size_t
covariance_tile_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t
cross_covariance_tile_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t alpha_tile_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t i)
{
    return (2 * i) % sched.num_localities;
}

constexpr std::size_t prediction_tile_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t i)
{
    return (2 * i) % sched.num_localities;
}

constexpr std::size_t
t_cross_covariance_tile_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t
prior_K_tile_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t
K_inv_tile_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t
K_grad_v_tile_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t
K_grad_l_tile_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t uncertainty_tile_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t i)
{
    return (2 * i) % sched.num_localities;
}

constexpr std::size_t inter_alpha_tile_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t i)
{
    return (2 * i) % sched.num_localities;
}

constexpr std::size_t diag_tile_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t i)
{
    return i % sched.num_localities;
}

constexpr std::size_t cholesky_potrf_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

constexpr std::size_t cholesky_syrk_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t m)
{
    return (2 * m) % sched.num_localities;
}

constexpr std::size_t
cholesky_trsm_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t k, std::size_t m)
{
    return (k + m) % sched.num_localities;
}

constexpr std::size_t cholesky_gemm_on(
    const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t /*k*/, std::size_t m, std::size_t n)
{
    return (m + n) % sched.num_localities;
}

constexpr std::size_t solve_trsv_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

constexpr std::size_t solve_trsm_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

constexpr std::size_t
solve_gemv_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t k, std::size_t m)
{
    return (k + m) % sched.num_localities;
}

constexpr std::size_t
solve_matrix_trsm_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t c, std::size_t k)
{
    return (k + c) % sched.num_localities;
}

constexpr std::size_t solve_matrix_gemm_on(
    const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t c, std::size_t /*k*/, std::size_t m)
{
    return (c + m) % sched.num_localities;
}

constexpr std::size_t
multiply_gemv_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t k, std::size_t m)
{
    return (k + m) % sched.num_localities;
}

constexpr std::size_t k_rank_dot_diag_syrk_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

constexpr std::size_t k_rank_gemm_on(
    const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t /*c*/, std::size_t k, std::size_t m)
{
    return (k + m) % sched.num_localities;
}

constexpr std::size_t vector_axpy_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

constexpr std::size_t get_diagonal_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

constexpr std::size_t compute_loss_on(const tiled_scheduler_sma &sched, std::size_t /*n_tiles*/, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

GPRAT_NS_END

#endif
