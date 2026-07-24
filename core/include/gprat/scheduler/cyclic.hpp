#ifndef GPRAT_SCHEDULER_CYCLIC_HPP
#define GPRAT_SCHEDULER_CYCLIC_HPP

#pragma once

#include "gprat/detail/actions.hpp"
#include "gprat/detail/config.hpp"
#include "gprat/scheduler.hpp"

GPRAT_NS_BEGIN

struct tiled_scheduler_cyclic : tiled_scheduler_distributed
{
    using tiled_scheduler_distributed::tiled_scheduler_distributed;

    /// @brief Create a new scheduler that targets all localities.
    explicit tiled_scheduler_cyclic(std::size_t in_width = 1) :
        num_localities(localities_.size()),
        width(in_width),
        height(num_localities / width)
    {
        if (num_localities % width != 0)
        {
            throw std::invalid_argument("num_localities must be divisible by width");
        }
    }

    /// @brief Create a new scheduler that targets the given localities.
    explicit tiled_scheduler_cyclic(std::vector<hpx::id_type> in_localities, std::size_t in_width = 1) :
        tiled_scheduler_distributed(std::move(in_localities)),
        num_localities(localities_.size()),
        width(in_width),
        height(num_localities / width)
    {
        if (num_localities % width != 0)
        {
            throw std::invalid_argument("num_localities must be divisible by width");
        }
    }

    std::size_t num_localities;
    std::size_t width;
    std::size_t height;
};

constexpr std::size_t
covariance_tile_on(const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t row, std::size_t col)
{
    return (row % sched.height) + (col % sched.width);
}

constexpr std::size_t
cross_covariance_tile_on(const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t row, std::size_t col)
{
    return (row % sched.height) + (col % sched.width);
}

constexpr std::size_t alpha_tile_on(const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t i)
{
    return (i % sched.height) + (i % sched.width);
}

constexpr std::size_t prediction_tile_on(const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t i)
{
    return (i % sched.height) + (i % sched.width);
}

constexpr std::size_t cholesky_potrf_on(const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t k)
{
    return (k % sched.height) + (k % sched.width);
}

constexpr std::size_t cholesky_syrk_on(const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t m)
{
    return (m % sched.height) + (m % sched.width);
}

constexpr std::size_t
cholesky_trsm_on(const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t k, std::size_t m)
{
    return (m % sched.height) + (k % sched.width);
}

constexpr std::size_t cholesky_gemm_on(
    const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t k, std::size_t m, std::size_t n)
{
    return (m % sched.height) + (n % sched.width);
}

constexpr std::size_t solve_trsv_on(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k)
{
    return (k % sched.height) + (k % sched.width);
}

constexpr std::size_t solve_trsm_on(const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t k)
{
    return (k % sched.height) + (k % sched.width);
}

constexpr std::size_t
solve_gemv_on(const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t k, std::size_t m)
{
    return (k % sched.height) + (m % sched.width);
}

constexpr std::size_t
solve_matrix_trsm_on(const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t c, std::size_t k)
{
    return (k % sched.height) + (c % sched.width);
}

constexpr std::size_t solve_matrix_gemm_on(
    const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t c, std::size_t k, std::size_t m)
{
    return (m % sched.height) + (c % sched.width);
}

constexpr std::size_t
multiply_gemv_on(const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t k, std::size_t m)
{
    return (k % sched.height) + (m % sched.width);
}

constexpr std::size_t
k_rank_dot_diag_syrk_on(const tiled_scheduler_cyclic &sched, std::size_t /*n_tiles*/, std::size_t k)
{
    return (k % sched.height) + (k % sched.width);
}

constexpr std::size_t
k_rank_gemm_on(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t c, std::size_t k, std::size_t m)
{
    return (k * n_tiles + m) % sched.num_localities;
}

constexpr std::size_t vector_axpy_on(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k)
{
    return (k * n_tiles + k) % sched.num_localities;
}

constexpr std::size_t get_diagonal_on(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k)
{
    return (k * n_tiles + k) % sched.num_localities;
}

constexpr std::size_t compute_loss_on(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k)
{
    return (k * n_tiles + k) % sched.num_localities;
}

GPRAT_NS_END

#endif
