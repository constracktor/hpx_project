#ifndef GPRAT_CPU_SCHEDULER_HPP
#define GPRAT_CPU_SCHEDULER_HPP

#pragma once

#include "gprat/detail/async_helpers.hpp"

// TODO: move to separate header
#include "gprat/tile_data.hpp"

#include <hpx/future.hpp>
#include <vector>

GPRAT_NS_BEGIN

using tiled_scheduler_local = basic_local_scheduler;

template <typename T>
using tiled_dataset_local = std::vector<hpx::shared_future<mutable_tile_data<T>>>;

template <typename Scheduler, typename T>
struct tile_dataset_type;

template <typename T>
struct tile_dataset_type<tiled_scheduler_local, T>
{
    using type = tiled_dataset_local<T>;
};

template <typename T, typename Mapper>
tiled_dataset_local<T> make_tiled_dataset(const tiled_scheduler_local &, std::size_t num_tiles, Mapper &&)
{
    return std::vector<hpx::shared_future<mutable_tile_data<T>>>{ num_tiles };
}

// =============================================================
// local scheduler

constexpr std::size_t covariance_tile_on(
    const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*row*/, std::size_t /*col*/)
{
    return 0;
}

constexpr std::size_t cross_covariance_tile_on(
    const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*row*/, std::size_t /*col*/)
{
    return 0;
}

constexpr std::size_t alpha_tile_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*i*/)
{
    return 0;
}

constexpr std::size_t
prediction_tile_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*i*/)
{
    return 0;
}

constexpr std::size_t t_cross_covariance_tile_on(
    const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*row*/, std::size_t /*col*/)
{
    return 0;
}

constexpr std::size_t prior_K_tile_on(
    const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*row*/, std::size_t /*col*/)
{
    return 0;
}

constexpr std::size_t K_inv_tile_on(
    const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*row*/, std::size_t /*col*/)
{
    return 0;
}

constexpr std::size_t K_grad_v_tile_on(
    const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*row*/, std::size_t /*col*/)
{
    return 0;
}

constexpr std::size_t K_grad_l_tile_on(
    const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*row*/, std::size_t /*col*/)
{
    return 0;
}

constexpr std::size_t
uncertainty_tile_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*i*/)
{
    return 0;
}

constexpr std::size_t
inter_alpha_tile_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*i*/)
{
    return 0;
}

constexpr std::size_t diag_tile_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*i*/)
{
    return 0;
}

constexpr std::size_t
cholesky_potrf_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*k*/)
{
    return 0;
}

constexpr std::size_t
cholesky_syrk_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*m*/)
{
    return 0;
}

constexpr std::size_t
cholesky_trsm_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*k*/, std::size_t /*m*/)
{
    return 0;
}

constexpr std::size_t cholesky_gemm_on(const tiled_scheduler_local & /*sched*/,
                                       std::size_t /*n_tiles*/,
                                       std::size_t /*k*/,
                                       std::size_t /*m*/,
                                       std::size_t /*n*/)
{
    return 0;
}

constexpr std::size_t solve_trsv_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*k*/)
{
    return 0;
}

constexpr std::size_t solve_trsm_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*k*/)
{
    return 0;
}

constexpr std::size_t
solve_gemv_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*k*/, std::size_t /*m*/)
{
    return 0;
}

constexpr std::size_t solve_matrix_trsm_on(
    const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*c*/, std::size_t /*k*/)
{
    return 0;
}

constexpr std::size_t solve_matrix_gemm_on(const tiled_scheduler_local & /*sched*/,
                                           std::size_t /*n_tiles*/,
                                           std::size_t /*c*/,
                                           std::size_t /*k*/,
                                           std::size_t /*m*/)
{
    return 0;
}

constexpr std::size_t
multiply_gemv_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*k*/, std::size_t /*m*/)
{
    return 0;
}

constexpr std::size_t
k_rank_dot_diag_syrk_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*k*/)
{
    return 0;
}

constexpr std::size_t k_rank_gemm_on(const tiled_scheduler_local & /*sched*/,
                                     std::size_t /*n_tiles*/,
                                     std::size_t /*c*/,
                                     std::size_t /*k*/,
                                     std::size_t /*m*/)
{
    return 0;
}

constexpr std::size_t
vector_axpy_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*k*/)
{
    return 0;
}

constexpr std::size_t
get_diagonal_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*k*/)
{
    return 0;
}

constexpr std::size_t
compute_loss_on(const tiled_scheduler_local & /*sched*/, std::size_t /*n_tiles*/, std::size_t /*k*/)
{
    return 0;
}

GPRAT_NS_END

#endif
