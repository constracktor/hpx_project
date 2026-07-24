#ifndef GPRAT_CPU_GP_OPTIMIZER_ACTIONS_HPP
#define GPRAT_CPU_GP_OPTIMIZER_ACTIONS_HPP

#pragma once

#include "gprat/cpu/gp_optimizer.hpp"
#include "gprat/detail/actions.hpp"
#include "gprat/detail/config.hpp"
#include "gprat/tiled_dataset.hpp"

GPRAT_NS_BEGIN

namespace cpu
{
hpx::future<tile_handle<double>> gen_tile_covariance_with_distance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    const SEKParams &sek_params,
    const const_tile_data<double> &distance);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_covariance_with_distance_distributed);

hpx::future<tile_handle<double>> gen_tile_grad_l_distributed(
    const tile_handle<double> &tile,
    std::size_t N,
    const SEKParams &sek_params,
    const const_tile_data<double> &distance);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_grad_l_distributed);

hpx::future<tile_handle<double>> gen_tile_grad_v_distributed(
    const tile_handle<double> &tile,
    std::size_t N,
    const SEKParams &sek_params,
    const const_tile_data<double> &distance);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_grad_v_distributed);

hpx::future<double> compute_loss_distributed(const tile_handle<double> &K_diag_tile,
                                             const tile_handle<double> &alpha_tile,
                                             const tile_handle<double> &y_tile,
                                             std::size_t N);
HPX_DEFINE_PLAIN_DIRECT_ACTION(compute_loss_distributed);

hpx::future<double> compute_trace_distributed(const tile_handle<double> &diagonal, double trace);
HPX_DEFINE_PLAIN_DIRECT_ACTION(compute_trace_distributed);

hpx::future<double>
compute_dot_distributed(const tile_handle<double> &vector_T, const tile_handle<double> &vector, double result);
HPX_DEFINE_PLAIN_DIRECT_ACTION(compute_dot_distributed);

hpx::future<double> compute_trace_diag_distributed(const tile_handle<double> &tile, double trace, std::size_t N);
HPX_DEFINE_PLAIN_DIRECT_ACTION(compute_trace_diag_distributed);

}  // namespace cpu

GPRAT_NS_END

GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_covariance_with_distance,
                               GPRAT_NS::cpu::gen_tile_covariance_with_distance_distributed_action,
                               "gen_tile_covariance_with_distance");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_grad_l,
                               GPRAT_NS::cpu::gen_tile_grad_l_distributed_action,
                               "gen_tile_grad_l");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_grad_v,
                               GPRAT_NS::cpu::gen_tile_grad_v_distributed_action,
                               "gen_tile_grad_v");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::compute_loss,
                               GPRAT_NS::cpu::compute_loss_distributed_action,
                               "compute_loss");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::compute_trace,
                               GPRAT_NS::cpu::compute_trace_distributed_action,
                               "compute_trace");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::compute_dot,
                               GPRAT_NS::cpu::compute_dot_distributed_action,
                               "compute_dot");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::compute_trace_diag,
                               GPRAT_NS::cpu::compute_trace_diag_distributed_action,
                               "compute_trace_diag");

#endif
