#ifndef GPRAT_CPU_GP_ALGORITHMS_ACTIONS_HPP
#define GPRAT_CPU_GP_ALGORITHMS_ACTIONS_HPP

#pragma once

#include "gprat/cpu/gp_algorithms.hpp"
#include "gprat/detail/actions.hpp"
#include "gprat/detail/config.hpp"
#include "gprat/tiled_dataset.hpp"

GPRAT_NS_BEGIN

namespace cpu
{

hpx::future<tile_handle<double>> gen_tile_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_covariance_distributed);

hpx::future<tile_handle<double>> gen_tile_prior_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_prior_covariance_distributed);

hpx::future<tile_handle<double>> gen_tile_full_prior_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_full_prior_covariance_distributed);

hpx::future<tile_handle<double>> gen_tile_cross_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N_row,
    std::size_t N_col,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &row_input,
    const std::vector<double> &col_input);

HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_cross_covariance_distributed);

hpx::future<tile_handle<double>> gen_tile_transpose_distributed(
    const tile_handle<double> &tile, std::size_t N_row, std::size_t N_col, const tile_handle<double> &src);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_transpose_distributed);

hpx::future<tile_handle<double>> gen_tile_output_distributed(
    const tile_handle<double> &tile, std::size_t row, std::size_t N, const std::vector<double> &output);

HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_output_distributed);

hpx::future<tile_handle<double>> gen_tile_zeros_distributed(const tile_handle<double> &tile, std::size_t N);

HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_zeros_distributed);

hpx::future<tile_handle<double>> gen_tile_identity_distributed(const tile_handle<double> &tile, std::size_t N);

HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_identity_distributed);
}  // namespace cpu

GPRAT_NS_END

GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_covariance,
                               GPRAT_NS::cpu::gen_tile_covariance_distributed_action,
                               "cpu::gen_tile_covariance");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_prior_covariance,
                               GPRAT_NS::cpu::gen_tile_prior_covariance_distributed_action,
                               "gen_tile_prior_covariance");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_full_prior_covariance,
                               GPRAT_NS::cpu::gen_tile_full_prior_covariance_distributed_action,
                               "gen_tile_full_prior_covariance");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_cross_covariance,
                               GPRAT_NS::cpu::gen_tile_cross_covariance_distributed_action,
                               "gen_tile_cross_covariance");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_transpose,
                               GPRAT_NS::cpu::gen_tile_transpose_distributed_action,
                               "gen_tile_transpose");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_output,
                               GPRAT_NS::cpu::gen_tile_output_distributed_action,
                               "gen_tile_output");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_zeros,
                               GPRAT_NS::cpu::gen_tile_zeros_distributed_action,
                               "gen_tile_zeros");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_identity,
                               GPRAT_NS::cpu::gen_tile_identity_distributed_action,
                               "gen_tile_identity");

#endif
