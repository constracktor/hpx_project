#include "gprat/cpu/gp_algorithms_actions.hpp"

#include <hpx/include/performance_counters.hpp>

GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_covariance,
                              GPRAT_NS::cpu::gen_tile_covariance_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_prior_covariance,
                              GPRAT_NS::cpu::gen_tile_prior_covariance_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_prior_covariance,
                              GPRAT_NS::cpu::gen_tile_full_prior_covariance_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_cross_covariance,
                              GPRAT_NS::cpu::gen_tile_cross_covariance_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_transpose, GPRAT_NS::cpu::gen_tile_transpose_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_output, GPRAT_NS::cpu::gen_tile_output_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_zeros, GPRAT_NS::cpu::gen_tile_zeros_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_identity, GPRAT_NS::cpu::gen_tile_identity_distributed_action)

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
    const std::vector<double> &input)
{
    return tile.set_async(cpu::gen_tile_covariance(row, col, N, n_regressors, sek_params, input));
}

hpx::future<tile_handle<double>> gen_tile_prior_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input)
{
    return tile.set_async(cpu::gen_tile_prior_covariance(row, col, N, n_regressors, sek_params, input));
}

hpx::future<tile_handle<double>> gen_tile_full_prior_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input)
{
    return tile.set_async(cpu::gen_tile_full_prior_covariance(row, col, N, n_regressors, sek_params, input));
}

hpx::future<tile_handle<double>> gen_tile_cross_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N_row,
    std::size_t N_col,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &row_input,
    const std::vector<double> &col_input)
{
    return tile.set_async(
        cpu::gen_tile_cross_covariance(row, col, N_row, N_col, n_regressors, sek_params, row_input, col_input));
}

hpx::future<tile_handle<double>> gen_tile_transpose_distributed(
    const tile_handle<double> &tile, std::size_t N_row, std::size_t N_col, const tile_handle<double> &src)
{
    return hpx::dataflow(
        hpx::launch::async,
        [=](hpx::future<mutable_tile_data<double>> &&tiled)
        { return tile.set_async(cpu::gen_tile_transpose(N_row, N_col, tiled.get().as_span())); },
        src.get_async());
}

hpx::future<tile_handle<double>> gen_tile_output_distributed(
    const tile_handle<double> &tile, std::size_t row, std::size_t N, const std::vector<double> &output)
{
    return tile.set_async(cpu::gen_tile_output(row, N, output));
}

hpx::future<tile_handle<double>> gen_tile_zeros_distributed(const tile_handle<double> &tile, std::size_t N)
{
    return tile.set_async(cpu::gen_tile_zeros(N));
}

hpx::future<tile_handle<double>> gen_tile_identity_distributed(const tile_handle<double> &tile, std::size_t N)
{
    return tile.set_async(cpu::gen_tile_identity(N));
}
}  // namespace cpu

GPRAT_NS_END
