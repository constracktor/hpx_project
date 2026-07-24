#include "gprat/cpu/gp_optimizer_actions.hpp"

#include <hpx/include/performance_counters.hpp>

GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_covariance_with_distance,
                              GPRAT_NS::cpu::gen_tile_covariance_with_distance_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_grad_l, GPRAT_NS::cpu::gen_tile_grad_l_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::gen_tile_grad_v, GPRAT_NS::cpu::gen_tile_grad_v_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::compute_loss, GPRAT_NS::cpu::compute_loss_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::compute_trace, GPRAT_NS::cpu::compute_trace_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::compute_dot, GPRAT_NS::cpu::compute_dot_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::compute_trace_diag, GPRAT_NS::cpu::compute_trace_diag_distributed_action)

GPRAT_NS_BEGIN

namespace cpu
{

hpx::future<tile_handle<double>> gen_tile_covariance_with_distance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    const SEKParams &sek_params,
    const const_tile_data<double> &distance)
{
    return tile.set_async(cpu::gen_tile_covariance_with_distance(row, col, N, sek_params, distance));
}

hpx::future<tile_handle<double>> gen_tile_grad_l_distributed(
    const tile_handle<double> &tile,
    std::size_t N,
    const SEKParams &sek_params,
    const const_tile_data<double> &distance)
{
    return tile.set_async(cpu::gen_tile_grad_l(N, sek_params, distance));
}

hpx::future<tile_handle<double>> gen_tile_grad_v_distributed(
    const tile_handle<double> &tile,
    std::size_t N,
    const SEKParams &sek_params,
    const const_tile_data<double> &distance)
{
    return tile.set_async(cpu::gen_tile_grad_v(N, sek_params, distance));
}

hpx::future<double> compute_loss_distributed(const tile_handle<double> &K_diag_tile,
                                             const tile_handle<double> &alpha_tile,
                                             const tile_handle<double> &y_tile,
                                             std::size_t N)
{
    return hpx::dataflow(
        hpx::launch::async,
        [=](hpx::future<mutable_tile_data<double>> &&K_diag_tiled,
            hpx::future<mutable_tile_data<double>> &&alpha_tiled,
            hpx::future<mutable_tile_data<double>> &&y_tiled) {
            return cpu::compute_loss(
                K_diag_tiled.get().as_span(), alpha_tiled.get().as_span(), y_tiled.get().as_span(), N);
        },
        K_diag_tile.get_async(),
        alpha_tile.get_async(),
        y_tile.get_async());
}

hpx::future<double> compute_trace_distributed(const tile_handle<double> &diagonal, double trace)
{
    return hpx::dataflow(
        hpx::launch::async,
        [=](hpx::future<mutable_tile_data<double>> &&diagonald)
        { return cpu::compute_trace(diagonald.get().as_span(), trace); },
        diagonal.get_async());
}

hpx::future<double>
compute_dot_distributed(const tile_handle<double> &vector_T, const tile_handle<double> &vector, double result)
{
    return hpx::dataflow(
        hpx::launch::async,
        [=](hpx::future<mutable_tile_data<double>> &&vector_Td, hpx::future<mutable_tile_data<double>> &&vectord)
        { return cpu::compute_dot(vector_Td.get().as_span(), vectord.get().as_span(), result); },
        vector_T.get_async(),
        vector.get_async());
}

hpx::future<double> compute_trace_diag_distributed(const tile_handle<double> &tile, double trace, std::size_t N)
{
    return hpx::dataflow(
        hpx::launch::async,
        [=](hpx::future<mutable_tile_data<double>> &&tiled)
        { return cpu::compute_trace_diag(tiled.get().as_span(), trace, N); },
        tile.get_async());
}

}  // namespace cpu

GPRAT_NS_END
