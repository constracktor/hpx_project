#include "gprat/cpu/gp_uncertainty_actions.hpp"

#include <hpx/include/performance_counters.hpp>

GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::get_matrix_diagonal,
                              GPRAT_NS::cpu::get_matrix_diagonal_distributed_action)

GPRAT_NS_BEGIN

namespace cpu
{
hpx::future<tile_handle<double>> get_matrix_diagonal_distributed(const tile_handle<double> &A, std::size_t M)
{
    return hpx::dataflow(
        hpx::launch::async,
        [A, M](hpx::future<mutable_tile_data<double>> &&Ad)
        { return A.set_async(cpu::get_matrix_diagonal(Ad.get(), M)); },
        A.get_async());
}

}  // namespace cpu

GPRAT_NS_END
