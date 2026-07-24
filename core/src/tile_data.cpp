#include "gprat/tile_data.hpp"

#include "gprat/performance_counters.hpp"

#include <hpx/runtime_local/runtime_local.hpp>

GPRAT_NS_BEGIN

namespace detail
{

void *allocate_tile_data(std::size_t num_bytes)
{
    auto &topology = hpx::get_runtime().get_topology();
    const auto bitmap = topology.cpuset_to_nodeset(topology.get_machine_affinity_mask());

    track_tile_data_allocation(num_bytes);
    return topology.allocate_membind(num_bytes, bitmap, hpx::threads::hpx_hwloc_membind_policy::membind_firsttouch, 0);
}

void deallocate_tile_data(void *p, std::size_t num_bytes)
{
    track_tile_data_deallocation(num_bytes);

    if (hpx::is_running())
    {
        auto &topology = hpx::get_runtime().get_topology();
        topology.deallocate(p, num_bytes);
    }
}

}  // namespace detail

GPRAT_NS_END
