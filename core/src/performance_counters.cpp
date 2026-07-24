#include "gprat/performance_counters.hpp"

#include "gprat/detail/config.hpp"
#if GPRAT_WITH_DISTRIBUTED
#include "gprat/tile_cache.hpp"
#endif

#include <atomic>
#include <emmintrin.h>
#include <hpx/util/get_and_reset_value.hpp>
#ifdef HPX_HAVE_MODULE_PERFORMANCE_COUNTERS
#include <hpx/performance_counters/manage_counter_type.hpp>
#endif

GPRAT_NS_BEGIN

#define GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(name)                                                                       \
    static std::atomic<std::uint64_t> name(0);                                                                         \
    std::uint64_t get_##name(bool reset) { return hpx::util::get_and_reset_value(name, reset); }

GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_data_allocations)
GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_data_deallocations)
GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_server_allocations)
GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_server_deallocations)
GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_transmission_time)
GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_transmission_count)

#undef GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR

void track_tile_data_allocation(std::size_t /*size*/) { tile_data_allocations += 1; }

void track_tile_data_deallocation(std::size_t /*size*/) { tile_data_deallocations += 1; }

void track_tile_server_allocation(std::size_t /*size*/) { tile_server_allocations += 1; }

void track_tile_server_deallocation(std::size_t /*size*/) { tile_server_deallocations += 1; }

void record_transmission_time(std::int64_t elapsed_ns)
{
    HPX_ASSERT(elapsed_ns >= 0);
    tile_transmission_count += 1;
    if (elapsed_ns > 0)
    {
        tile_transmission_time += static_cast<std::uint64_t>(elapsed_ns);
    }
}

#ifdef HPX_HAVE_MODULE_PERFORMANCE_COUNTERS
// These are non-public functions of their respective CUs.
namespace detail
{
void register_fp32_performance_counters();
void register_fp64_performance_counters();
}  // namespace detail

void register_performance_counters()
{
    // XXX: you can do this with templates, but it's quite a bit more complicated
#define GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(name, stats_expr)                                                           \
    hpx::performance_counters::install_counter_type(                                                                   \
        name,                                                                                                          \
        [](bool reset) { return hpx::util::get_and_reset_value(stats_expr, reset); },                                  \
        #stats_expr,                                                                                                   \
        "",                                                                                                            \
        hpx::performance_counters::counter_type::monotonically_increasing)

    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/tile_data/num_allocations", tile_data_allocations);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/tile_data/num_deallocations", tile_data_deallocations);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/tile_server/num_allocations", tile_server_allocations);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/tile_server/num_deallocations", tile_server_deallocations);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/tile_cache/transmission_time", tile_transmission_time);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/tile_cache/transmission_count", tile_transmission_count);

#undef GPRAT_MAKE_STATISTICS_ACCESSOR

    // XXX: you can do this with templates, but it's quite a bit more complicated
#define GPRAT_MAKE_STATISTICS_ACCESSOR(name, stats_expr)                                                               \
    hpx::performance_counters::install_counter_type(                                                                   \
        name,                                                                                                          \
        [](bool reset) { return (stats_expr) (reset); },                                                               \
        #stats_expr,                                                                                                   \
        "",                                                                                                            \
        hpx::performance_counters::counter_type::monotonically_increasing)

#if GPRAT_WITH_DISTRIBUTED
    GPRAT_MAKE_STATISTICS_ACCESSOR("/gprat/tile_cache/hits", detail::get_global_statistics().hits);
    GPRAT_MAKE_STATISTICS_ACCESSOR("/gprat/tile_cache/misses", detail::get_global_statistics().misses);
    GPRAT_MAKE_STATISTICS_ACCESSOR("/gprat/tile_cache/evictions", detail::get_global_statistics().evictions);
    GPRAT_MAKE_STATISTICS_ACCESSOR("/gprat/tile_cache/insertions", detail::get_global_statistics().insertions);
#endif

#undef GPRAT_MAKE_STATISTICS_ACCESSOR

    detail::register_fp32_performance_counters();
    detail::register_fp64_performance_counters();
}

#else
void register_performance_counters()
{
    // no-op for binary compatibility
}
#endif

void force_evict_memory(const void *start, std::size_t size)
{
    // A cache line size of 64 seems to be a safe estimate.
    // see: https://lemire.me/blog/2023/12/12/measuring-the-size-of-the-cache-line-empirically/
    constexpr std::size_t cache_line_size = 64;

    const char *p = static_cast<const char *>(start);
    const char *end = p + size;

    _mm_mfence();
    do {
        // Intel recommends clflushopt over normal clflush due to higher performance, see:
        // http://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf
        _mm_clflush(p);
        p += cache_line_size;
    } while (p < end);

    // Make sure we don't miss a cache line at the end
    if ((reinterpret_cast<std::uintptr_t>(p) & (cache_line_size - 1))
        != (reinterpret_cast<std::uintptr_t>(end - 1) & (cache_line_size - 1)))
    {
        _mm_clflush(end - 1);
    }
    _mm_mfence();
}

GPRAT_NS_END
