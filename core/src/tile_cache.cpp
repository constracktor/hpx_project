#include "gprat/tile_cache.hpp"

GPRAT_NS_BEGIN

namespace detail
{
hpx::util::cache::statistics::local_full_statistics &get_global_statistics()
{
    static hpx::util::cache::statistics::local_full_statistics stats;
    return stats;
}

}  // namespace detail

GPRAT_NS_END
