#ifndef GPRAT_DETAIL_DATAFLOW_HELPERS_HPP
#define GPRAT_DETAIL_DATAFLOW_HELPERS_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include <hpx/async_base/async.hpp>
#include <hpx/async_base/dataflow.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/threading_base/annotated_function.hpp>

GPRAT_NS_BEGIN

/// @brief Empty type representing local scheduling (always on this locality)
struct basic_local_scheduler
{ };

namespace detail
{

// Functions prefixed with named_* allow the user to specify a custom name for this entry in the
// execution graph. Much like wrapping your function with hpx::annotated_function would.

// =============================================================
// non-scheduler aware

template <auto F, typename... Args>
decltype(auto) named_dataflow(const char *name, Args &&...args)
{
    return hpx::dataflow(hpx::annotated_function(hpx::unwrapping(F), name), std::forward<Args>(args)...);
}

template <auto F, typename... Args>
decltype(auto) named_async(const char *name, Args &&...args)
{
    return hpx::async(hpx::annotated_function(F, name), std::forward<Args>(args)...);
}

// =============================================================
// local shared-memory scheduling
// (no-op, same as above)

template <auto F, typename TileReference, typename... Args>
decltype(auto) named_make_tile(const basic_local_scheduler & /*sched*/,
                               std::size_t /*on*/,
                               const char *name,
                               TileReference & /*target*/,
                               Args &&...args)
{
    // This method basically ignores the reference to the target tile as the non-action factories don't need it.
    // (They always create the tile_data locally and return that - only the HPX action wrappers need a reference)
    return hpx::dataflow(hpx::annotated_function(hpx::unwrapping(F), name), std::forward<Args>(args)...);
}

template <auto F, typename... Args>
decltype(auto)
named_dataflow(const basic_local_scheduler & /*sched*/, std::size_t /*on*/, const char *name, Args &&...args)
{
    return hpx::dataflow(hpx::annotated_function(hpx::unwrapping(F), name), std::forward<Args>(args)...);
}

template <auto F, typename... Args>
decltype(auto)
named_async(const basic_local_scheduler & /*sched*/, std::size_t /*on*/, const char *name, Args &&...args)
{
    return hpx::async(hpx::annotated_function(F, name), std::forward<Args>(args)...);
}

}  // namespace detail

GPRAT_NS_END

#endif
