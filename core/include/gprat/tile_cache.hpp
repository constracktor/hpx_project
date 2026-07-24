#ifndef GPRAT_TILECACHE_HPP
#define GPRAT_TILECACHE_HPP

#pragma once

#include "gprat/tile_data.hpp"

#include <hpx/modules/cache.hpp>
#include <hpx/modules/components_base.hpp>

GPRAT_NS_BEGIN

namespace detail
{
hpx::util::cache::statistics::local_full_statistics &get_global_statistics();

/// @brief Statistics implementation that uses counters shared between all tile_cache instances
class global_full_statistics
{
  public:
    using update_on_exit = hpx::util::cache::statistics::local_full_statistics::update_on_exit;

    // ReSharper disable once CppNonExplicitConversionOperator
    operator hpx::util::cache::statistics::local_full_statistics &() const { return get_global_statistics(); }

    void got_hit() noexcept { get_global_statistics().got_hit(); }

    void got_miss() noexcept { get_global_statistics().got_miss(); }

    void got_insertion() noexcept { get_global_statistics().got_insertion(); }

    void got_eviction() noexcept { get_global_statistics().got_eviction(); }

    void clear() noexcept { get_global_statistics().clear(); }
};
}  // namespace detail

/**
 * @brief LRU cache for mutable_tile_data objects with versioning support
 * @tparam T Tile data type.
 */
template <typename T>
class tile_cache
{
    friend struct tile_cache_counters;

  public:
    explicit tile_cache(std::size_t max_size = 16) :
        cache_(max_size)
    { }

    bool try_get(const hpx::naming::gid_type &key, std::size_t generation, mutable_tile_data<T> &cached_data)
    {
        std::lock_guard g(mutex_);

        entry e;
        {
            hpx::naming::gid_type unused;
            if (!cache_.get_entry(key, unused, e))
            {
                return false;
            }
        }

        if (e.generation == generation)
        {
            cached_data = e.data;
            return true;
        }

        // Erase the obsolete entry
        cache_.erase([&](const auto &p) { return p.first == key; });
        return false;
    }

    void insert(const hpx::naming::gid_type &key, std::size_t generation, const mutable_tile_data<T> &data)
    {
        std::lock_guard g(mutex_);
        cache_.insert(key, entry{ data, generation });
    }

    void clear() { cache_.clear(); }

  private:
    struct entry
    {
        mutable_tile_data<T> data;
        std::size_t generation = 0;
    };

    hpx::mutex mutex_;  // lru_cache is not thread-safe!
    hpx::util::cache::lru_cache<hpx::naming::gid_type, entry, detail::global_full_statistics> cache_;
};

GPRAT_NS_END

#endif
