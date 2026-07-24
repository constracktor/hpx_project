#ifndef GPRAT_COMPONENTS_TILED_DATASET_HPP
#define GPRAT_COMPONENTS_TILED_DATASET_HPP

#pragma once

#include "gprat/detail/actions.hpp"
#include "gprat/detail/config.hpp"
#include "gprat/performance_counters.hpp"
#include "gprat/tile_cache.hpp"
#include "gprat/tile_data.hpp"

#include <hpx/modules/actions.hpp>
#include <hpx/modules/actions_base.hpp>
#include <hpx/modules/components.hpp>
#include <hpx/modules/components_base.hpp>
#include <hpx/modules/runtime_components.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <span>
#include <utility>

GPRAT_NS_BEGIN

namespace server
{

/**
 * Server component owning a single tile's data.
 *
 * @tparam T Element type of the tile. Usually some numeric type like double or float. This class currently only
 * requires T to be serializable by HPX.
 */
template <typename T>
struct tile_holder : hpx::components::component_base<tile_holder<T>>
{
    tile_holder() { track_tile_server_allocation(0); }

    explicit tile_holder(const mutable_tile_data<T> &data) :
        data_(data)
    {
        track_tile_server_allocation(data.size());
    }

    ~tile_holder() { track_tile_server_deallocation(data_.size()); }

    [[nodiscard]] mutable_tile_data<T> get_data() const
    {
        std::shared_lock lock(mutex_);
        return data_;
    }

    void set_data(const mutable_tile_data<T> &data)
    {
        std::unique_lock lock(mutex_);
        data_ = data;
    }

    // Every member function that has to be invoked remotely needs to be
    // wrapped into a component action.
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(tile_holder, get_data)
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(tile_holder, set_data)

  private:
    mutable hpx::shared_mutex mutex_;
    mutable_tile_data<T> data_;
};

template <typename T>
struct tile_manager_shared_data
{
    struct tile_entry
    {
        tile_entry() :
            locality_id(hpx::naming::invalid_locality_id)
        { }

        tile_entry(hpx::id_type tile, std::uint32_t in_locality_id) :
            id(std::move(tile)),
            locality_id(in_locality_id)
        { }

        hpx::id_type id;
        std::uint32_t locality_id;
        std::shared_ptr<tile_holder<T>> local_data;

      private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive &ar, unsigned)
        {
            ar & id & locality_id;
        }
    };

    std::vector<tile_entry> tiles;

  private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & tiles;
    }
};

template <typename T>
struct tile_manager : hpx::components::component_base<tile_manager<T>>
{
    explicit tile_manager(tile_manager_shared_data<T> &&data) :
        data_(std::move(data))
    {
        const auto here = hpx::get_locality_id();
        for (auto &tile : data_.tiles)
        {
            if (tile.locality_id == here)
            {
                tile.local_data = hpx::get_ptr<tile_holder<T>>(hpx::launch::sync, tile.id);
            }
        }
    }

    mutable_tile_data<T> get_tile_data(std::size_t tile_index, std::size_t generation)
    {
        const auto &target_tile = data_.tiles[tile_index];

        // Best is always to rely on local data
        if (target_tile.local_data)
        {
            return target_tile.local_data->get_data();
        }

        // Next, try the tile cache - maybe we have current data
        {
            mutable_tile_data<T> cached_data;
            if (cache_.try_get(target_tile.id.get_gid(), generation, cached_data))
            {
                return cached_data;
            }
        }

        hpx::chrono::high_resolution_timer timer;
        auto data = hpx::async(typename tile_holder<T>::get_data_action{}, target_tile.id).get();

        record_transmission_time(timer.elapsed_nanoseconds());
        cache_.insert(target_tile.id.get_gid(), generation, data);

        return data;
    }

    hpx::future<mutable_tile_data<T>> get_tile_data_async(std::size_t tile_index, std::size_t generation)
    {
        const auto &target_tile = data_.tiles[tile_index];

        // Best is always to rely on local data
        if (target_tile.local_data)
        {
            return hpx::make_ready_future(target_tile.local_data->get_data());
        }

        // Next, try the tile cache - maybe we have current data
        {
            mutable_tile_data<T> cached_data;
            if (cache_.try_get(target_tile.id.get_gid(), generation, cached_data))
            {
                return hpx::make_ready_future(cached_data);
            }
        }

        return hpx::async(typename tile_holder<T>::get_data_action{}, target_tile.id)
            .then(
                [this,
                 self = this->get_id(),
                 generation,
                 gid = target_tile.id.get_gid(),
                 timer = hpx::chrono::high_resolution_timer()](hpx::future<mutable_tile_data<T>> &&f) mutable
                {
                    record_transmission_time(timer.elapsed_nanoseconds());
                    auto data = f.get();
                    cache_.insert(gid, generation, data);
                    self = {};  // release our reference
                    return data;
                });
    }

    hpx::future<void>
    set_tile_data_async(std::size_t tile_index, std::size_t generation, const mutable_tile_data<T> &data)
    {
        const auto &target_tile = data_.tiles[tile_index];

        if (target_tile.local_data)
        {
            target_tile.local_data->set_data(data);
            return hpx::make_ready_future();
        }

        // Insert into cache only after the remote write confirms success; inserting
        // before the write would leave a stale cache entry if the remote call fails.
        return hpx::async(typename tile_holder<T>::set_data_action{}, target_tile.id, data)
            .then(
                [this, self = this->get_id(), gid = target_tile.id.get_gid(), generation, data](
                    hpx::future<void> &&f) mutable
                {
                    f.get();  // rethrow any remote exception
                    cache_.insert(gid, generation, data);
                    self = {};  // release our reference
                });
    }

  private:
    tile_manager_shared_data<T> data_;
    tile_cache<T> cache_;
};

}  // namespace server

// DECLARATION macros (use in a single header)

#define GPRAT_REGISTER_TILE_HOLDER_DECLARATION_IMPL(type, name)                                                        \
    HPX_REGISTER_ACTION_DECLARATION(type::get_data_action, HPX_PP_CAT(_tile_holder_get_data_action_, name))            \
    HPX_REGISTER_ACTION_DECLARATION(type::set_data_action, HPX_PP_CAT(_tile_holder_set_data_action_, name))            \
    /**/

#define GPRAT_REGISTER_TILED_DATASET_DECLARATION(type, name)                                                           \
    typedef ::GPRAT_NS::server::tile_holder<type> HPX_PP_CAT(_server_tile_holder_, HPX_PP_CAT(type, name));            \
    GPRAT_REGISTER_TILE_HOLDER_DECLARATION_IMPL(HPX_PP_CAT(_server_tile_holder_, HPX_PP_CAT(type, name)), name)

// REGISTRATION macros (use in a single .cpp file)

#define GPRAT_REGISTER_TILE_HOLDER_IMPL(type, name)                                                                    \
    HPX_REGISTER_ACTION(type::get_data_action, HPX_PP_CAT(_tile_holder_get_data_action_, name))                        \
    HPX_REGISTER_ACTION(type::set_data_action, HPX_PP_CAT(_tile_holder_set_data_action_, name))                        \
    typedef ::hpx::components::component<type> HPX_PP_CAT(_server_tile_holder_component_, name);                       \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(_server_tile_holder_component_, name))                                           \
    /**/

#define GPRAT_REGISTER_TILE_MANAGER_IMPL(type, name)                                                                   \
    typedef ::hpx::components::component<type> HPX_PP_CAT(_server_tile_manager_component_, name);                      \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(_server_tile_manager_component_, name))                                          \
    /**/

#define GPRAT_REGISTER_TILED_DATASET(type, name)                                                                       \
    typedef ::GPRAT_NS::server::tile_holder<type> HPX_PP_CAT(_server_tile_holder_, HPX_PP_CAT(type, name));            \
    GPRAT_REGISTER_TILE_HOLDER_IMPL(HPX_PP_CAT(_server_tile_holder_, HPX_PP_CAT(type, name)), name)                    \
    typedef ::GPRAT_NS::server::tile_manager<type> HPX_PP_CAT(_server_tile_manager_, HPX_PP_CAT(type, name));          \
    GPRAT_REGISTER_TILE_MANAGER_IMPL(HPX_PP_CAT(_server_tile_manager_, HPX_PP_CAT(type, name)), name)

template <typename T>
class tile_handle
{
  public:
    tile_handle() = default;

    tile_handle(std::vector<hpx::id_type> managers, std::size_t tile_index, std::size_t generation) :
        managers_(std::make_shared<std::vector<hpx::id_type>>(std::move(managers))),
        tile_index_(tile_index),
        generation_(generation)
    { }

    tile_handle(std::shared_ptr<std::vector<hpx::id_type>> managers, std::size_t tile_index, std::size_t generation) :
        managers_(std::move(managers)),
        tile_index_(tile_index),
        generation_(generation)
    { }

    // ReSharper disable once CppNonExplicitConversionOperator
    operator mutable_tile_data<T>() const { return get(); }  // NOLINT(*-explicit-constructor)

    mutable_tile_data<T> get() const { return local_manager()->get_tile_data(tile_index_, generation_); }

    hpx::future<mutable_tile_data<T>> get_async() const
    {
        return local_manager()->get_tile_data_async(tile_index_, generation_);
    }

    // Returns a new tile_handle with the incremented generation once the write completes.
    // Callers MUST use the returned handle for subsequent reads; the original handle's
    // generation_ is not updated.
    [[nodiscard]] hpx::future<tile_handle> set_async(const mutable_tile_data<T> &data) const
    {
        return local_manager()
            ->set_tile_data_async(tile_index_, generation_ + 1, data)
            .then(
                [self = *this](hpx::future<void> &&) mutable
                {
                    ++self.generation_;
                    return self;
                });
    }

  private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        // Serialize the vector contents, not the shared_ptr itself.
        // cached_manager_ is a runtime cache and is not serialized.
        ar &*managers_ &tile_index_ & generation_;
    }

    std::shared_ptr<server::tile_manager<T>> local_manager() const
    {
        if (cached_manager_)
        {
            return cached_manager_;
        }
        const auto here = hpx::get_locality_id();
        for (const auto &id : *managers_)
        {
            if (here == hpx::naming::get_locality_id_from_id(id))
            {
                cached_manager_ = hpx::get_ptr<server::tile_manager<T>>(hpx::launch::sync, id);
                return cached_manager_;
            }
        }
        throw std::runtime_error("This locality is not known");
    }

    std::shared_ptr<std::vector<hpx::id_type>> managers_ = std::make_shared<std::vector<hpx::id_type>>();
    std::size_t tile_index_ = 0;
    std::size_t generation_ = 0;
    mutable std::shared_ptr<server::tile_manager<T>> cached_manager_;
};

template <typename T>
class tiled_dataset
{
  public:
    using value_type = hpx::shared_future<tile_handle<T>>;

    tiled_dataset() = default;

    explicit tiled_dataset(std::size_t size) :
        data_(std::make_unique<value_type[]>(size)),
        size_(size)
    { }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    const value_type *data() const noexcept { return data_.get(); }

    const value_type *begin() const noexcept { return data_.get(); }

    const value_type *end() const noexcept { return data_.get() + size_; }

    value_type &operator[](std::size_t i)
    {
        if (i >= size_)
        {
            throw std::out_of_range("tiled_dataset::operator[]");
        }
        return data_[i];
    }

    const value_type &operator[](std::size_t i) const
    {
        if (i >= size_)
        {
            throw std::out_of_range("tiled_dataset::operator[]");
        }
        return data_[i];
    }

  private:
    std::unique_ptr<value_type[]> data_;
    std::size_t size_ = 0;
};

template <typename T>
tiled_dataset<T>
create_tiled_dataset(std::span<const std::pair<hpx::id_type, std::size_t>> targets, std::size_t num_tiles)
{
    // First, create the actual tile data holders
    std::vector<hpx::future<std::vector<hpx::id_type>>> holders;
    holders.reserve(targets.size());
    for (const auto &target : targets)
    {
#if (HPX_VERSION_FULL >= 0x011100)
        holders.emplace_back(
            hpx::components::bulk_create_async<false, server::tile_holder<T>>(target.first, target.second));
#else
        holders.emplace_back(hpx::components::bulk_create_async<server::tile_holder<T>>(target.first, target.second));
#endif
    }

    // Next, we prepare our shared data for the manager components
    server::tile_manager_shared_data<T> manager_data;
    manager_data.tiles.reserve(num_tiles);

    for (std::size_t i = 0; i < targets.size() && manager_data.tiles.size() < num_tiles; ++i)
    {
        const auto locality = hpx::naming::get_locality_id_from_id(targets[i].first);
        for (hpx::id_type &id : holders[i].get())
        {
            manager_data.tiles.emplace_back(std::move(id), locality);
            if (manager_data.tiles.size() == num_tiles)
            {
                break;
            }
        }
    }

    if (manager_data.tiles.size() != num_tiles)
    {
        throw std::runtime_error(
            "create_tiled_dataset: targets provided fewer slots (" + std::to_string(manager_data.tiles.size())
            + ") than num_tiles (" + std::to_string(num_tiles) + ")");
    }

    // Now we move on to the manager components
    std::vector<hpx::id_type> managers;
    managers.reserve(targets.size());
    for (const auto &target : targets)
    {
        managers.emplace_back(hpx::components::create<server::tile_manager<T>>(target.first, manager_data));
    }

    // Finally, we create our fat tile_handles — all sharing one managers vector.
    auto shared_managers = std::make_shared<std::vector<hpx::id_type>>(std::move(managers));
    tiled_dataset<T> tiles(num_tiles);
    for (std::size_t i = 0; i < num_tiles; ++i)
    {
        tiles[i] = hpx::make_ready_future(tile_handle<T>{ shared_managers, i, 0 });
    }
    return tiles;
}

template <typename T, typename Mapper>
tiled_dataset<T> make_tiled_dataset(const tiled_scheduler_distributed &sched, std::size_t num_tiles, Mapper &&mapper)
{
    const auto num_localities = sched.localities_.size();
    std::vector<std::pair<hpx::id_type, std::size_t>> targets;
    targets.reserve(num_localities);

    for (std::size_t i = 0; i < num_localities; ++i)
    {
        targets.emplace_back(sched.localities_[i], 0);
    }

    for (std::size_t i = 0; i < num_tiles; i++)
    {
        ++targets[mapper(i) % num_localities].second;
    }

    return create_tiled_dataset<T>(targets, num_tiles);
}

GPRAT_NS_END

// Register the double version by default
// Users can register custom types in the same way
GPRAT_REGISTER_TILED_DATASET_DECLARATION(double, double)

#endif
