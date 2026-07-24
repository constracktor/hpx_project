#ifndef GPRAT_TARGET_H
#define GPRAT_TARGET_H

#pragma once

#include "gprat/detail/config.hpp"

#include <atomic>
#include <string>

#if GPRAT_WITH_CUDA
#include <cuda_runtime.h>
#include <hpx/async_cuda/cublas_executor.hpp>
#endif

#if GPRAT_WITH_SYCL
#include <sycl/sycl.hpp>
#endif

GPRAT_NS_BEGIN

struct DeviceParameters
{
    std::size_t id;
    std::size_t n_queues;
};

/**
 * @brief This class represents the target on which to perform the Gaussian
 *        Process computations: either CPU or GPU.
 *
 * The respective subclasses implement specific targets: CPU, CUDA_GPU, SYCL_DEVICE.
 * They may also set additional attributes or function that are required when
 * using this target.
 */
struct Target
{
    /**
     * @brief Returns true if target is CPU.
     *
     * Implemented by subclasses.
     */
    virtual bool is_cpu() = 0;

    /**
     * @brief Returns true if target is a CUDA GPU.
     *
     * Implemented by subclasses.
     */
    virtual bool is_gpu() = 0;

    /**
     * @brief Returns true if target is a SYCL device.
     *
     * Implemented by subclasses.
     */
    virtual bool is_sycl() = 0;

    /**
     * @brief Returns string representation of the target.
     *
     * Implemented by subclasses.
     */
    virtual std::string repr() const = 0;

    virtual ~Target() { }

  protected:
    Target() = default;
};

// CPU ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CPU : public Target
{
  public:
    /**
     * @brief Returns CPU target.
     */
    CPU();

    /**
     * @brief Returns true because target is CPU.
     */
    bool is_cpu() override;

    /**
     * @brief Returns false because CPU target is not a CUDA GPU.
     */
    bool is_gpu() override;

    /**
     * @brief Returns false because CPU target is not a SYCL device.
     */
    bool is_sycl() override;

    /**
     * @brief Returns string representation of the CPU target.
     */
    std::string repr() const override;
};

/**
 * @brief Creates and returns handle for CPU target.
 *
 * @return CPU target
 */
CPU get_cpu();

// CUDA GPU ///////////////////////////////////////////////////////////////////////////////////////////////////////////

#if GPRAT_WITH_CUDA
struct CUDA_GPU : public Target
{
    /**
     * @brief Identifier of GPU device.
     *
     * Can be set to a value between 0 and gpu_count().
     */
    int id;

    /**
     * @brief Number of CUDA streams used asynchronous computation and data
     *        transfer.
     */
    int n_streams;

    /**
     * @brief Index of next CUDA stream assigned on next_stream() or
     *        next_cublas_handle().
     */
    std::atomic<int> i_stream;

    /** @brief Default amount of CUDA shared memory used by CUDA kernels. */
    int shared_memory_size;

    /**
     * @brief Returns GPU target that uses CUDA.
     */
    CUDA_GPU(int id, int n_streams);

    CUDA_GPU(const CUDA_GPU &o) :
        id(o.id),
        n_streams(o.n_streams),
        i_stream(o.i_stream.load()),
        shared_memory_size(o.shared_memory_size),
        streams(o.streams),
        cublas_handles(o.cublas_handles)
    { }

    CUDA_GPU &operator=(const CUDA_GPU &o)
    {
        id = o.id;
        n_streams = o.n_streams;
        i_stream.store(o.i_stream.load());
        shared_memory_size = o.shared_memory_size;
        streams = o.streams;
        cublas_handles = o.cublas_handles;
        return *this;
    }

    /**
     * @brief Returns false because target is not CPU.
     */
    bool is_cpu() override;

    /**
     * @brief Returns true because target is a CUDA GPU.
     */
    bool is_gpu() override;

    /**
     * @brief Returns false because target is not a SYCL device.
     */
    bool is_sycl() override;

    /**
     * @brief Returns string representation of the GPU target.
     */
    std::string repr() const override;

    /**
     * @brief Creates n_streams CUDA streams and cublas handles.
     *
     * WARNING: Call destroy() to free both resources after using them.
     */
    void create();

    /**
     * @brief Destroys the CUDA streams and cublas handles previously created
     *        with create().
     */
    void destroy();

    /**
     * @brief Returns the next CUDA streams.
     *
     * It regards the collection of CUDA streams as a cyclic list and returns
     * the next CUDA stream in the cycle. The returned stream was already
     * created when calling create() and will be destroyed by using destroy().
     *
     * @return CUDA stream
     */
    cudaStream_t next_stream();

    /**
     * @brief Synchronizes the collection of CUDA streams.
     *
     * The streams must have be retrieved by next_stream(). Thus, it can use the
     * cyclic ordering to sync each stream in subset_of_streams only once.
     *
     * @param subset_of_streams Vector of CUDA streams, previously retrieved
     *                          with next_stream().
     */
    void sync_streams(std::vector<cudaStream_t> &subset_of_streams);

    /**
     * @brief Returns the next cuBLAS handle.
     *
     * It regards the collection of cuBLAS handles as a cyclic list and returns
     * the next handle in the cycle. The returned handle was already
     * created when calling create() and will be destroyed by using destroy().
     *
     * @return cuBLAS handle
     */
    std::pair<cublasHandle_t, cudaStream_t> next_cublas_handle();

  private:
    std::vector<cudaStream_t> streams;
    std::vector<cublasHandle_t> cublas_handles;
};

/**
 * @brief Creates and returns handle for GPU target.
 *
 * @param id ID of GPU.
 * @param n_streams Number of streams to be created on GPU.
 *
 * @return GPU target
 */
CUDA_GPU get_gpu(int id, int n_streams);

/**
 * @brief Returns handle for GPU target with ID 0.
 *
 * Uses only one stream, so single-threaded GPU execution.
 */
CUDA_GPU get_gpu();
#endif

// SYCL ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if GPRAT_WITH_SYCL
struct SYCL_DEVICE : public Target
{
    /**
     * @brief Identifier of SYCL device.
     *
     * Can be set to a value between 0 and device_count().
     */
    std::size_t id;

    /**
     * @brief Number of SYCL queues used asynchronous computation and data transfer.
     */
    std::size_t n_queues;

    /**
     * @brief Index of next SYCL queue assigned on next_queue().
     */
    std::atomic<std::size_t> i_queue;

    /** @brief Default amount of SYCL local memory used by kernels. */
    std::size_t local_memory_size;

    /**
     * @brief Returns GPU target that uses SYCL.
     */
    SYCL_DEVICE(int gpu_id, int n_queues);

    SYCL_DEVICE(const SYCL_DEVICE &o) :
        id(o.id),
        n_queues(o.n_queues),
        i_queue(o.i_queue.load()),
        local_memory_size(o.local_memory_size),
        selected_device_(o.selected_device_),
        queues(o.queues)
    { }

    SYCL_DEVICE &operator=(const SYCL_DEVICE &o)
    {
        id = o.id;
        n_queues = o.n_queues;
        i_queue.store(o.i_queue.load());
        local_memory_size = o.local_memory_size;
        selected_device_ = o.selected_device_;
        queues = o.queues;
        return *this;
    }

    /**
     * @brief Returns false because target is not CPU.
     */
    bool is_cpu() override;

    /**
     * @brief Returns false because target is not a CUDA GPU.
     */
    bool is_gpu() override;

    /**
     * @brief Returns true because target is a SYCL device.
     */
    bool is_sycl() override;

    /**
     * @brief Returns string representation of the SYCL target.
     */
    std::string repr() const override;

    /**
     * @brief Creates n_queues SYCL queues.
     *
     * WARNING: Call destroy() to free both resources after using them.
     */
    void create();

    /**
     * @brief Destroys the SYCL queues previously created with create().
     */
    void destroy();

    /**
     * @brief Returns the next SYCL queue.
     *
     * It regards the collection of SYCL queues as a cyclic list and returns
     * the next SYCL queue in the cycle. The returned queue was already
     * created when calling create() and will be destroyed by using destroy().
     *
     * @return SYCL queue
     */
    sycl::queue next_queue();

    /**
     * @brief Synchronizes the collection of SYCL queues.
     *
     * The queue must be retrieved by next_queue(). Thus, it can use the
     * cyclic ordering to sync each queue in subset_of_queues only once.
     *
     * @param subset_of_queue Vector of SYCL queues, previously retrieved
     *                          with next_queue().
     */
    void sync_queues(std::vector<sycl::queue> &subset_of_queues);

  private:
    sycl::device selected_device_;
    std::vector<sycl::queue> queues;
};

/**
 * @brief Creates and returns handle for SYCL target.
 *
 * @param id ID of SYCL device.
 * @param n_queues Number of queues to be created on SYCL device.
 *
 * @return SYCL target
 */
SYCL_DEVICE get_sycl_device(int id, int n_queues);

/**
 * @brief Returns handle for SYCL target with ID 0.
 *
 * Uses only one queue, so single-threaded GPU execution.
 */
SYCL_DEVICE get_sycl_device();
#endif

// General ////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Lists available GPUs with their properties.
 */
void print_available_gpus();

/**
 * @brief Returns number of available GPUs.
 */
int gpu_count();

GPRAT_NS_END

#endif
