#include "gprat/target.hpp"

#include <iostream>
#include <unordered_map>
#include <vector>

#if GPRAT_WITH_CUDA
#include "gprat/gpu/cuda_utils.cuh"
#endif

#if GPRAT_WITH_SYCL
#include "gpu/sycl/sycl_utils.hpp"
#endif

GPRAT_NS_BEGIN

CPU::CPU() = default;

bool CPU::is_cpu() { return true; }

bool CPU::is_gpu() { return false; }

bool CPU::is_sycl() { return false; }

std::string CPU::repr() const { return "CPU"; }

CPU get_cpu() { return CPU(); }

// CUDA ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if GPRAT_WITH_CUDA
CUDA_GPU::CUDA_GPU(int id, int n_streams) :
    id(id),
    n_streams(n_streams),
    i_stream(0),
    shared_memory_size(0),
    streams()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (id >= deviceCount)
    {
        throw std::runtime_error("Requested GPU device is not available.");
    }
    check_cuda_error(cudaSetDevice(id));
}

bool CUDA_GPU::is_cpu() { return false; }

bool CUDA_GPU::is_gpu() { return true; }

bool CUDA_GPU::is_sycl() { return false; }

std::string CUDA_GPU::repr() const
{
    std::ostringstream oss;
    oss << "GPU (CUDA): [id=" << id << ", n_streams=" << n_streams << "]";
    return oss.str();
}

void CUDA_GPU::create()
{
    // Streams and cuBLAS handles are bound to the device active at creation time,
    // so make sure the requested device is selected (also on this thread).
    check_cuda_error(cudaSetDevice(id));
    streams = std::vector<cudaStream_t>(static_cast<std::size_t>(n_streams));
    cublas_handles = std::vector<cublasHandle_t>(static_cast<std::size_t>(n_streams));
    for (size_t i = 0; i < streams.size(); ++i)
    {
        check_cuda_error(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        cublasCreate(&cublas_handles[i]);
    }
}

void CUDA_GPU::destroy()
{
    check_cuda_error(cudaSetDevice(id));
    for (size_t i = 0; i < streams.size(); ++i)
    {
        check_cuda_error(cudaStreamDestroy(streams[i]));
        cublasDestroy(cublas_handles[i]);
    }
}

cudaStream_t CUDA_GPU::next_stream()
{
    return streams[static_cast<std::size_t>(i_stream++) % static_cast<std::size_t>(n_streams)];
}

void CUDA_GPU::sync_streams(std::vector<cudaStream_t> &subset_of_streams)
{
    if (subset_of_streams.size() < streams.size())
    {
        for (cudaStream_t &stream : subset_of_streams)
        {
            check_cuda_error(cudaStreamSynchronize(stream));
        }
    }
    else
    {
        for (cudaStream_t &stream : streams)
        {
            check_cuda_error(cudaStreamSynchronize(stream));
        }
    }
}

std::pair<cublasHandle_t, cudaStream_t> CUDA_GPU::next_cublas_handle()
{
    std::size_t i = static_cast<std::size_t>(i_stream++);
    cublasHandle_t cublas = cublas_handles[i % static_cast<std::size_t>(n_streams)];
    cudaStream_t stream = streams[i % static_cast<std::size_t>(n_streams)];
    cublasSetStream(cublas, stream);

    return std::make_pair(cublas, stream);
}

CUDA_GPU get_gpu(int id, int n_streams) { return CUDA_GPU(id, n_streams); }

CUDA_GPU get_gpu() { return CUDA_GPU(0, 1); }

#endif

// SYCL ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if GPRAT_WITH_SYCL

SYCL_DEVICE::SYCL_DEVICE(int id, int n_queues) :
    id(static_cast<std::size_t>(id)),
    n_queues(static_cast<std::size_t>(n_queues)),
    i_queue(0),
    local_memory_size(0),
    queues()
{
    try
    {
        std::vector<sycl::device> all_gpus;
        std::vector<sycl::platform> platforms = sycl::platform::get_platforms();

        for (const auto &platform : platforms)
        {
            std::vector<sycl::device> devices = platform.get_devices();
            for (const auto &device : devices)
            {
                if (device.get_info<sycl::info::device::device_type>() == sycl::info::device_type::gpu)
                {
                    all_gpus.push_back(device);
                }
            }
        }

        const std::size_t device_count = all_gpus.size();
        if (static_cast<std::size_t>(id) >= device_count)
        {
            throw std::runtime_error("Requested GPU device is not available.");
        }

        // Store the selected device so create() can target it specifically.
        selected_device_ = all_gpus[static_cast<std::size_t>(id)];
    }
    catch (const std::exception &e)
    {
        std::cout << "SYCL error during device selection: " << e.what() << "\n";
        throw;
    }
}

bool SYCL_DEVICE::is_cpu() { return false; }

bool SYCL_DEVICE::is_gpu() { return false; }

bool SYCL_DEVICE::is_sycl() { return true; }

std::string SYCL_DEVICE::repr() const
{
    std::ostringstream oss;
    oss << "SYCL DEVICE: [id=" << id << ", n_queues=" << n_queues << "]";
    return oss.str();
}

void SYCL_DEVICE::create()
{
    try
    {
        // Each fresh GP object creates its own queue from a bare selector, which
        // creates a new Level-Zero context; hundreds of cycles exhaust driver
        // resources (DEVICE_LOST). Share one context per device process-wide instead.
        static std::unordered_map<std::size_t, sycl::context> shared_contexts;
        auto it = shared_contexts.find(id);
        if (it == shared_contexts.end())
        {
            it = shared_contexts.emplace(id, sycl::context(selected_device_)).first;
        }

        queues = std::vector<sycl::queue>(n_queues);

        for (size_t i = 0; i < n_queues; ++i)
        {
            queues[i] = sycl::queue(it->second, selected_device_);
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "SYCL error during queue creation: " << e.what() << "\n";
        throw;
    }
}

void SYCL_DEVICE::destroy()
{
    try
    {
        queues.clear();
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception during destruction: " << e.what() << "\n";
    }
}

sycl::queue SYCL_DEVICE::next_queue()
{
    return queues[static_cast<std::size_t>(i_queue++) % static_cast<std::size_t>(n_queues)];
}

void SYCL_DEVICE::sync_queues(std::vector<sycl::queue> &subset_of_queues)
{
    try
    {
        if (subset_of_queues.size() < queues.size())
        {
            for (sycl::queue &queue : subset_of_queues)
            {
                queue.wait();
            }
        }
        else
        {
            for (sycl::queue &queue : queues)
            {
                queue.wait();
            }
        }
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
    }
}

SYCL_DEVICE get_sycl_device(const std::size_t id, const std::size_t n_queues)
{
    return SYCL_DEVICE(static_cast<int>(id), static_cast<int>(n_queues));
}

SYCL_DEVICE get_sycl_device() { return SYCL_DEVICE(0, 1); }

#endif

// General ////////////////////////////////////////////////////////////////////////////////////////////////////////////

void print_available_gpus()
{
#if GPRAT_WITH_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        // clang-format off
        std::cout
            << "Device " << i << ": " << deviceProp.name << "\n"
            << "  Total Global Memory: " << deviceProp.totalGlobalMem << "\n"
            << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock << "\n"
            << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << "\n"
            << "  Total Constant Memory: " << deviceProp.totalConstMem << "\n"
            << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n"
            << "  Multiprocessor Count: " << deviceProp.multiProcessorCount << "\n"
            << "  Clock Rate: " << deviceProp.clockRate << " kHz\n"
            << "  Memory Clock Rate: " << deviceProp.memoryClockRate << " kHz\n"
            << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        // clang-format on
    }
#elif GPRAT_WITH_SYCL
    try
    {
        // Get all available platforms
        std::vector<sycl::platform> platforms = sycl::platform::get_platforms();

        // Loop over all platforms
        for (const auto &platform : platforms)
        {
            std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << "\n";

            // Get all devices for each platform
            std::vector<sycl::device> devices = platform.get_devices();

            for (size_t i = 0; i < devices.size(); ++i)
            {
                sycl::device device = devices[i];

                // Check if the device is a GPU
                if (device.get_info<sycl::info::device::device_type>() == sycl::info::device_type::gpu)
                {
                    std::cout << "Device " << i << ": " << device.get_info<sycl::info::device::name>() << "\n";

                    // Query various device properties for GPUs
                    try
                    {
                        std::cout << "  Total Global Memory: " << device.get_info<sycl::info::device::global_mem_size>()
                                  << " bytes\n"
                                  << "  Max Compute Units: " << device.get_info<sycl::info::device::max_compute_units>()
                                  << "\n"
                                  << "  Max Work Group Size: "
                                  << device.get_info<sycl::info::device::max_work_group_size>() << "\n"
                                  << "  Max Work Item Dimensions: "
                                  << device.get_info<sycl::info::device::max_work_item_dimensions>() << "\n"
                                  << "  Max Clock Frequency: "
                                  << device.get_info<sycl::info::device::max_clock_frequency>() << " MHz\n"
                                  << "  Max Memory Allocation Size: "
                                  << device.get_info<sycl::info::device::max_mem_alloc_size>() << " bytes\n";
                    }
                    catch (const sycl::exception &e)
                    {
                        std::cerr << "Error querying device properties: " << e.what() << std::endl;
                    }
                }
            }
        }
    }
    catch (const sycl::exception &e)
    {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
    }
#else
    std::cout << "There are no GPUs available. You can only "
                 "`get_cpu()` to utilize the CPU for computation."
              << std::endl;
#endif
}

int gpu_count()
{
#if GPRAT_WITH_CUDA

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;

#elif GPRAT_WITH_SYCL

    try
    {
        std::vector<sycl::device> all_gpus;
        std::vector<sycl::platform> platforms = sycl::platform::get_platforms();

        for (const auto &platform : platforms)
        {
            std::vector<sycl::device> devices = platform.get_devices();
            for (const auto &device : devices)
            {
                if (device.get_info<sycl::info::device::device_type>() == sycl::info::device_type::gpu)
                {
                    all_gpus.push_back(device);
                }
            }
        }
        return static_cast<int>(all_gpus.size());
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
    }
    return 0;

#else

    std::cout << "GPRat has been compiled without GPU support. You can only "
                 "use `get_cpu()` to utilize the CPU for computation."
              << std::endl;
    return 0;

#endif
}

GPRAT_NS_END
