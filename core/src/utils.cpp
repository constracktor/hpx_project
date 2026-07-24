#include "gprat/utils.hpp"

#include <cstdio>
#if GPRAT_WITH_SYCL
#include <dlfcn.h>
#endif

GPRAT_NS_BEGIN

std::size_t compute_train_tiles(std::size_t n_samples, std::size_t n_tile_size)
{
    if (n_tile_size > 0)
    {
        // n_tiles
        return n_samples / n_tile_size;
    }
    else
    {
        throw std::runtime_error("Error: Please specify a valid value for train_tile_size.\n");
    }
}

std::size_t compute_train_tile_size(std::size_t n_samples, std::size_t n_tiles)
{
    if (n_tiles > 0)
    {
        // n_tile_size
        return n_samples / n_tiles;
    }
    else
    {
        throw std::runtime_error("Error: Please specify a valid value for train_tiles.\n");
    }
}

std::pair<std::size_t, std::size_t> compute_test_tiles(std::size_t n_test, std::size_t n_tiles, std::size_t n_tile_size)
{
    std::size_t m_tiles;
    std::size_t m_tile_size;

    // if n_test is not divisible by (incl. smaller than) n_tile_size, use the same number of tiles
    if ((n_test % n_tile_size) > 0)
    {
        m_tiles = n_tiles;
        m_tile_size = n_test / m_tiles;
    }
    // if n_test is divisible by n_tile_size, use the same tile size
    else
    {
        m_tiles = n_test / n_tile_size;
        m_tile_size = n_tile_size;
    }
    return { m_tiles, m_tile_size };
}

std::size_t guess_good_tile_count_per_dimension(std::size_t n)
{
    // These have been found through experimentation - they are only estimates that have been shown to perform
    // better than fixed tile counts.
    const auto hw_concurrency = hpx::threads::hardware_concurrency();

    // For small datasets / few cores we shouldn't bother
    if (n < (1 << 8) || hw_concurrency < 4)
    {
        return 1;
    }

    if (n < (1 << 12) || hw_concurrency < 16)
    {
        return 4;
    }

    if (n < (1 << 18) || hw_concurrency < 32)
    {
        return 16;
    }

    return std::min<std::size_t>(hw_concurrency, n / 256);
}

std::vector<double> load_data(const std::string &file_path, std::size_t n_samples, std::size_t offset)
{
    std::vector<double> _data;
    _data.resize(n_samples + offset, 0.0);

    FILE *input_file = fopen(file_path.c_str(), "r");
    if (input_file == NULL)
    {
        throw std::runtime_error("Error: File not found: " + file_path);
    }

    // load data
    std::size_t scanned_elements = 0;
    for (std::size_t i = 0; i < n_samples; i++)
    {
        const auto r = fscanf(input_file, "%lf", &_data[(i + offset)]);
        if (r > 0)
        {
            scanned_elements += static_cast<std::size_t>(r);
        }
    }

    fclose(input_file);

    if (scanned_elements != n_samples)
    {
        throw std::runtime_error("Error: Data not correctly read. Expected " + std::to_string(n_samples)
                                 + " elements, but read " + std::to_string(scanned_elements));
    }
    return _data;
}

void print_vector(const std::vector<double> &vec, int start, int end, const std::string &separator)
{
    // Convert negative indices to positive
    if (start < 0)
    {
        start += static_cast<int>(vec.size());
    }
    if (end < 0)
    {
        end += static_cast<int>(vec.size()) + 1;
    }

    // Ensure the indices are within bounds
    if (start < 0)
    {
        start = 0;
    }
    if (end > static_cast<int>(vec.size()))
    {
        end = static_cast<int>(vec.size());
    }

    // Validate the range
    if (start >= static_cast<int>(vec.size()) || start >= end)
    {
        std::cerr << "Invalid range" << std::endl;
        return;
    }

    for (int i = start; i < end; i++)
    {
        std::cout << vec[static_cast<std::size_t>(i)];
        if (i < end - 1)
        {
            std::cout << separator;
        }
    }
    std::cout << std::endl;
}

void start_hpx_runtime(int argc, char **argv) { hpx::start(nullptr, argc, argv); }

void resume_hpx_runtime() { hpx::resume(); }

void suspend_hpx_runtime() { hpx::suspend(); }

void stop_hpx_runtime()
{
    hpx::post([]() { hpx::finalize(); });
    hpx::stop();
}

bool compiled_with_cuda()
{
#if GPRAT_WITH_CUDA
    return true;
#else
    return false;
#endif
}

bool compiled_with_sycl()
{
#if GPRAT_WITH_SYCL
    return true;
#else
    return false;
#endif
}

bool sycl_gpu_functional()
{
#if GPRAT_WITH_SYCL
    // libonemath_lapack_cusolver.so requires sycl::_V1::detail::SubmissionInfo::SubmissionInfo()
    // which was introduced after oneAPI 2025.0. If the symbol is absent the LAPACK backend will
    // crash with a dynamic linker error at the first call site.
    dlerror();  // clear any previous error
    void *sym = dlsym(RTLD_DEFAULT, "_ZN4sycl3_V16detail14SubmissionInfoC1Ev");
    return sym != nullptr;
#else
    return false;
#endif
}

GPRAT_NS_END
