#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#define WORK_GROUP_SIZE 16

// GPRat
#include "gprat/target.hpp"

// HPX
#include <hpx/algorithm.hpp>

// SYCL
#include <sycl/sycl.hpp>

namespace gprat::sycl_backend
{

/**
 * @brief Copies a vector from the host to the device using the next SYCL queue of sycl_device.
 *
 * Allocates device memory for the vector and synchronizes the stream after
 * copying the data.
 *
 * @param h_vector The vector to copy from the host
 * @param sycl_device The SYCL target
 *
 * @return A pointer to the copied vector on the device
 */
inline double *copy_to_device(const std::vector<double> &h_vector, gprat::SYCL_DEVICE &sycl_device)
{
    double *d_vector;
    sycl::queue queue = sycl_device.next_queue();

    try
    {
        d_vector = sycl::malloc_device<double>(h_vector.size(), queue);
        auto copy_process = queue.memcpy(d_vector, h_vector.data(), h_vector.size() * sizeof(double));
        copy_process.wait();
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
    }
    return d_vector;
}

/**
 * @brief Frees the device memory allocated in a vector of shared futures.
 *
 * @param vector The vector of shared futures to free
 * @param queue The SYCL queue to use for freeing the memory
 */
inline void free(std::vector<hpx::shared_future<double *>> &vector, sycl::queue queue)
{
    try
    {
        for (auto &ptr : vector)
        {
            sycl::free(ptr.get(), queue);
        }
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
    }
}

}  // end of namespace gprat::sycl_backend

#endif  // end of SYCL_UTILS_HPP
