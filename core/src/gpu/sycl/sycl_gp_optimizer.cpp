#include "gpu/sycl/sycl_gp_optimizer.hpp"

#include "gpu/sycl/adapter_onemath.hpp"
#include "gpu/sycl/sycl_kernels.hpp"
#include "gpu/sycl/sycl_utils.hpp"

namespace gprat::sycl_backend
{

// gen_tile_grad_l_trans //////////////////////////////////////////////////////////////////////////////////////////////

hpx::shared_future<double *>
gen_tile_grad_l_trans(std::size_t N, const hpx::shared_future<double *> f_grad_l_tile, gprat::SYCL_DEVICE &sycl_device)
{
    try
    {
        sycl::queue queue = sycl_device.next_queue();

        double *transposed = sycl::malloc_device<double>(N * N, queue);
        double *d_grad_l_tile = f_grad_l_tile.get();

        sycl::range<2> global_range(((N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE) * WORK_GROUP_SIZE,
                                    ((N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE) * WORK_GROUP_SIZE);
        sycl::range<2> local_range(WORK_GROUP_SIZE, WORK_GROUP_SIZE);

        auto event = queue.submit(
            [&](sycl::handler &cgh)
            {
                auto kernel = TransposeKernel(transposed, d_grad_l_tile, N, N, cgh);
                cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kernel);
            });

        event.wait();
        return hpx::make_ready_future(transposed);
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
        return hpx::make_ready_future(static_cast<double *>(nullptr));
    }
}

// compute_loss ///////////////////////////////////////////////////////////////////////////////////////////////////////

double compute_loss(const hpx::shared_future<double *> &K_diag_tile,
                    const hpx::shared_future<double *> &alpha_tile,
                    const hpx::shared_future<double *> &y_tile,
                    std::size_t N,
                    gprat::SYCL_DEVICE &sycl_device)
{
    sycl::queue queue = sycl_device.next_queue();
    // l = y^T * alpha + \sum_i^N log(L_ii^2)
    double l;
    // Compute y^T * alpha (result in shared USM, readable from host)
    double *d_dot = dot(queue, y_tile.get(), alpha_tile.get(), N);
    l = *d_dot;
    sycl::free(d_dot, queue);
    // Copy diagonal tile to host so we can read it
    std::vector<double> h_diag(N * N);
    queue.memcpy(h_diag.data(), K_diag_tile.get(), N * N * sizeof(double)).wait();
    // Compute \sum_i^N log(L_ii^2)
    for (std::size_t i = 0; i < N; i++)
    {
        l += std::log(h_diag[i * N + i] * h_diag[i * N + i]);
    }
    return l;
}

// add_losses /////////////////////////////////////////////////////////////////////////////////////////////////////////

hpx::shared_future<double>
add_losses(const std::vector<hpx::shared_future<double>> &losses, std::size_t n_tile_size, std::size_t n_tiles)
{
    // Add the squared difference to the error
    double l = 0.0;
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        l += losses[i].get();
    }
    l += static_cast<double>(n_tile_size) * static_cast<double>(n_tiles) * log(2.0 * M_PI);

    return hpx::make_ready_future(0.5 * l / static_cast<double>((n_tile_size * n_tiles)));
}

}  // end of namespace gprat::sycl_backend
