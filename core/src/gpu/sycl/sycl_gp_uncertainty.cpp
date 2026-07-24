// GPRat
#include "gpu/sycl/sycl_gp_uncertainty.hpp"

#include "gprat/target.hpp"

#include "gpu/sycl/sycl_utils.hpp"

// oneMath
#include <oneapi/math.hpp>

namespace gprat::sycl_backend
{

double *diag_posterior(double *A, double *B, std::size_t M)
{
    sycl::queue queue(sycl::gpu_selector_v);

    double *tile = sycl::malloc_device<double>(M, queue);

    // tile = A - B
    queue.parallel_for(sycl::range<1>(M), [=](sycl::id<1> i) { tile[i] = A[i] - B[i]; });

    queue.wait();

    return tile;
}

double *diag_tile(double *A, std::size_t M)
{
    sycl::queue queue(sycl::gpu_selector_v);

    double *diag_tile = sycl::malloc_device<double>(M, queue);

    // diag_tile = diagonal of the MxM matrix A (leading dimension M)
    queue.parallel_for(sycl::range<1>(M), [=](sycl::id<1> i) { diag_tile[i] = A[i * (M + 1)]; });

    queue.wait();

    return diag_tile;
}

}  // end of namespace gprat::sycl_backend
