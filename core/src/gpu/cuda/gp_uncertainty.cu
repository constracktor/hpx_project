#include "gprat/gpu/gp_uncertainty.cuh"

#include "gprat/gpu/cuda_utils.cuh"
#include "gprat/target.hpp"

#include <hpx/async_cuda/cuda_exception.hpp>

GPRAT_NS_BEGIN

using hpx::cuda::experimental::check_cuda_error;

namespace gpu
{

hpx::shared_future<double *>
diag_posterior(const hpx::shared_future<double *> A, const hpx::shared_future<double *> B, std::size_t M, CUDA_GPU &gpu)
{
    auto [cublas, stream] = gpu.next_cublas_handle();

    double *tile;
    check_cuda_error(cudaMalloc(&tile, M * sizeof(double)));

    const double add_A = 1.0;
    const double subtract_B = -1.0;

    // tile = 1.0*A + (-1.0)*B
    cublasDgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N, 1, M, &add_A, A.get(), 1, &subtract_B, B.get(), 1, tile, 1);
    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(tile);
}

hpx::shared_future<double *> diag_tile(const hpx::shared_future<double *> A, std::size_t M, CUDA_GPU &gpu)
{
    double *diag_tile;
    check_cuda_error(cudaMalloc(&diag_tile, M * sizeof(double)));

    auto [cublas, stream] = gpu.next_cublas_handle();

    cublasDcopy(cublas, M, A.get(), M + 1, diag_tile, 1);
    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(diag_tile);
}

}  // end of namespace gpu

GPRAT_NS_END
