#include "gpu/cuda/gp_algorithms.cuh"

#include "gp_kernels.hpp"
#include "gpu/cuda/cuda_kernels.cuh"
#include "gpu/cuda/cuda_utils.cuh"
#include "gpu/cuda/gp_optimizer.cuh"
#include "target.hpp"
#include <cuda_runtime.h>
#include <hpx/algorithm.hpp>
#include <hpx/async_cuda/cuda_exception.hpp>

namespace gpu
{

// Kernel function to compute covariance
__global__ void gen_tile_covariance_kernel(
    double *d_tile,
    const double *d_input,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const gprat_hyper::SEKParams sek_params)
{
    // Compute the global indices of the thread
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_tile_size && j < n_tile_size)
    {
        std::size_t i_global = n_tile_size * tile_row + i;
        std::size_t j_global = n_tile_size * tile_column + j;

        double distance = 0.0;
        double z_ik_minus_z_jk;

        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            z_ik_minus_z_jk = d_input[i_global + k] - d_input[j_global + k];
            distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
        }

        // Compute the covariance value
        double covariance =
            sek_params.vertical_lengthscale * exp(-0.5 * distance / (sek_params.lengthscale * sek_params.lengthscale));

        // Add noise variance if diagonal
        if (i_global == j_global)
        {
            covariance += sek_params.noise_variance;
        }

        d_tile[i * n_tile_size + j] = covariance;
    }
}

double *gen_tile_covariance(const double *d_input,
                            const std::size_t tile_row,
                            const std::size_t tile_column,
                            const std::size_t n_tile_size,
                            const std::size_t n_regressors,
                            const gprat_hyper::SEKParams sek_params,
                            gprat::CUDA_GPU &gpu)
{
    double *d_tile;

    dim3 threads_per_block(16, 16);
    dim3 n_blocks((n_tile_size + 15) / 16, (n_tile_size + 15) / 16);

    cudaStream_t stream = gpu.next_stream();

    check_cuda_error(cudaMalloc(&d_tile, n_tile_size * n_tile_size * sizeof(double)));
    gen_tile_covariance_kernel<<<n_blocks, threads_per_block, gpu.shared_memory_size, stream>>>(
        d_tile, d_input, n_tile_size, n_regressors, tile_row, tile_column, sek_params);

    check_cuda_error(cudaStreamSynchronize(stream));

    return d_tile;
}

__global__ void gen_tile_full_prior_covariance_kernel(
    double *d_tile,
    const double *d_input,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const gprat_hyper::SEKParams sek_params)
{
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_tile_size && j < n_tile_size)
    {
        std::size_t i_global = n_tile_size * tile_row + i;
        std::size_t j_global = n_tile_size * tile_column + j;

        double distance = 0.0;
        double z_ik_minus_z_jk;

        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            z_ik_minus_z_jk = d_input[i_global + k] - d_input[j_global + k];
            distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
        }

        double covariance =
            sek_params.vertical_lengthscale * exp(-0.5 * distance / (sek_params.lengthscale * sek_params.lengthscale));

        d_tile[i * n_tile_size + j] = covariance;
    }
}

double *gen_tile_full_prior_covariance(
    const double *d_input,
    const std::size_t tile_row,
    const std::size_t tile_colums,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::CUDA_GPU &gpu)
{
    double *d_tile;

    dim3 threads_per_block(16, 16);
    dim3 n_blocks((n_tile_size + 15) / 16, (n_tile_size + 15) / 16);

    cudaStream_t stream = gpu.next_stream();

    check_cuda_error(cudaMalloc(&d_tile, n_tile_size * n_tile_size * sizeof(double)));
    gen_tile_full_prior_covariance_kernel<<<n_blocks, threads_per_block, gpu.shared_memory_size, stream>>>(
        d_tile, d_input, n_tile_size, n_regressors, tile_row, tile_colums, sek_params);

    check_cuda_error(cudaStreamSynchronize(stream));

    return d_tile;
}

__global__ void gen_tile_prior_covariance_kernel(
    double *d_tile,
    const double *d_input,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const gprat_hyper::SEKParams sek_params)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_tile_size)
    {
        std::size_t i_global = n_tile_size * tile_row + i;
        std::size_t j_global = n_tile_size * tile_column + i;

        double distance = 0.0;
        double z_ik_minus_z_jk;

        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            z_ik_minus_z_jk = d_input[i_global + k] - d_input[j_global + k];
            distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
        }

        double covariance =
            sek_params.vertical_lengthscale * exp(-0.5 * distance / (sek_params.lengthscale * sek_params.lengthscale));

        d_tile[i] = covariance;
    }
}

double *gen_tile_prior_covariance(
    const double *d_input,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::CUDA_GPU &gpu)
{
    double *d_tile;

    dim3 threads_per_block(256);
    dim3 n_blocks((n_tile_size + 255) / 256);

    cudaStream_t stream = gpu.next_stream();

    check_cuda_error(cudaMalloc(&d_tile, n_tile_size * sizeof(double)));
    gen_tile_prior_covariance_kernel<<<n_blocks, threads_per_block, 0, stream>>>(
        d_tile, d_input, n_tile_size, n_regressors, tile_row, tile_column, sek_params);

    check_cuda_error(cudaStreamSynchronize(stream));

    return d_tile;
}

__global__ void gen_tile_cross_covariance_kernel(
    double *d_tile,
    const double *d_row_input,
    const double *d_col_input,
    const std::size_t n_row_tile_size,
    const std::size_t n_column_tile_size,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params)
{
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_row_tile_size && j < n_column_tile_size)
    {
        std::size_t i_global = n_row_tile_size * tile_row + i;
        std::size_t j_global = n_column_tile_size * tile_column + j;

        double distance = 0.0;
        double z_ik_minus_z_jk;

        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            z_ik_minus_z_jk = d_row_input[i_global + k] - d_col_input[j_global + k];
            distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
        }

        double covariance =
            sek_params.vertical_lengthscale * exp(-0.5 * distance / (sek_params.lengthscale * sek_params.lengthscale));
        d_tile[i * n_column_tile_size + j] = covariance;
    }
}

double *gen_tile_cross_covariance(
    const double *d_row_input,
    const double *d_col_input,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const std::size_t n_row_tile_size,
    const std::size_t n_column_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::CUDA_GPU &gpu)
{
    double *d_tile;

    dim3 threads_per_block(16, 16);
    dim3 n_blocks((n_column_tile_size + 15) / 16, (n_row_tile_size + 15) / 16);

    cudaStream_t stream = gpu.next_stream();

    check_cuda_error(cudaMalloc(&d_tile, n_row_tile_size * n_column_tile_size * sizeof(double)));
    gen_tile_cross_covariance_kernel<<<n_blocks, threads_per_block, 0, stream>>>(
        d_tile,
        d_row_input,
        d_col_input,
        n_row_tile_size,
        n_column_tile_size,
        tile_row,
        tile_column,
        n_regressors,
        sek_params);

    check_cuda_error(cudaStreamSynchronize(stream));

    return d_tile;
}

hpx::shared_future<double *> gen_tile_cross_cov_T(std::size_t n_row_tile_size,
                                                  std::size_t n_column_tile_size,
                                                  const hpx::shared_future<double *> f_cross_covariance_tile,
                                                  gprat::CUDA_GPU &gpu)
{
    double *transposed;
    check_cuda_error(cudaMalloc(&transposed, n_row_tile_size * n_column_tile_size * sizeof(double)));
    double *d_cross_covariance_tile = f_cross_covariance_tile.get();

    cudaStream_t stream = gpu.next_stream();
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks((n_column_tile_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (n_row_tile_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    transpose<<<n_blocks, threads_per_block, 0, stream>>>(
        transposed, d_cross_covariance_tile, n_column_tile_size, n_row_tile_size);

    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(transposed);
}

__global__ void gen_tile_output_kernel(double *tile, const double *output, std::size_t row, std::size_t n_tile_size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_tile_size)
    {
        std::size_t i_global = n_tile_size * row + i;
        tile[i] = output[i_global];
    }
}

double *
gen_tile_output(const std::size_t row, const std::size_t n_tile_size, const double *d_output, gprat::CUDA_GPU &gpu)
{
    dim3 threads_per_block(256);
    dim3 n_blocks((n_tile_size + 255) / 256);

    cudaStream_t stream = gpu.next_stream();

    double *d_tile;
    check_cuda_error(cudaMalloc(&d_tile, n_tile_size * sizeof(double)));

    gen_tile_output_kernel<<<n_blocks, threads_per_block, 0, stream>>>(d_tile, d_output, row, n_tile_size);

    check_cuda_error(cudaStreamSynchronize(stream));

    return d_tile;
}

double *gen_tile_zeros(std::size_t n_tile_size, gprat::CUDA_GPU &gpu)
{
    double *d_tile;
    cudaStream_t stream = gpu.next_stream();
    check_cuda_error(cudaMalloc(&d_tile, n_tile_size * sizeof(double)));
    check_cuda_error(cudaMemsetAsync(d_tile, 0, n_tile_size * sizeof(double), stream));
    check_cuda_error(cudaStreamSynchronize(stream));
    return d_tile;
}

double compute_error_norm(std::size_t n_tiles,
                          std::size_t n_tile_size,
                          const std::vector<double> &b,
                          const std::vector<std::vector<double>> &tiles)
{
    double error = 0.0;
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        auto a = tiles[k];
        for (std::size_t i = 0; i < n_tile_size; i++)
        {
            std::size_t i_global = n_tile_size * k + i;
            // ||a - b||_2
            error += (b[i_global] - a[i]) * (b[i_global] - a[i]);
        }
    }
    return sqrt(error);
}

std::vector<hpx::shared_future<double *>> assemble_tiled_covariance_matrix(
    const double *d_training_input,
    const std::size_t n_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> d_tiles(n_tiles * n_tiles);

    for (std::size_t tile_row = 0; tile_row < n_tiles; ++tile_row)
    {
        for (std::size_t tile_column = 0; tile_column < tile_row + 1; ++tile_column)
        {
            d_tiles[tile_row * n_tiles + tile_column] = hpx::async(
                &gen_tile_covariance,
                d_training_input,
                tile_row,
                tile_column,
                n_tile_size,
                n_regressors,
                sek_params,
                std::ref(gpu));
        }
    }

    return d_tiles;
}

std::vector<hpx::shared_future<double *>> assemble_alpha_tiles(
    const double *d_output, const std::size_t n_tiles, const std::size_t n_tile_size, gprat::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> alpha_tiles(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_tiled_alpha"), i, n_tile_size, d_output, std::ref(gpu));
    }

    return alpha_tiles;
}

std::vector<hpx::shared_future<double *>> assemble_cross_covariance_tiles(
    const double *d_test_input,
    const double *d_training_input,
    const std::size_t m_tiles,
    const std::size_t n_tiles,
    const std::size_t m_tile_size,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> cross_covariance_tiles;
    cross_covariance_tiles.resize(m_tiles * n_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            cross_covariance_tiles[i * n_tiles + j] = hpx::async(
                hpx::annotated_function(&gen_tile_cross_covariance, "assemble_pred"),
                d_test_input,
                d_training_input,
                i,
                j,
                m_tile_size,
                n_tile_size,
                n_regressors,
                sek_params,
                std::ref(gpu));
        }
    }
    return cross_covariance_tiles;
}

std::vector<hpx::shared_future<double *>>
assemble_tiles_with_zeros(std::size_t n_tile_size, std::size_t n_tiles, gprat::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> tiles(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        tiles[i] = hpx::async(&gen_tile_zeros, n_tile_size, std::ref(gpu));
    }
    return tiles;
}

std::vector<hpx::shared_future<double *>> assemble_prior_K_tiles(
    const double *d_test_input,
    const std::size_t m_tiles,
    const std::size_t m_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> d_prior_K_tiles;
    d_prior_K_tiles.resize(m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        d_prior_K_tiles[i] = hpx::async(
            &gen_tile_prior_covariance, d_test_input, i, i, m_tile_size, n_regressors, sek_params, std::ref(gpu));
    }
    return d_prior_K_tiles;
}

std::vector<hpx::shared_future<double *>> assemble_prior_K_tiles_full(
    const double *d_test_input,
    const std::size_t m_tiles,
    const std::size_t m_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> d_prior_K_tiles(m_tiles * m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            d_prior_K_tiles[i * m_tiles + j] = hpx::async(
                &gen_tile_full_prior_covariance,
                d_test_input,
                i,
                j,
                m_tile_size,
                n_regressors,
                sek_params,
                std::ref(gpu));

            if (i != j)
            {
                d_prior_K_tiles[j * m_tiles + i] =
                    hpx::dataflow(&gen_tile_grad_l_trans, m_tile_size, d_prior_K_tiles[i * m_tiles + j], std::ref(gpu));
            }
        }
    }
    return d_prior_K_tiles;
}

std::vector<hpx::shared_future<double *>> assemble_t_cross_covariance_tiles(
    const std::vector<hpx::shared_future<double *>> &d_cross_covariance_tiles,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    gprat::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> d_t_cross_covariance_tiles(m_tiles * n_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            d_t_cross_covariance_tiles[j * m_tiles + i] = hpx::dataflow(
                &gen_tile_cross_cov_T,
                m_tile_size,
                n_tile_size,
                d_cross_covariance_tiles[i * n_tiles + j],
                std::ref(gpu));
        }
    }
    return d_t_cross_covariance_tiles;
}

std::vector<hpx::shared_future<double *>> assemble_y_tiles(
    const double *d_training_output, const std::size_t n_tiles, const std::size_t n_tile_size, gprat::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> d_y_tiles(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        d_y_tiles[i] = hpx::async(&gen_tile_output, i, n_tile_size, d_training_output, std::ref(gpu));
    }
    return d_y_tiles;
}

std::vector<double> copy_tiled_vector_to_host_vector(std::vector<hpx::shared_future<double *>> &d_tiles,
                                                     std::size_t n_tile_size,
                                                     std::size_t n_tiles,
                                                     gprat::CUDA_GPU &gpu)
{
    std::vector<double> h_vector(n_tiles * n_tile_size);
    std::vector<cudaStream_t> streams(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        streams[i] = gpu.next_stream();
        check_cuda_error(cudaMemcpyAsync(
            h_vector.data() + i * n_tile_size,
            d_tiles[i].get(),
            n_tile_size * sizeof(double),
            cudaMemcpyDeviceToHost,
            streams[i]));
    }
    gpu.sync_streams(streams);
    return h_vector;
}

std::vector<std::vector<double>> move_lower_tiled_matrix_to_host(
    const std::vector<hpx::shared_future<double *>> &d_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gprat::CUDA_GPU &gpu)
{
    std::vector<std::vector<double>> h_tiles(n_tiles * n_tiles);

    std::vector<cudaStream_t> streams(n_tiles * (n_tiles + 1) / 2);
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            streams[i] = gpu.next_stream();
            h_tiles[i * n_tiles + j].resize(n_tile_size * n_tile_size);
            check_cuda_error(cudaMemcpyAsync(
                h_tiles[i * n_tiles + j].data(),
                d_tiles[i * n_tiles + j].get(),
                n_tile_size * n_tile_size * sizeof(double),
                cudaMemcpyDeviceToHost,
                streams[i]));
            check_cuda_error(cudaFree(d_tiles[i * n_tiles + j].get()));
        }
    }
    gpu.sync_streams(streams);

    return h_tiles;
}

void free_lower_tiled_matrix(const std::vector<hpx::shared_future<double *>> &d_tiles, const std::size_t n_tiles)
{
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            check_cuda_error(cudaFree(d_tiles[i * n_tiles + j].get()));
        }
    }
}

void free_full_tiled_matrix(const std::vector<hpx::shared_future<double *>> &d_tiles, const std::size_t n_tiles)
{
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            check_cuda_error(cudaFree(d_tiles[i * n_tiles + j].get()));
        }
    }
}

}  // end of namespace gpu
