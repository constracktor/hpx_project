#include "gprat/gpu/tiled_algorithms.cuh"

#include "gprat/gpu/adapter_cublas.cuh"
#include "gprat/gpu/gp_optimizer.cuh"
#include "gprat/gpu/gp_uncertainty.cuh"

#include <hpx/algorithm.hpp>

GPRAT_NS_BEGIN

namespace gpu
{

// Tiled Cholesky Algorithm

void right_looking_cholesky_tiled(std::vector<hpx::shared_future<double *>> &ft_tiles,
                                  const std::size_t n_tile_size,
                                  const std::size_t n_tiles,
                                  CUDA_GPU &gpu,
                                  const cusolverDnHandle_t &cusolver)
{
    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        cudaStream_t stream = gpu.next_stream();
        cusolverDnSetStream(cusolver, stream);

        // POTRF
        ft_tiles[k * n_tiles + k] = hpx::dataflow(
            hpx::annotated_function(&potrf, "Cholesky POTRF"),
            cusolver,
            stream,
            ft_tiles[k * n_tiles + k],
            n_tile_size);

        // NOTE: The result is immediately needed by TRSM. Also TRSM may throw
        // an error otherwise.
        ft_tiles[k * n_tiles + k].get();

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // TRSM
            ft_tiles[m * n_tiles + k] = hpx::dataflow(
                &trsm,
                cublas,
                stream,
                ft_tiles[k * n_tiles + k],
                ft_tiles[m * n_tiles + k],
                n_tile_size,
                n_tile_size,
                Blas_trans,
                Blas_right);
        }

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // SYRK
            ft_tiles[m * n_tiles + m] =
                hpx::dataflow(&syrk, cublas, stream, ft_tiles[m * n_tiles + k], ft_tiles[m * n_tiles + m], n_tile_size);

            for (std::size_t n = k + 1; n < m; ++n)
            {
                auto [cublas, stream] = gpu.next_cublas_handle();

                // GEMM
                ft_tiles[m * n_tiles + n] = hpx::dataflow(
                    &gemm,
                    cublas,
                    stream,
                    ft_tiles[m * n_tiles + k],
                    ft_tiles[n * n_tiles + k],
                    ft_tiles[m * n_tiles + n],
                    n_tile_size,
                    n_tile_size,
                    n_tile_size,
                    Blas_no_trans,
                    Blas_trans);
            }
        }
    }
}

// Tiled Triangular Solve Algorithms

void forward_solve_tiled(std::vector<hpx::shared_future<double *>> &ft_tiles,
                         std::vector<hpx::shared_future<double *>> &ft_rhs,
                         const std::size_t n_tile_size,
                         const std::size_t n_tiles,
                         CUDA_GPU &gpu)
{
    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        auto [cublas, stream] = gpu.next_cublas_handle();

        // TRSM: Solve L * x = a
        ft_rhs[k] =
            hpx::dataflow(&trsv, cublas, stream, ft_tiles[k * n_tiles + k], ft_rhs[k], n_tile_size, Blas_no_trans);

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // GEMV: b = b - A * a
            ft_rhs[m] = hpx::dataflow(
                &gemv,
                cublas,
                stream,
                ft_tiles[m * n_tiles + k],
                ft_rhs[k],
                ft_rhs[m],
                n_tile_size,
                n_tile_size,
                Blas_substract,
                Blas_no_trans);
        }
    }
}

void backward_solve_tiled(std::vector<hpx::shared_future<double *>> &ft_tiles,
                          std::vector<hpx::shared_future<double *>> &ft_rhs,
                          const std::size_t n_tile_size,
                          const std::size_t n_tiles,
                          CUDA_GPU &gpu)
{
    // NOTE: The loops traverse backwards. Its last comparisons require the
    // usage negative numbers. Therefore they use signed int instead of the
    // unsigned std::size_t.

    for (int k = n_tiles - 1; k >= 0; k--)
    {
        auto [cublas, stream] = gpu.next_cublas_handle();

        // TRSM: Solve L^T * x = a
        ft_rhs[k] = hpx::dataflow(&trsv, cublas, stream, ft_tiles[k * n_tiles + k], ft_rhs[k], n_tile_size, Blas_trans);

        for (int m = k - 1; m >= 0; m--)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // GEMV: b = b - A^T * a
            ft_rhs[m] = hpx::dataflow(
                &gemv,
                cublas,
                stream,
                ft_tiles[k * n_tiles + m],
                ft_rhs[k],
                ft_rhs[m],
                n_tile_size,
                n_tile_size,
                Blas_substract,
                Blas_trans);
        }
    }
}

void forward_solve_tiled_matrix(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    CUDA_GPU &gpu)
{
    for (std::size_t c = 0; c < m_tiles; ++c)
    {
        for (std::size_t k = 0; k < n_tiles; ++k)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // TRSM: solve L * X = A
            ft_rhs[k * m_tiles + c] = hpx::dataflow(
                &trsm,
                cublas,
                stream,
                ft_tiles[k * n_tiles + k],
                ft_rhs[k * m_tiles + c],
                n_tile_size,
                m_tile_size,
                Blas_no_trans,
                Blas_left);

            for (std::size_t m = k + 1; m < n_tiles; ++m)
            {
                auto [cublas, stream] = gpu.next_cublas_handle();

                // GEMM: C = C - A * B
                ft_rhs[m * m_tiles + c] = hpx::dataflow(
                    &gemm,
                    cublas,
                    stream,
                    ft_tiles[m * n_tiles + k],
                    ft_rhs[k * m_tiles + c],
                    ft_rhs[m * m_tiles + c],
                    n_tile_size,
                    m_tile_size,
                    n_tile_size,
                    Blas_no_trans,
                    Blas_no_trans);
            }
        }
    }
}

void backward_solve_tiled_matrix(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    CUDA_GPU &gpu)
{
    for (std::size_t c = 0; c < m_tiles; ++c)
    {
        for (std::size_t k = 0; k < n_tiles; ++k)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // TRSM: solve L^T * X = A
            ft_rhs[k * m_tiles + c] = hpx::dataflow(
                &trsm,
                cublas,
                stream,
                ft_tiles[k * n_tiles + k],
                ft_rhs[k * m_tiles + c],
                n_tile_size,
                m_tile_size,
                Blas_trans,
                Blas_left);

            for (std::size_t m = 0; m < k; ++m)
            {
                auto [cublas, stream] = gpu.next_cublas_handle();

                // GEMM: C = C - A^T * B
                ft_rhs[m * m_tiles + c] = hpx::dataflow(
                    &gemm,
                    cublas,
                    stream,
                    ft_tiles[k * n_tiles + m],
                    ft_rhs[k * m_tiles + c],
                    ft_rhs[m * m_tiles + c],
                    n_tile_size,
                    m_tile_size,
                    n_tile_size,
                    Blas_trans,
                    Blas_no_trans);
            }
        }
    }
}

void matrix_vector_tiled(std::vector<hpx::shared_future<double *>> &ft_tiles,
                         std::vector<hpx::shared_future<double *>> &ft_vector,
                         std::vector<hpx::shared_future<double *>> &ft_rhs,
                         const std::size_t N_row,
                         const std::size_t N_col,
                         const std::size_t n_tiles,
                         const std::size_t m_tiles,
                         CUDA_GPU &gpu)
{
    for (std::size_t k = 0; k < m_tiles; ++k)
    {
        for (std::size_t m = 0; m < n_tiles; ++m)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            ft_rhs[k] = hpx::dataflow(
                &gemv,
                cublas,
                stream,
                ft_tiles[k * n_tiles + m],
                ft_vector[m],
                ft_rhs[k],
                N_row,
                N_col,
                Blas_add,
                Blas_no_trans);
        }
    }
}

void symmetric_matrix_matrix_diagonal_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<double *>> &ft_inter_tiles,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    CUDA_GPU &gpu)
{
    for (std::size_t i = 0; i < m_tiles; ++i)
    {
        for (std::size_t n = 0; n < n_tiles; ++n)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // Compute inner product to obtain diagonal elements of
            // (K_MxN * (K^-1_NxN * K_NxM))
            ft_inter_tiles[i] = hpx::dataflow(
                &dot_diag_syrk,
                cublas,
                stream,
                ft_tCC_tiles[n * m_tiles + i],
                ft_inter_tiles[i],
                n_tile_size,
                m_tile_size);
        }
    }
}

void compute_gemm_of_invK_y(std::vector<hpx::shared_future<double *>> &ft_invK,
                            std::vector<hpx::shared_future<double *>> &ft_y,
                            std::vector<hpx::shared_future<double *>> &ft_alpha,
                            const std::size_t n_tile_size,
                            const std::size_t n_tiles,
                            CUDA_GPU &gpu)
{
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            ft_alpha[i] = hpx::dataflow(
                &gemv,
                cublas,
                stream,
                ft_invK[i * n_tiles + j],
                ft_y[j],
                ft_alpha[i],
                n_tile_size,
                n_tile_size,
                Blas_add,
                Blas_no_trans);
        }
    }
}

hpx::shared_future<double> compute_loss_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_alpha,
    std::vector<hpx::shared_future<double *>> &ft_y,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double>> loss_tiled(n_tiles);

    for (std::size_t k = 0; k < n_tiles; k++)
    {
        loss_tiled[k] =
            hpx::dataflow(&compute_loss, ft_tiles[k * n_tiles + k], ft_alpha[k], ft_y[k], n_tile_size, std::ref(gpu));
    }

    return hpx::dataflow(&add_losses, loss_tiled, n_tile_size, n_tiles);
}

void symmetric_matrix_matrix_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    CUDA_GPU &gpu)
{
    for (std::size_t c = 0; c < m_tiles; ++c)
    {
        for (std::size_t k = 0; k < m_tiles; ++k)
        {
            for (std::size_t m = 0; m < n_tiles; ++m)
            {
                auto [cublas, stream] = gpu.next_cublas_handle();

                // GEMM:  C = C - A^T * B
                ft_priorK[c * m_tiles + k] = hpx::dataflow(
                    &gemm,
                    cublas,
                    stream,
                    ft_tCC_tiles[m * m_tiles + c],
                    ft_tCC_tiles[m * m_tiles + k],
                    ft_priorK[c * m_tiles + k],
                    n_tile_size,
                    m_tile_size,
                    m_tile_size,
                    Blas_trans,
                    Blas_no_trans);
            }
        }
    }
}

void vector_difference_tiled(std::vector<hpx::shared_future<double *>> &ft_priorK,
                             std::vector<hpx::shared_future<double *>> &ft_inter,
                             std::vector<hpx::shared_future<double *>> &ft_vector,
                             const std::size_t m_tile_size,
                             const std::size_t m_tiles,
                             CUDA_GPU &gpu)
{
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_vector[i] = hpx::dataflow(&diag_posterior, ft_priorK[i], ft_inter[i], m_tile_size, std::ref(gpu));
    }
}

void matrix_diagonal_tiled(std::vector<hpx::shared_future<double *>> &ft_priorK,
                           std::vector<hpx::shared_future<double *>> &ft_vector,
                           const std::size_t m_tile_size,
                           const std::size_t m_tiles,
                           CUDA_GPU &gpu)
{
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_vector[i] = hpx::dataflow(&diag_tile, ft_priorK[i * m_tiles + i], m_tile_size, std::ref(gpu));
    }
}

void update_grad_K_tiled_mkl(std::vector<hpx::shared_future<double *>> &ft_tiles,
                             const std::vector<hpx::shared_future<double *>> &ft_v1,
                             const std::vector<hpx::shared_future<double *>> &ft_v2,
                             const std::size_t n_tile_size,
                             const std::size_t n_tiles,
                             CUDA_GPU &gpu)
{
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            ft_tiles[i * n_tiles + j] =
                hpx::dataflow(&ger, cublas, stream, ft_tiles[i * n_tiles + j], ft_v1[i], ft_v2[j], n_tile_size);
        }
    }
}

static double update_hyperparameter(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    double &hyperparameter,  // lengthscale or vertical-lengthscale
    SEKParams sek_params,
    AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    int param_idx,  // 0 for lengthscale, 1 for vertical-lengthscale
    CUDA_GPU &gpu)
{
    throw std::logic_error("Function not implemented for GPU");
    // return 0;
}

double update_lengthscale(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    SEKParams sek_params,
    AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    CUDA_GPU &gpu)
{
    return update_hyperparameter(
        ft_invK,
        ft_gradparam,
        ft_alpha,
        sek_params.lengthscale,
        sek_params,
        adam_params,
        n_tile_size,
        n_tiles,
        m_T,
        v_T,
        beta1_T,
        beta2_T,
        iter,
        0,
        gpu);
}

double update_vertical_lengthscale(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    SEKParams sek_params,
    AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    CUDA_GPU &gpu)
{
    return update_hyperparameter(
        ft_invK,
        ft_gradparam,
        ft_alpha,
        sek_params.vertical_lengthscale,
        sek_params,
        adam_params,
        n_tile_size,
        n_tiles,
        m_T,
        v_T,
        beta1_T,
        beta2_T,
        iter,
        1,
        gpu);
}

double update_noise_variance(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    SEKParams sek_params,
    AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    CUDA_GPU &gpu)
{
    throw std::logic_error("Function not implemented for GPU");
    // return 0;
}

}  // end of namespace gpu

GPRAT_NS_END
