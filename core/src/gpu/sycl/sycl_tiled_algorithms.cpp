#include "gpu/sycl/sycl_tiled_algorithms.hpp"

#include "gpu/sycl/adapter_onemath.hpp"
#include "gpu/sycl/sycl_gp_optimizer.hpp"
#include "gpu/sycl/sycl_gp_uncertainty.hpp"
#include <hpx/algorithm.hpp>
#include <mutex>

namespace gprat::sycl_backend
{

// Tiled Cholesky Algorithm

void right_looking_cholesky_tiled(std::vector<hpx::shared_future<double *>> &ft_tiles,
                                  const std::size_t n_tile_size,
                                  const std::size_t n_tiles,
                                  gprat::SYCL_DEVICE &sycl_device)
{
    double *result;

    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        result = potrf(sycl_device.next_queue(), ft_tiles[k * n_tiles + k].get(), n_tile_size);
        ft_tiles[k * n_tiles + k] = hpx::make_ready_future(result);

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            result = trsm(sycl_device.next_queue(),
                          ft_tiles[k * n_tiles + k].get(),
                          ft_tiles[m * n_tiles + k].get(),
                          n_tile_size,
                          n_tile_size,
                          oneapi::math::transpose::trans,
                          oneapi::math::side::right);

            ft_tiles[m * n_tiles + k] = hpx::make_ready_future(result);
        }

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            result = syrk(sycl_device.next_queue(),
                          ft_tiles[m * n_tiles + k].get(),
                          ft_tiles[m * n_tiles + m].get(),
                          n_tile_size);

            ft_tiles[m * n_tiles + m] = hpx::make_ready_future(result);

            for (std::size_t n = k + 1; n < m; ++n)
            {
                result = gemm(
                    sycl_device.next_queue(),
                    ft_tiles[m * n_tiles + k].get(),
                    ft_tiles[n * n_tiles + k].get(),
                    ft_tiles[m * n_tiles + n].get(),
                    n_tile_size,
                    n_tile_size,
                    n_tile_size,
                    oneapi::math::transpose::nontrans,
                    oneapi::math::transpose::trans);

                ft_tiles[m * n_tiles + n] = hpx::make_ready_future(result);
            }
        }
    }
}

// Tiled Triangular Solve Algorithms

void forward_solve_tiled(std::vector<hpx::shared_future<double *>> &ft_tiles,
                         std::vector<hpx::shared_future<double *>> &ft_rhs,
                         const std::size_t n_tile_size,
                         const std::size_t n_tiles,
                         gprat::SYCL_DEVICE &sycl_device)
{
    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        // TRSM: Solve L * x = a
        double *result =
            trsv(sycl_device.next_queue(),
                 ft_tiles[k * n_tiles + k].get(),
                 ft_rhs[k].get(),
                 n_tile_size,
                 oneapi::math::transpose::nontrans);

        ft_rhs[k] = hpx::make_ready_future(result);

        double *x_k = result;
        auto gemv_queue = sycl_device.next_queue();

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            // GEMV: b = b - A * x_k
            result = gemv(gemv_queue,
                          ft_tiles[m * n_tiles + k].get(),
                          x_k,
                          ft_rhs[m].get(),
                          n_tile_size,
                          n_tile_size,
                          -1,
                          oneapi::math::transpose::nontrans);

            ft_rhs[m] = hpx::make_ready_future(result);
        }
    }
}

void backward_solve_tiled(std::vector<hpx::shared_future<double *>> &ft_tiles,
                          std::vector<hpx::shared_future<double *>> &ft_rhs,
                          const std::size_t n_tile_size,
                          const std::size_t n_tiles,
                          gprat::SYCL_DEVICE &sycl_device)
{
    double *result;
    // NOTE: The loops traverse backwards. Its last comparisons require the
    // usage negative numbers. Therefore they use signed int instead of the
    // unsigned std::size_t.

    for (int k = static_cast<int>(n_tiles) - 1; k >= 0; k--)
    {
        // TRSM: Solve L^T * x = a
        result = trsv(sycl_device.next_queue(),
                      ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)].get(),
                      ft_rhs[static_cast<std::size_t>(k)].get(),
                      n_tile_size,
                      oneapi::math::transpose::trans);

        ft_rhs[static_cast<std::size_t>(k)] = hpx::make_ready_future(result);

        for (int m = k - 1; m >= 0; m--)
        {
            // GEMV: b = b - A^T * a
            result = gemv(sycl_device.next_queue(),
                          ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(m)].get(),
                          ft_rhs[static_cast<std::size_t>(k)].get(),
                          ft_rhs[static_cast<std::size_t>(m)].get(),
                          n_tile_size,
                          n_tile_size,
                          -1,
                          oneapi::math::transpose::trans);

            ft_rhs[static_cast<std::size_t>(m)] = hpx::make_ready_future(result);
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
    gprat::SYCL_DEVICE &sycl_device)
{
    double *result;
    double *result_gemm;

    for (std::size_t c = 0; c < m_tiles; ++c)
    {
        for (std::size_t k = 0; k < n_tiles; ++k)
        {
            // TRSM: solve L * X = A
            result = trsm(sycl_device.next_queue(),
                          ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)].get(),
                          ft_rhs[static_cast<std::size_t>(k * m_tiles + c)].get(),
                          n_tile_size,
                          m_tile_size,
                          oneapi::math::transpose::nontrans,
                          oneapi::math::side::left);

            ft_rhs[static_cast<std::size_t>(k * m_tiles + c)] = hpx::make_ready_future(result);

            for (std::size_t m = k + 1; m < n_tiles; ++m)
            {
                // GEMM: C = C - A * B
                result_gemm = gemm(
                    sycl_device.next_queue(),
                    ft_tiles[m * n_tiles + k].get(),
                    ft_rhs[static_cast<std::size_t>(k * m_tiles + c)].get(),
                    ft_rhs[m * m_tiles + c].get(),
                    n_tile_size,
                    m_tile_size,
                    n_tile_size,
                    oneapi::math::transpose::nontrans,
                    oneapi::math::transpose::nontrans);

                ft_rhs[m * m_tiles + c] = hpx::make_ready_future(result_gemm);
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
    gprat::SYCL_DEVICE &sycl_device)
{
    double *result;

    for (std::size_t c = 0; c < m_tiles; ++c)
    {
        for (std::size_t k = 0; k < n_tiles; ++k)
        {
            // TRSM: solve L^T * X = A
            result = trsm(sycl_device.next_queue(),
                          ft_tiles[static_cast<std::size_t>(k) * n_tiles + static_cast<std::size_t>(k)].get(),
                          ft_rhs[static_cast<std::size_t>(k * m_tiles + c)].get(),
                          n_tile_size,
                          m_tile_size,
                          oneapi::math::transpose::trans,
                          oneapi::math::side::left);

            ft_rhs[static_cast<std::size_t>(k * m_tiles + c)] = hpx::make_ready_future(result);

            for (std::size_t m = 0; m < k; ++m)
            {
                // GEMM: C = C - A^T * B
                result = gemm(
                    sycl_device.next_queue(),
                    ft_tiles[k * n_tiles + m].get(),
                    ft_rhs[static_cast<std::size_t>(k * m_tiles + c)].get(),
                    ft_rhs[m * m_tiles + c].get(),
                    n_tile_size,
                    m_tile_size,
                    n_tile_size,
                    oneapi::math::transpose::trans,
                    oneapi::math::transpose::nontrans);

                ft_rhs[m * m_tiles + c] = hpx::make_ready_future(result);
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
                         gprat::SYCL_DEVICE &sycl_device)
{
    double *result;

    for (std::size_t k = 0; k < m_tiles; ++k)
    {
        for (std::size_t m = 0; m < n_tiles; ++m)
        {
            result = gemv(sycl_device.next_queue(),
                          ft_tiles[k * n_tiles + m].get(),
                          ft_vector[m].get(),
                          ft_rhs[k].get(),
                          N_row,
                          N_col,
                          1,
                          oneapi::math::transpose::nontrans);

            ft_rhs[k] = hpx::make_ready_future(result);
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
    gprat::SYCL_DEVICE &sycl_device)
{
    double *result;

    for (std::size_t i = 0; i < m_tiles; ++i)
    {
        for (std::size_t n = 0; n < n_tiles; ++n)
        {
            // Compute inner product to obtain diagonal elements of
            // (K_MxN * (K^-1_NxN * K_NxM))
            result = dot_diag_syrk(sycl_device.next_queue(),
                                   ft_tCC_tiles[n * m_tiles + i].get(),
                                   ft_inter_tiles[i].get(),
                                   n_tile_size,
                                   m_tile_size);

            ft_inter_tiles[i] = hpx::make_ready_future(result);
        }
    }
}

void compute_gemm_of_invK_y(std::vector<hpx::shared_future<double *>> &ft_invK,
                            std::vector<hpx::shared_future<double *>> &ft_y,
                            std::vector<hpx::shared_future<double *>> &ft_alpha,
                            const std::size_t n_tile_size,
                            const std::size_t n_tiles,
                            gprat::SYCL_DEVICE &sycl_device)
{
    double *result;

    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            result = gemv(sycl_device.next_queue(),
                          ft_invK[i * n_tiles + j].get(),
                          ft_y[j].get(),
                          ft_alpha[i].get(),
                          n_tile_size,
                          n_tile_size,
                          1,
                          oneapi::math::transpose::nontrans);

            ft_alpha[i] = hpx::make_ready_future(result);
        }
    }
}

hpx::shared_future<double> compute_loss_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_alpha,
    std::vector<hpx::shared_future<double *>> &ft_y,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gprat::SYCL_DEVICE &sycl_device)
{
    double total_loss = 0.0;
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        total_loss += compute_loss(ft_tiles[k * n_tiles + k], ft_alpha[k], ft_y[k], n_tile_size, sycl_device);
    }
    total_loss += static_cast<double>(n_tile_size) * static_cast<double>(n_tiles) * std::log(2.0 * M_PI);
    return hpx::make_ready_future(0.5 * total_loss / static_cast<double>(n_tile_size * n_tiles));
}

void symmetric_matrix_matrix_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gprat::SYCL_DEVICE &sycl_device)
{
    double *result;

    for (std::size_t c = 0; c < m_tiles; ++c)
    {
        for (std::size_t k = 0; k < m_tiles; ++k)
        {
            for (std::size_t m = 0; m < n_tiles; ++m)
            {
                // GEMM:  C = C - A^T * B
                result = gemm(
                    sycl_device.next_queue(),
                    ft_tCC_tiles[m * m_tiles + c].get(),
                    ft_tCC_tiles[m * m_tiles + k].get(),
                    ft_priorK[c * m_tiles + k].get(),
                    n_tile_size,
                    m_tile_size,
                    m_tile_size,
                    oneapi::math::transpose::trans,
                    oneapi::math::transpose::nontrans);

                ft_priorK[c * m_tiles + k] = hpx::make_ready_future(result);
            }
        }
    }
}

void vector_difference_tiled(std::vector<hpx::shared_future<double *>> &ft_priorK,
                             std::vector<hpx::shared_future<double *>> &ft_inter,
                             std::vector<hpx::shared_future<double *>> &ft_vector,
                             const std::size_t m_tile_size,
                             const std::size_t m_tiles)
{
#ifdef GPRAT_SYCL_INTEL_GPU
    // First-ever JIT compile of a kernel crashes on Intel Level-Zero drivers
    // (Arc B580) if it happens amid the concurrent dataflow below, so warm it
    // up synchronously once. Not needed on NVIDIA/AMD backends.
    static std::once_flag warm_up_flag;
    std::call_once(warm_up_flag,
                   []()
                   {
                       sycl::queue queue(sycl::gpu_selector_v);
                       double *dummy = sycl::malloc_device<double>(1, queue);
                       double *result = diag_posterior(dummy, dummy, 1);
                       sycl::free(dummy, queue);
                       sycl::free(result, queue);
                   });
#endif

    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_vector[i] = hpx::dataflow(hpx::unwrapping(&diag_posterior), ft_priorK[i], ft_inter[i], m_tile_size);
    }
}

void matrix_diagonal_tiled(std::vector<hpx::shared_future<double *>> &ft_priorK,
                           std::vector<hpx::shared_future<double *>> &ft_vector,
                           const std::size_t m_tile_size,
                           const std::size_t m_tiles)
{
#ifdef GPRAT_SYCL_INTEL_GPU
    // See vector_difference_tiled.
    static std::once_flag warm_up_flag;
    std::call_once(warm_up_flag,
                   []()
                   {
                       sycl::queue queue(sycl::gpu_selector_v);
                       double *dummy = sycl::malloc_device<double>(1, queue);
                       double *result = diag_tile(dummy, 1);
                       sycl::free(dummy, queue);
                       sycl::free(result, queue);
                   });
#endif

    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_vector[i] = hpx::dataflow(hpx::unwrapping(&diag_tile), ft_priorK[i * m_tiles + i], m_tile_size);
    }
}

void update_grad_K_tiled_mkl(std::vector<hpx::shared_future<double *>> &ft_tiles,
                             const std::vector<hpx::shared_future<double *>> &ft_v1,
                             const std::vector<hpx::shared_future<double *>> &ft_v2,
                             const std::size_t n_tile_size,
                             const std::size_t n_tiles,
                             gprat::SYCL_DEVICE &sycl_device)
{
    double *result;

    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            result = ger(
                sycl_device.next_queue(), ft_tiles[i * n_tiles + j].get(), ft_v1[i].get(), ft_v2[j].get(), n_tile_size);

            ft_tiles[i * n_tiles + j] = hpx::make_ready_future(result);
        }
    }
}

static double update_hyperparameter(
    const std::vector<hpx::shared_future<double *>> & /*ft_invK*/,
    const std::vector<hpx::shared_future<double *>> & /*ft_gradparam*/,
    const std::vector<hpx::shared_future<double *>> & /*ft_alpha*/,
    double & /*hyperparameter*/,
    gprat::SEKParams /*sek_params*/,
    gprat::AdamParams /*adam_params*/,
    const std::size_t /*n_tile_size*/,
    const std::size_t /*n_tiles*/,
    std::vector<hpx::shared_future<double>> & /*m_T*/,
    std::vector<hpx::shared_future<double>> & /*v_T*/,
    const std::vector<hpx::shared_future<double>> & /*beta1_T*/,
    const std::vector<hpx::shared_future<double>> & /*beta2_T*/,
    int /*iter*/,
    int /*param_idx*/,
    gprat::SYCL_DEVICE & /*sycl_device*/)
{
    throw std::logic_error("Function not implemented for GPU");
}

double update_lengthscale(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    gprat::SEKParams sek_params,
    gprat::AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    gprat::SYCL_DEVICE &sycl_device)
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
        sycl_device);
}

double update_vertical_lengthscale(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    gprat::SEKParams sek_params,
    gprat::AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    gprat::SYCL_DEVICE &sycl_device)
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
        sycl_device);
}

double update_noise_variance(
    const std::vector<hpx::shared_future<double *>> & /*ft_invK*/,
    const std::vector<hpx::shared_future<double *>> & /*ft_alpha*/,
    gprat::SEKParams /*sek_params*/,
    gprat::AdamParams /*adam_params*/,
    const std::size_t /*n_tile_size*/,
    const std::size_t /*n_tiles*/,
    std::vector<hpx::shared_future<double>> & /*m_T*/,
    std::vector<hpx::shared_future<double>> & /*v_T*/,
    const std::vector<hpx::shared_future<double>> & /*beta1_T*/,
    const std::vector<hpx::shared_future<double>> & /*beta2_T*/,
    int /*iter*/,
    gprat::SYCL_DEVICE & /*sycl_device*/)
{
    throw std::logic_error("Function not implemented for GPU");
}

}  // namespace gprat::sycl_backend
