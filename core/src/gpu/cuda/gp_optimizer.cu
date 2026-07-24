#include "gprat/gpu/gp_optimizer.cuh"

#include "gprat/gpu/adapter_cublas.cuh"
#include "gprat/gpu/cuda_kernels.cuh"
#include "gprat/gpu/cuda_utils.cuh"

#include <numbers>

GPRAT_NS_BEGIN

namespace gpu
{

double to_constrained(const double parameter, bool noise)
{
    if (noise)
    {
        return log(1.0 + exp(parameter)) + 1e-6;
    }
    else
    {
        return log(1.0 + exp(parameter));
    }
}

double to_unconstrained(const double parameter, bool noise)
{
    if (noise)
    {
        return log(exp(parameter - 1e-6) - 1.0);
    }
    else
    {
        return log(exp(parameter) - 1.0);
    }
}

double compute_sigmoid(const double parameter) { return 1.0 / (1.0 + exp(-parameter)); }

double compute_covariance_distance(std::size_t i_global,
                                   std::size_t j_global,
                                   std::size_t n_regressors,
                                   SEKParams sek_params,
                                   const std::vector<double> &i_input,
                                   const std::vector<double> &j_input)
{
    // C(z_i,z_j) = vertical_lengthscale * exp(-0.5*lengthscale*(z_i-z_j)^2)
    double distance = 0.0;
    double z_ik_minus_z_jk;

    for (std::size_t k = 0; k < n_regressors; ++k)
    {
        z_ik_minus_z_jk = i_input[i_global + k] - j_input[j_global + k];
        distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
    }

    return -1.0 / (2.0 * pow(sek_params.lengthscale, 2.0)) * distance;
}

std::vector<double> gen_tile_distance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    SEKParams sek_params,
    const std::vector<double> &input)
{
    std::size_t i_global, j_global;
    double cov_dist;
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            j_global = N * col + j;
            // compute covariance function
            cov_dist = compute_covariance_distance(i_global, j_global, n_regressors, sek_params, input, input);
            tile[i * N + j] = cov_dist;
        }
    }
    return std::move(tile);
}

std::vector<double> gen_tile_covariance_with_distance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    SEKParams sek_params,
    const std::vector<double> &cov_dists)
{
    std::size_t i_global, j_global;
    double covariance;
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            j_global = N * col + j;
            // compute covariance function
            covariance = sek_params.vertical_lengthscale * exp(cov_dists[i * N + j]);
            if (i_global == j_global)
            {
                // noise variance on diagonal
                covariance += sek_params.noise_variance;
            }
            tile[i * N + j] = covariance;
        }
    }
    return std::move(tile);
}

std::vector<double>
gen_tile_grad_v(std::size_t row,
                std::size_t col,
                std::size_t N,
                std::size_t n_regressors,
                SEKParams sek_params,
                const std::vector<double> &cov_dists)
{
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N * N);
    double hyperparam_der = compute_sigmoid(to_unconstrained(sek_params.vertical_lengthscale, false));
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            // compute covariance function
            tile[i * N + j] = exp(cov_dists[i * N + j]) * hyperparam_der;
        }
    }
    return std::move(tile);
}

std::vector<double>
gen_tile_grad_l(std::size_t row,
                std::size_t col,
                std::size_t N,
                std::size_t n_regressors,
                SEKParams sek_params,
                const std::vector<double> &cov_dists)
{
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N * N);
    double hyperparam_der = compute_sigmoid(to_unconstrained(sek_params.lengthscale, false));
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            // compute covariance function
            tile[i * N + j] = -2.0 * (sek_params.vertical_lengthscale / sek_params.lengthscale) * cov_dists[i * N + j]
                              * exp(cov_dists[i * N + j]) * hyperparam_der;
        }
    }
    return std::move(tile);
}

std::vector<double> gen_tile_grad_v_trans(std::size_t N, const std::vector<double> &grad_l_tile)
{
    std::vector<double> transposed;
    transposed.resize(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            // Mapping (i, j) in the original matrix to (j, i) in the transposed
            // matrix
            transposed[j * N + i] = grad_l_tile[i * N + j];
        }
    }
    return std::move(transposed);
}

hpx::shared_future<double *>
gen_tile_grad_l_trans(std::size_t N, const hpx::shared_future<double *> f_grad_l_tile, CUDA_GPU &gpu)
{
    double *transposed;
    check_cuda_error(cudaMalloc(&transposed, N * N * sizeof(double)));
    double *d_grad_l_tile = f_grad_l_tile.get();

    cudaStream_t stream = gpu.next_stream();
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    transpose<<<n_blocks, threads_per_block, 0, stream>>>(transposed, d_grad_l_tile, N, N);

    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(transposed);
}

double gen_beta_T(int t, double beta) { return pow(beta, t); }

__global__ void
add_log_squared_K_diag(double *K_diag_tile, double *alpha_tile, double *y_tile, double *loss, std::size_t N)
{
    for (std::size_t i = 0; i < N; i++)
    {
        *loss += log(K_diag_tile[i * N + i] * K_diag_tile[i * N + i]);
    }
}

hpx::shared_future<double>
compute_loss(const hpx::shared_future<double *> &K_diag_tile,
             const hpx::shared_future<double *> &alpha_tile,
             const hpx::shared_future<double *> &y_tile,
             std::size_t N,
             CUDA_GPU &gpu)
{
    auto [cublas, stream] = gpu.next_cublas_handle();

    hpx::shared_future<double *> d_loss = dot(cublas, stream, y_tile, alpha_tile, N);
    add_log_squared_K_diag<<<1, 1, 0, stream>>>(K_diag_tile.get(), alpha_tile.get(), y_tile.get(), d_loss.get(), N);

    double h_loss;
    check_cuda_error(cudaMemcpyAsync(&h_loss, d_loss.get(), sizeof(double), cudaMemcpyDeviceToHost, stream));
    check_cuda_error(cudaFree(d_loss.get()));
    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(h_loss);
}

hpx::shared_future<double>
add_losses(const std::vector<hpx::shared_future<double>> &losses, std::size_t n_tile_size, std::size_t n_tiles)
{
    // Add the squared difference to the error
    double l = 0.0;
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        l += losses[i].get();
    }
    l += n_tile_size * n_tiles * log(2.0 * std::numbers::pi);

    return hpx::make_ready_future(0.5 * l / (n_tile_size * n_tiles));
}

double compute_gradient(const double &grad_l, const double &grad_r, std::size_t N, std::size_t n_tiles)
{
    double grad = 0.0;
    grad = 1.0 / (2.0 * N * n_tiles) * (grad_l - grad_r);

    return std::move(grad);
}

double compute_gradient_noise(
    const std::vector<std::vector<double>> &ft_tiles, double noise_variance, std::size_t N, std::size_t n_tiles)
{
    // Initialize tile
    double trace = 0.0;
    double hyperparam_der = compute_sigmoid(to_unconstrained(noise_variance, true));
    for (std::size_t d = 0; d < n_tiles; d++)
    {
        auto tile = ft_tiles[d * n_tiles + d];
        for (std::size_t i = 0; i < N; ++i)
        {
            trace += (tile[i * N + i] * hyperparam_der);
        }
    }
    trace = 1.0 / (2.0 * N * n_tiles) * trace;
    return std::move(trace);
}

double update_first_moment(const double &gradient, double m_T, const double &beta_1)
{
    return beta_1 * m_T + (1.0 - beta_1) * gradient;
}

double update_second_moment(const double &gradient, double v_T, const double &beta_2)
{
    return beta_2 * v_T + (1.0 - beta_2) * gradient * gradient;
}

hpx::shared_future<double>
update_param(const double unconstrained_hyperparam,
             SEKParams sek_params,
             AdamParams adam_params,
             double m_T,
             double v_T,
             const std::vector<double> beta1_T,
             const std::vector<double> beta2_T,
             int iter)
{
    double alpha_T = sek_params.noise_variance * sqrt(1.0 - beta2_T[iter]) / (1.0 - beta1_T[iter]);
    return hpx::make_ready_future(unconstrained_hyperparam - alpha_T * m_T / (sqrt(v_T) + adam_params.epsilon));
}

std::vector<double> gen_tile_identity(std::size_t row, std::size_t col, std::size_t N)
{
    std::size_t i_global, j_global;
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N * N);
    std::fill(tile.begin(), tile.end(), 0.0);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = i; j <= i; j++)
        {
            j_global = N * col + j;
            if (i_global == j_global)
            {
                tile[i * N + j] = 1.0;
            }
        }
    }
    return std::move(tile);
}

std::vector<double> gen_tile_zeros_diag(std::size_t N)
{
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N);
    std::fill(tile.begin(), tile.end(), 0.0);
    return std::move(tile);
}

double gen_moment()
{
    double z = 0.0;
    return z;
}

double sum_gradleft(const std::vector<double> &diagonal, double grad)
{
    grad += std::reduce(diagonal.begin(), diagonal.end());
    return grad;
}

double
sum_gradright(const std::vector<double> &inter_alpha, const std::vector<double> &alpha, double grad, std::size_t N)
{
    // grad += dot(inter_alpha, alpha, N);
    // return grad;
    return 0.0;
}

double sum_noise_gradleft(
    const std::vector<double> &ft_invK, double grad, SEKParams sek_params, std::size_t N, std::size_t n_tiles)
{
    double noise_der = compute_sigmoid(to_unconstrained(sek_params.noise_variance, true));
    for (std::size_t i = 0; i < N; ++i)
    {
        grad += (ft_invK[i * N + i] * noise_der);
    }
    return std::move(grad);
}

double sum_noise_gradright(const std::vector<double> &alpha, double grad, SEKParams sek_params, std::size_t N)
{
    // double noise_der =
    //     compute_sigmoid(to_unconstrained(sek_params.noise_variance, true));
    // grad += (noise_der * dot(alpha, alpha, N));
    // return grad;
    return 0.0;
}

}  // end of namespace gpu

GPRAT_NS_END
