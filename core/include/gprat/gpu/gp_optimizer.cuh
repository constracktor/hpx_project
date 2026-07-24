#ifndef GPRAT_GPU_GP_OPTIMIZER_HPP
#define GPRAT_GPU_GP_OPTIMIZER_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include "gprat/hyperparameters.hpp"
#include "gprat/kernels.hpp"
#include "gprat/target.hpp"

#include <hpx/future.hpp>
#include <vector>

GPRAT_NS_BEGIN

namespace gpu
{

/**
 * @brief Transform hyperparameter to enforce constraints using softplus.
 *
 * @param parameter The parameter to constrain
 * @param noise A flag to apply noise
 *
 * @return The constrained parameter
 */
double to_constrained(const double parameter, bool noise);

/**
 * @brief Transform hyperparmeter to entire real line using inverse of softplus.
 *
 * Optimizers, such as gradient descent or Adam, work better with unconstrained parameters.
 *
 * @param parameter The parameter to constrain
 * @param noise A flag to apply noise
 *
 * @return The unconstrained parameter
 */
double to_unconstrained(const double parameter, bool noise);

/**
 * @brief Calculate the sigmoid function for a given value
 *
 * @param parameter The parameter to input into the function
 *
 * @return The sigmoid value for the given parameter
 */
double compute_sigmoid(const double parameter);

/**
 * @brief Compute the distance between two feature vectors divided by the lengthscale
 *
 * @param i_global The global index of the first feature vector
 * @param j_global The global index of the second feature vector
 * @param n_regressors The number of regressors
 * @param hyperparameters The kernel hyperparameters
 * @param i_input The first feature vector
 * @param j_input The second feature vector
 *
 * @return The distance between two features at position i_global,j_global
 */
double compute_covariance_distance(std::size_t i_global,
                                   std::size_t j_global,
                                   std::size_t n_regressors,
                                   SEKParams sek_params,
                                   const std::vector<double> &i_input,
                                   const std::vector<double> &j_input);

/**
 * @brief Generate a tile of distances divided by the lengthscale
 *
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param n_regressors The number of regressors
 * @param hyperparameters The kernel hyperparameters
 * @param input The input data vector
 *
 * @return A quadratic tile containing the distance between the features of size N x N
 */
std::vector<double> gen_tile_distance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    SEKParams sek_params,
    const std::vector<double> &input);

/**
 * @brief Generate a tile of the covariance matrix with given distances
 *
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param hyperparameters The kernel hyperparameters
 * @param cov_dists The pre-computed distances for the tile
 *
 * @return A quadratic tile of the covariance matrix of size N x N
 */
std::vector<double> gen_tile_covariance_with_distance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    SEKParams sek_params,
    const std::vector<double> &cov_dists);

/**
 * @brief  Generate a derivative tile w.r.t. vertical_lengthscale v
 *
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 * @param cov_dists The pre-computed distances for the tile
 *
 * @return A quadratic tile of the derivative of v of size N x N
 */
std::vector<double>
gen_tile_grad_v(std::size_t row,
                std::size_t col,
                std::size_t N,
                std::size_t n_regressors,
                SEKParams sek_params,
                const std::vector<double> &cov_dists);

/**
 * @brief  Generate a derivative tile w.r.t. lengthscale l
 *
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 * @param cov_dists The pre-computed distances for the tile
 *
 * @return A quadratic tile of the derivative of l of size N x N
 */
std::vector<double>
gen_tile_grad_l(std::size_t row,
                std::size_t col,
                std::size_t N,
                std::size_t n_regressors,
                SEKParams sek_params,
                const std::vector<double> &cov_dists);

/**
 * @brief Generate a derivative tile w.r.t. lengthscale.
 *
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param grad_l_tile The gradient of the left side
 *
 * @return A quadratic tile of the derivative of l of size N x N
 */
std::vector<double> gen_tile_grad_v_trans(std::size_t N, const std::vector<double> &grad_l_tile);

/**
 * @brief Generate a derivative tile w.r.t. lengthscale.
 *
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param f_grad_l_tile The gradient of the left side
 * @param gpu The GPU target for computations
 *
 * @return A quadratic tile of the derivative of l of size N x N
 */
hpx::shared_future<double *>
gen_tile_grad_l_trans(std::size_t N, const hpx::shared_future<double *> f_grad_l_tile, CUDA_GPU &gpu);

/**
 * @brief Compute hyper-parameter beta_1 or beta_2 to power t.
 *
 * @param t The iteration number
 * @param beta The hyper-parameter
 *
 * @return The hyper-parameter to the power of t
 */
double gen_beta_T(int t, double beta);

/**
 * @brief Compute negative-log likelihood on one tile.
 *
 * @param K_diag_tile The Cholesky factor L (in a diagonal tile)
 * @param alpha_tile The tiled solution of K * alpha = y
 * @param y_tile The output tile
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param gpu The GPU target for computations
 *
 * @return Return l = y^T * alpha + \sum_i^N log(L_ii^2)
 */
hpx::shared_future<double>
compute_loss(const hpx::shared_future<double *> &K_diag_tile,
             const hpx::shared_future<double *> &alpha_tile,
             const hpx::shared_future<double *> &y_tile,
             std::size_t N,
             CUDA_GPU &gpu);

/**
 * @brief Add up negative-log likelihood loss for all tiles.
 *
 * @param losses A vector contianing the loss per tile
 * @param n_tile_size The size of a tile
 * @param n_tiles The number of tiles
 *
 * @return The added up loss plus the constant factor
 */
hpx::shared_future<double>
add_losses(const std::vector<hpx::shared_future<double>> &losses, std::size_t n_tile_size, std::size_t n_tiles);

/**
 * @brief Compute the loss gradient.
 *
 * @param grad_l The gradient of the left side
 * @param grad_r The gradient of the right side
 * @param N The size of a tile
 * @param n_tiles The number of tiles
 *
 * @return The added up loss plus the constant factor
 */
double compute_gradient(const double &grad_l, const double &grad_r, std::size_t N, std::size_t n_tiles);

/**
 * @brief Compute trace for noise variance.
 *
 * Same function as compute_trace with() the only difference that we only use
 * diag tiles multiplied by derivative of noise_variance.
 */
double compute_gradient_noise(
    const std::vector<std::vector<double>> &ft_tiles, double noise_variance, std::size_t N, std::size_t n_tiles);

/**
 * @brief Update biased first raw moment estimate: m_T+1 = beta_1 * m_T + (1 - beta_1) * g_T.
 *
 * @param gradient The value of the computed gradient
 * @param m_T The previous first moment
 * @param beta_1 The decay rate of the first moment
 *
 * @return The updated first moment
 */
double update_first_moment(const double &gradient, double m_T, const double &beta_1);

/**
 * @brief Update biased second raw moment estimate: v_T+1 = beta_2 + v_T + (1 - beta_2) * g_T^2
 *
 * @param gradient The value of the computed gradient
 * @param v_T The previous second moment
 * @param beta_2 The decay rate of the second moment
 *
 * @return The updated second moment
 */
double update_second_moment(const double &gradient, double v_T, const double &beta_2);

/**
 * @brief Update the hyperparameter using Adam optimizer.
 *
 * @param unconstrained_hyperparam The unconstrained hyperparameter
 * @param sek_params The kernel hyperparameters
 * @param adam_params The Adam optimizer hyperparameters
 * @param m_T The first moment
 * @param v_T The second moment
 * @param beta1_T The decay rate of the first moment
 * @param beta2_T The decay rate of the second moment
 * @param iter The number of iterations
 *
 * @return The updated hyperparameter
 */
hpx::shared_future<double>
update_param(const double unconstrained_hyperparam,
             SEKParams sek_params,
             AdamParams adam_params,
             double m_T,
             double v_T,
             const std::vector<double> beta1_T,
             const std::vector<double> beta2_T,
             int iter);

/**
 * @brief Generate an identity tile if i==j.
 */
std::vector<double> gen_tile_identity(std::size_t row, std::size_t col, std::size_t N);

/**
 * @brief Generate an empty tile NxN.
 */
std::vector<double> gen_tile_zeros_diag(std::size_t N);

/**
 * @brief return zero - used to initialize moment vectors
 */
double gen_moment();

/**
 * @brief Sum up the gradient for the left side.
 *
 * @param diagonal The diagonal elements of the covariance matrix
 * @param grad The gradient
 *
 * @return The sum of the gradient
 */
double sum_gradleft(const std::vector<double> &diagonal, double grad);

/**
 * @brief Sum up the gradient for the left side.
 *
 * @param inter_alpha The alpha vector
 * @param alpha The alpha vector
 * @param grad The gradient
 * @param N The size of a tile
 *
 * @return The sum of the gradient
 */
double
sum_gradright(const std::vector<double> &inter_alpha, const std::vector<double> &alpha, double grad, std::size_t N);

/**
 * @brief Sum up the noise gradient for the left side.
 *
 * @param ft_invK The inverse of the covariance matrix
 * @param grad The gradient
 * @param sek_params The kernel hyperparameters
 * @param N The size of a tile
 * @param n_tiles The number of tiles
 *
 * @return The sum of the noise gradient
 */
double sum_noise_gradleft(const std::vector<double> &ft_invK,
                          double grad,
                          SEKParams sek_params,
                          std::size_t N,
                          std::size_t n_tiles);

/**
 * @brief Sum up the noise gradient for the right side.
 *
 * @param alpha The alpha vector
 * @param grad The gradient
 * @param sek_params The kernel hyperparameters
 * @param N The size of a tile
 *
 * @return The sum of the noise gradient
 */
double
sum_noise_gradright(const std::vector<double> &alpha, double grad, SEKParams sek_params, std::size_t N);

}  // end of namespace gpu

GPRAT_NS_END

#endif
