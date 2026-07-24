#ifndef GPRAT_CPU_GP_OPTIMIZER_H
#define GPRAT_CPU_GP_OPTIMIZER_H

#pragma once

#include "gprat/detail/config.hpp"
#include "gprat/hyperparameters.hpp"
#include "gprat/kernels.hpp"
#include "gprat/tile_data.hpp"

#include <vector>

GPRAT_NS_BEGIN

namespace cpu
{

/**
 * @brief Transform hyperparameter to enforce constraints using softplus.
 *
 * @param parameter The parameter to constrain
 * @param noise A flag to apply noise
 *
 * @return The constrained parameter
 */
double to_constrained(double parameter, bool noise);

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
double to_unconstrained(double parameter, bool noise);

/**
 * @brief Calculate the sigmoid function for a given value
 *
 * @param parameter The parameter to input into the function
 *
 * @return The sigmoid value for the given parameter
 */
double compute_sigmoid(double parameter);

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
                                   const SEKParams &sek_params,
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
mutable_tile_data<double> gen_tile_distance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
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
mutable_tile_data<double> gen_tile_covariance_with_distance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    const SEKParams &sek_params,
    const const_tile_data<double> &distance);

/**
 * @brief  Generate a derivative tile w.r.t. vertical_lengthscale v
 *
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param hyperparameters The kernel hyperparameters
 * @param cov_dists The pre-computed distances for the tile
 *
 * @return A quadratic tile of the derivative of v of size N x N
 */
mutable_tile_data<double>
gen_tile_grad_v(std::size_t N, const SEKParams &sek_params, const const_tile_data<double> &distance);

/**
 * @brief  Generate a derivative tile w.r.t. lengthscale l
 *
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param hyperparameters The kernel hyperparameters
 * @param cov_dists The pre-computed distances for the tile
 *
 * @return A quadratic tile of the derivative of l of size N x N
 */
mutable_tile_data<double>
gen_tile_grad_l(std::size_t N, const SEKParams &sek_params, const const_tile_data<double> &distance);

/**
 * @brief Update biased first raw moment estimate: m_T+1 = beta_1 * m_T + (1 - beta_1) * g_T.
 *
 * @param gradient The value of the computed gradient
 * @param m_T The previous first moment
 * @param beta_1 The decay rate of the first moment
 *
 * @return The updated first moment
 */
double update_first_moment(double gradient, double m_T, double beta_1);

/**
 * @brief Update biased second raw moment estimate: v_T+1 = beta_2 + v_T + (1 - beta_2) * g_T^2
 *
 * @param gradient The value of the computed gradient
 * @param v_T The previous second moment
 * @param beta_2 The decay rate of the second moment
 *
 * @return The updated second moment
 */
double update_second_moment(double gradient, double v_T, double beta_2);

/**
 * @brief Update hyperparameter using the Adam update.
 *
 * @param The hyperparameter to update
 * @param The Adam optimization parameter
 * @param m_T The first moment
 * @param v_T The second moment
 * @param iter The current iteration
 *
 * @return The updated hyperparameter
 */
double
adam_step(double unconstrained_hyperparam, const AdamParams &adam_params, double m_T, double v_T, std::size_t iter);

/**
 * @brief Compute negative-log likelihood on one tile.
 *
 * @param K_diag_tile The Cholesky factor L (in a diagonal tile)
 * @param alpha_tile The tiled solution of K * alpha = y
 * @param y_tile The output tile
 *
 * @return Return l = y^T * alpha + \sum_i^N log(L_ii^2)
 */
double compute_loss(std::span<const double> K_diag_tile,
                    std::span<const double> alpha_tile,
                    std::span<const double> y_tile,
                    std::size_t N);

/**
 * @brief Add up negative-log likelihood loss for all tiles.
 *
 * @param losses A vector contianing the loss per tile
 * @param N The size of a tile
 * @param n_tiles The number of tiles
 *
 * @return The added up loss plus the constant factor
 */
double add_losses(std::span<const double> losses, std::size_t N, std::size_t n);

/**
 * @brief Compute the loss gradient.
 *
 * @param trace The first part of the gradient: trace(K^-1 * delta(K)/delta(theta_i))
 * @param dot The second part of the gradient:  beta^T * delta(K)/delta(theta_i) * beta
 * @param N The size of a tile
 * @param n_tiles The number of tiles
 *
 * @return The added up loss plus the constant factor
 */
double compute_gradient(double trace, double dot, std::size_t N, std::size_t n_tiles);

/**
 * @brief Add the local trace of a tile to the global trace.
 *
 * @param diagonal The tile to compute the local trace
 * @param trace The current global trace
 *
 * @return The updated global trace
 */
double compute_trace(std::span<const double> diagonal, double trace);

/**
 * @brief Add the dot product of a vector to a global result.
 *
 * @param vector_T The transposed vector
 * @param vector The vector
 * @param result The current global result
 *
 * @return The updated global result
 */
double compute_dot(std::span<const double> vector_T, std::span<const double> vector, double result);

/**
 * @brief Add the local trace of a matrix tile to the global trace
 *
 * @param tile The matrix tile
 * @param trace The current global trace
 * @param N The dimension of the tile
 *
 * @return The updated global trace
 */
double compute_trace_diag(std::span<const double> tile, double trace, std::size_t N);

}  // end of namespace cpu

GPRAT_NS_END

#endif
