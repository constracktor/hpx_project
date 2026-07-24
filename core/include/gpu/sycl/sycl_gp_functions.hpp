#ifndef SYCL_GP_FUNCTIONS_H
#define SYCL_GP_FUNCTIONS_H

#include "gprat/hyperparameters.hpp"
#include "gprat/kernels.hpp"
#include "gprat/target.hpp"

namespace gprat::sycl_backend
{

/**
 * @brief Compute the predictions without uncertainties.
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param test_input The test input data
 * @param sek_params The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param m_tiles The number of test tiles
 * @param m_tile_size The size of each test tile
 * @param n_regressors The number of regressors
 * @param sycl_device SYCL target for computations
 *
 * @return A vector containing the predictions
 */
std::vector<double>
predict(const std::vector<double> &training_input,
        const std::vector<double> &training_output,
        const std::vector<double> &test_input,
        const gprat::SEKParams &sek_params,
        int n_tiles,
        int n_tile_size,
        int m_tiles,
        int m_tile_size,
        int n_regressors,
        gprat::SYCL_DEVICE &sycl_device);

/**
 * @brief Compute the predictions with uncertainties.
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param test_input The test input data
 * @param sek_params The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param m_tiles The number of test tiles
 * @param m_tile_size The size of each test tile
 * @param n_regressors The number of regressors
 * @param sycl_device SYCL target for computations
 *
 * @return A vector containing the prediction vector and the uncertainty vector
 */
std::vector<std::vector<double>> predict_with_uncertainty(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    const std::vector<double> &test_input,
    const gprat::SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors,
    gprat::SYCL_DEVICE &sycl_device);

/**
 * @brief Compute the predictions with full covariance matrix.
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param test_data The test input data
 * @param sek_params The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param m_tiles The number of test tiles
 * @param m_tile_size The size of each test tile
 * @param n_regressors The number of regressors
 * @param sycl_device SYCL target for computations
 *
 * @return A vector containing the prediction vector and the full posterior covariance matrix
 */
std::vector<std::vector<double>> predict_with_full_cov(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    const std::vector<double> &test_data,
    const gprat::SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors,
    gprat::SYCL_DEVICE &sycl_device);

/**
 * @brief Compute loss for given data and Gaussian process model
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param sek_params The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 * @param sycl_device SYCL target for computations
 *
 * @return The loss
 */
double compute_loss(const std::vector<double> &training_input,
                    const std::vector<double> &training_output,
                    const gprat::SEKParams &sek_params,
                    int n_tiles,
                    int n_tile_size,
                    int n_regressors,
                    gprat::SYCL_DEVICE &sycl_device);

/**
 * @brief Perform Cholesky decompositon (+ Assembly)
 *
 * @param training_input The training input data
 * @param sek_params The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 * @param sycl_device SYCL target for computations
 *
 * @return The tiled Cholesky factor
 */
std::vector<std::vector<double>>
cholesky(const std::vector<double> &training_input,
         const gprat::SEKParams &sek_params,
         int n_tiles,
         int n_tile_size,
         int n_regressors,
         gprat::SYCL_DEVICE &sycl_device);

}  // namespace gprat::sycl_backend

#endif  // ! SYCL_GP_FUNCTIONS_H
