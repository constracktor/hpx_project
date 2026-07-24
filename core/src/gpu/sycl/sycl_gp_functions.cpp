#include "gpu/sycl/sycl_gp_functions.hpp"

#include "gprat/kernels.hpp"
#include "gprat/target.hpp"

#include "gpu/sycl/sycl_gp_algorithms.hpp"
#include "gpu/sycl/sycl_tiled_algorithms.hpp"
#include "gpu/sycl/sycl_utils.hpp"
#include <hpx/algorithm.hpp>

namespace gprat::sycl_backend
{

// predict ////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<double>
predict(const std::vector<double> &h_training_input,
        const std::vector<double> &h_training_output,
        const std::vector<double> &h_test_input,
        const gprat::SEKParams &sek_params,
        int n_tiles,
        int n_tile_size,
        int m_tiles,
        int m_tile_size,
        int n_regressors,
        gprat::SYCL_DEVICE &sycl_device)
{
    sycl_device.create();

    double *d_training_input = copy_to_device(h_training_input, sycl_device);
    double *d_training_output = copy_to_device(h_training_output, sycl_device);
    double *d_test_input = copy_to_device(h_test_input, sycl_device);

    auto d_tiles = assemble_tiled_covariance_matrix(
        d_training_input,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(n_regressors),
        sek_params,
        sycl_device);

    auto alpha_tiles = assemble_alpha_tiles(
        d_training_output, static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(n_tile_size), sycl_device);

    auto cross_covariance_tiles = assemble_cross_covariance_tiles(
        d_test_input,
        d_training_input,
        static_cast<std::size_t>(m_tiles),
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(n_regressors),
        sek_params,
        sycl_device);

    auto prediction_tiles = assemble_tiles_with_zeros(
        static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    right_looking_cholesky_tiled(
        d_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    // Triangular solve K_NxN * alpha = y

    forward_solve_tiled(
        d_tiles, alpha_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    backward_solve_tiled(
        d_tiles, alpha_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    matrix_vector_tiled(
        cross_covariance_tiles,
        alpha_tiles,
        prediction_tiles,
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles),
        sycl_device);

    std::vector<double> prediction = copy_tiled_vector_to_host_vector(
        prediction_tiles, static_cast<std::size_t>(m_tile_size), std::size_t(m_tiles), sycl_device);

    gprat::sycl_backend::free_lower_tiled_matrix(d_tiles, static_cast<std::size_t>(n_tiles), sycl_device);

    sycl::queue queue = sycl_device.next_queue();

    gprat::sycl_backend::free(alpha_tiles, queue);
    gprat::sycl_backend::free(cross_covariance_tiles, queue);
    gprat::sycl_backend::free(prediction_tiles, queue);

    sycl::free(d_training_input, queue);
    sycl::free(d_training_output, queue);
    sycl::free(d_test_input, queue);

    sycl_device.destroy();

    return prediction;
}

// predict_with_uncertainty ///////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<double>> predict_with_uncertainty(
    const std::vector<double> &h_training_input,
    const std::vector<double> &h_training_output,
    const std::vector<double> &h_test_input,
    const gprat::SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors,
    gprat::SYCL_DEVICE &sycl_device)
{
    sycl_device.create();

    double *d_training_input = copy_to_device(h_training_input, sycl_device);
    double *d_training_output = copy_to_device(h_training_output, sycl_device);
    double *d_test_input = copy_to_device(h_test_input, sycl_device);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles = assemble_tiled_covariance_matrix(
        d_training_input,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(n_regressors),
        sek_params,
        sycl_device);

    auto d_alpha_tiles = assemble_alpha_tiles(
        d_training_output, static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(n_tile_size), sycl_device);

    auto d_prior_K_tiles = assemble_prior_K_tiles(
        d_test_input,
        static_cast<std::size_t>(m_tiles),
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(n_regressors),
        sek_params,
        sycl_device);

    auto d_cross_covariance_tiles = assemble_cross_covariance_tiles(
        d_test_input,
        d_training_input,
        static_cast<std::size_t>(m_tiles),
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(n_regressors),
        sek_params,
        sycl_device);

    auto d_t_cross_covariance_tiles = assemble_t_cross_covariance_tiles(
        d_cross_covariance_tiles,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles),
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(m_tile_size),
        sycl_device);

    // Assemble placeholder matrix for diag(K_MxN * (K^-1_NxN * K_NxM))
    auto d_prior_inter_tiles = assemble_tiles_with_zeros(
        static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    auto d_prediction_tiles = assemble_tiles_with_zeros(
        static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    // Assemble placeholder for uncertainty
    auto d_prediction_uncertainty_tiles = assemble_tiles_with_zeros(
        static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    right_looking_cholesky_tiled(
        d_K_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles,
                        d_alpha_tiles,
                        static_cast<std::size_t>(n_tile_size),
                        static_cast<std::size_t>(n_tiles),
                        sycl_device);
    backward_solve_tiled(d_K_tiles,
                         d_alpha_tiles,
                         static_cast<std::size_t>(n_tile_size),
                         static_cast<std::size_t>(n_tiles),
                         sycl_device);

    // Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_tiled_matrix(
        d_K_tiles,
        d_t_cross_covariance_tiles,
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles),
        sycl_device);

    // Compute predictions
    matrix_vector_tiled(
        d_cross_covariance_tiles,
        d_alpha_tiles,
        d_prediction_tiles,
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles),
        sycl_device);

    // posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM
    symmetric_matrix_matrix_diagonal_tiled(
        d_t_cross_covariance_tiles,
        d_prior_inter_tiles,
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles),
        sycl_device);

    // Compute predicition uncertainty
    vector_difference_tiled(
        d_prior_K_tiles,
        d_prior_inter_tiles,
        d_prediction_uncertainty_tiles,
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(m_tiles));

    // Get predictions and uncertainty to return them
    std::vector<double> prediction = copy_tiled_vector_to_host_vector(
        d_prediction_tiles, static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    std::vector<double> pred_var_full = copy_tiled_vector_to_host_vector(
        d_prediction_uncertainty_tiles,
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(m_tiles),
        sycl_device);

    sycl::queue queue = sycl_device.next_queue();
    sycl::free(d_training_input, queue);
    sycl::free(d_training_output, queue);
    sycl::free(d_test_input, queue);

    gprat::sycl_backend::free_lower_tiled_matrix(d_K_tiles, static_cast<std::size_t>(n_tiles), sycl_device);

    gprat::sycl_backend::free(d_alpha_tiles, queue);
    gprat::sycl_backend::free(d_prior_K_tiles, queue);
    gprat::sycl_backend::free(d_cross_covariance_tiles, queue);
    gprat::sycl_backend::free(d_t_cross_covariance_tiles, queue);
    gprat::sycl_backend::free(d_prior_inter_tiles, queue);
    gprat::sycl_backend::free(d_prediction_tiles, queue);
    gprat::sycl_backend::free(d_prediction_uncertainty_tiles, queue);

    sycl_device.destroy();

    return std::vector<std::vector<double>>{ prediction, pred_var_full };
}

// predict_with_full_cov //////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<double>> predict_with_full_cov(
    const std::vector<double> &h_training_input,
    const std::vector<double> &h_training_output,
    const std::vector<double> &h_test_input,
    const gprat::SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors,
    gprat::SYCL_DEVICE &sycl_device)
{
    sycl_device.create();

    double *d_training_input = copy_to_device(h_training_input, sycl_device);
    double *d_training_output = copy_to_device(h_training_output, sycl_device);
    double *d_test_input = copy_to_device(h_test_input, sycl_device);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles = assemble_tiled_covariance_matrix(
        d_training_input,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(n_regressors),
        sek_params,
        sycl_device);

    auto d_alpha_tiles = assemble_alpha_tiles(
        d_training_output, static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(n_tile_size), sycl_device);

    auto d_prior_K_tiles = assemble_prior_K_tiles_full(
        d_test_input,
        static_cast<std::size_t>(m_tiles),
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(n_regressors),
        sek_params,
        sycl_device);

    auto d_cross_covariance_tiles = assemble_cross_covariance_tiles(
        d_test_input,
        d_training_input,
        static_cast<std::size_t>(m_tiles),
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(n_regressors),
        sek_params,
        sycl_device);

    auto d_t_cross_covariance_tiles = assemble_t_cross_covariance_tiles(
        d_cross_covariance_tiles,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles),
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(m_tile_size),
        sycl_device);

    auto d_prediction_tiles = assemble_tiles_with_zeros(
        static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    // Assemble placeholder for uncertainty
    auto d_prediction_uncertainty_tiles = assemble_tiles_with_zeros(
        static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);

    right_looking_cholesky_tiled(
        d_K_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles,
                        d_alpha_tiles,
                        static_cast<std::size_t>(n_tile_size),
                        static_cast<std::size_t>(n_tiles),
                        sycl_device);
    backward_solve_tiled(d_K_tiles,
                         d_alpha_tiles,
                         static_cast<std::size_t>(n_tile_size),
                         static_cast<std::size_t>(n_tiles),
                         sycl_device);

    // Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_tiled_matrix(
        d_K_tiles,
        d_t_cross_covariance_tiles,
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles),
        sycl_device);

    // Compute predictions
    matrix_vector_tiled(
        d_cross_covariance_tiles,
        d_alpha_tiles,
        d_prediction_tiles,
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles),
        sycl_device);

    // posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM
    symmetric_matrix_matrix_tiled(
        d_t_cross_covariance_tiles,
        d_prior_K_tiles,
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles),
        sycl_device);

    // Compute predicition uncertainty
    matrix_diagonal_tiled(d_prior_K_tiles,
                          d_prediction_uncertainty_tiles,
                          static_cast<std::size_t>(m_tile_size),
                          static_cast<std::size_t>(m_tiles));

    // Get predictions and uncertainty to return them
    std::vector<double> prediction = copy_tiled_vector_to_host_vector(
        d_prediction_tiles, static_cast<std::size_t>(m_tile_size), static_cast<std::size_t>(m_tiles), sycl_device);
    std::vector<double> pred_var_full = copy_tiled_vector_to_host_vector(
        d_prediction_uncertainty_tiles,
        static_cast<std::size_t>(m_tile_size),
        static_cast<std::size_t>(m_tiles),
        sycl_device);

    sycl::queue queue = sycl_device.next_queue();

    sycl::free(d_training_input, queue);
    sycl::free(d_training_output, queue);
    sycl::free(d_test_input, queue);

    gprat::sycl_backend::free_lower_tiled_matrix(d_K_tiles, static_cast<std::size_t>(n_tiles), sycl_device);

    gprat::sycl_backend::free(d_alpha_tiles, queue);

    gprat::sycl_backend::free_full_tiled_matrix(d_prior_K_tiles, static_cast<std::size_t>(m_tiles), sycl_device);
    gprat::sycl_backend::free(d_cross_covariance_tiles, queue);
    gprat::sycl_backend::free(d_t_cross_covariance_tiles, queue);
    gprat::sycl_backend::free(d_prediction_tiles, queue);
    gprat::sycl_backend::free(d_prediction_uncertainty_tiles, queue);

    sycl_device.destroy();

    return std::vector<std::vector<double>>{ prediction, pred_var_full };
}

// compute_loss ///////////////////////////////////////////////////////////////////////////////////////////////////////

double compute_loss(const std::vector<double> &h_training_input,
                    const std::vector<double> &h_training_output,
                    const gprat::SEKParams &sek_params,
                    int n_tiles,
                    int n_tile_size,
                    int n_regressors,
                    gprat::SYCL_DEVICE &sycl_device)
{
    sycl_device.create();

    double *d_training_input = copy_to_device(h_training_input, sycl_device);
    double *d_training_output = copy_to_device(h_training_output, sycl_device);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles = assemble_tiled_covariance_matrix(
        d_training_input,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(n_regressors),
        sek_params,
        sycl_device);

    auto d_alpha_tiles = assemble_alpha_tiles(
        d_training_output, static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(n_tile_size), sycl_device);

    auto d_y_tiles = assemble_y_tiles(
        d_training_output, static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(n_tile_size), sycl_device);

    right_looking_cholesky_tiled(
        d_K_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles,
                        d_alpha_tiles,
                        static_cast<std::size_t>(n_tile_size),
                        static_cast<std::size_t>(n_tiles),
                        sycl_device);
    backward_solve_tiled(d_K_tiles,
                         d_alpha_tiles,
                         static_cast<std::size_t>(n_tile_size),
                         static_cast<std::size_t>(n_tiles),
                         sycl_device);

    // Compute loss
    hpx::shared_future<double> loss_value = compute_loss_tiled(
        d_K_tiles,
        d_alpha_tiles,
        d_y_tiles,
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(n_tiles),
        sycl_device);

    sycl::queue queue = sycl_device.next_queue();

    sycl::free(d_training_input, queue);
    sycl::free(d_training_output, queue);

    loss_value.get();

    gprat::sycl_backend::free_lower_tiled_matrix(d_K_tiles, static_cast<std::size_t>(n_tiles), sycl_device);

    gprat::sycl_backend::free(d_alpha_tiles, queue);
    gprat::sycl_backend::free(d_y_tiles, queue);

    sycl_device.destroy();

    return loss_value.get();
}

// cholesky ///////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<double>>
cholesky(const std::vector<double> &h_training_input,
         const gprat::SEKParams &sek_params,
         int n_tiles,
         int n_tile_size,
         int n_regressors,
         gprat::SYCL_DEVICE &sycl_device)
{
    sycl_device.create();

    double *d_training_input = copy_to_device(h_training_input, sycl_device);

    // Assemble tiled covariance matrix on GPU.
    std::vector<hpx::shared_future<double *>> d_tiles = assemble_tiled_covariance_matrix(
        d_training_input,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(n_tile_size),
        static_cast<std::size_t>(n_regressors),
        sek_params,
        sycl_device);

    // Compute Tiled Cholesky decomposition on device
    right_looking_cholesky_tiled(
        d_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    // Copy tiled matrix to host
    std::vector<std::vector<double>> h_tiles = move_lower_tiled_matrix_to_host(
        d_tiles, static_cast<std::size_t>(n_tile_size), static_cast<std::size_t>(n_tiles), sycl_device);

    sycl::queue queue = sycl_device.next_queue();
    sycl::free(d_training_input, queue);

    sycl_device.destroy();

    return h_tiles;
}

}  // namespace gprat::sycl_backend
