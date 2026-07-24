#include "gprat/gpu/gp_functions.cuh"

#include "gprat/gpu/cuda_utils.cuh"
#include "gprat/gpu/gp_algorithms.cuh"
#include "gprat/gpu/tiled_algorithms.cuh"
#include "gprat/kernels.hpp"
#include "gprat/target.hpp"
#include "gprat/tile_data.hpp"

#include "apex_utils.hpp"
#include <cuda_runtime.h>
#include <hpx/algorithm.hpp>
#include <hpx/async_cuda/cuda_exception.hpp>

GPRAT_NS_BEGIN

namespace gpu
{

std::vector<double>
predict(const std::vector<double> &h_training_input,
        const std::vector<double> &h_training_output,
        const std::vector<double> &h_test_input,
        const SEKParams &sek_params,
        int n_tiles,
        int n_tile_size,
        int m_tiles,
        int m_tile_size,
        int n_regressors,
        CUDA_GPU &gpu)
{
    GPRAT_START_STEP(alloc_timer);

    gpu.create();
    cusolverDnHandle_t cusolver = create_cusolver_handle();

    GPRAT_END_STEP(alloc_timer, "predict_step ressource allocation");
    GPRAT_START_STEP(assembly_timer);

    double *d_training_input = copy_to_device(h_training_input, gpu);
    double *d_training_output = copy_to_device(h_training_output, gpu);
    double *d_test_input = copy_to_device(h_test_input, gpu);

    auto d_tiles =
        assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);

    auto alpha_tiles = assemble_alpha_tiles(d_training_output, n_tiles, n_tile_size, gpu);
    auto cross_covariance_tiles = assemble_cross_covariance_tiles(
        d_test_input, d_training_input, m_tiles, n_tiles, m_tile_size, n_tile_size, n_regressors, sek_params, gpu);

    auto prediction_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    GPRAT_END_STEP(
        assembly_timer, "predict_step assembly", d_tiles, alpha_tiles, cross_covariance_tiles, prediction_tiles);
    GPRAT_START_STEP(assembly_cholesky_timer);

    right_looking_cholesky_tiled(d_tiles, n_tile_size, n_tiles, gpu, cusolver);

    GPRAT_END_STEP(assembly_cholesky_timer, "predict_step cholesky", d_tiles);
    GPRAT_START_STEP(forward_timer);

    // Forward solve of triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_tiles, alpha_tiles, n_tile_size, n_tiles, gpu);

    GPRAT_END_STEP(forward_timer, "predict_step forward", alpha_tiles);
    GPRAT_START_STEP(backward_timer);

    // Backward solve of triangular solve K_NxN * alpha = y
    backward_solve_tiled(d_tiles, alpha_tiles, n_tile_size, n_tiles, gpu);

    GPRAT_END_STEP(backward_timer, "predict_step backward", alpha_tiles);
    GPRAT_START_STEP(prediction_timer);

    matrix_vector_tiled(
        cross_covariance_tiles, alpha_tiles, prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles, gpu);

    GPRAT_END_STEP(prediction_timer, "predict_step prediction", prediction_tiles);
    GPRAT_START_STEP(copyback_timer);

    std::vector<double> prediction = copy_tiled_vector_to_host_vector(prediction_tiles, m_tile_size, m_tiles, gpu);

    GPRAT_END_STEP(copyback_timer, "predict_step copyback");
    GPRAT_START_STEP(destroy_timer);

    free_lower_tiled_matrix(d_tiles, n_tiles);
    free(alpha_tiles);
    free(cross_covariance_tiles);
    free(prediction_tiles);
    destroy(cusolver);

    check_cuda_error(cudaFree(d_training_input));
    check_cuda_error(cudaFree(d_training_output));
    check_cuda_error(cudaFree(d_test_input));

    gpu.destroy();

    GPRAT_END_STEP(destroy_timer, "predict_step ressource destroy");

    return prediction;
}

std::vector<std::vector<double>> predict_with_uncertainty(
    const std::vector<double> &h_training_input,
    const std::vector<double> &h_training_output,
    const std::vector<double> &h_test_input,
    const SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors,
    CUDA_GPU &gpu)
{
    GPRAT_START_STEP(alloc_timer);

    gpu.create();
    cusolverDnHandle_t cusolver = create_cusolver_handle();

    GPRAT_END_STEP(alloc_timer, "predict_uncer_step ressource allocation");
    GPRAT_START_STEP(assembly_timer);

    double *d_training_input = copy_to_device(h_training_input, gpu);
    double *d_training_output = copy_to_device(h_training_output, gpu);
    double *d_test_input = copy_to_device(h_test_input, gpu);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles =
        assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);

    auto d_alpha_tiles = assemble_alpha_tiles(d_training_output, n_tiles, n_tile_size, gpu);

    auto d_prior_K_tiles = assemble_prior_K_tiles(d_test_input, m_tiles, m_tile_size, n_regressors, sek_params, gpu);

    auto d_cross_covariance_tiles = assemble_cross_covariance_tiles(
        d_test_input, d_training_input, m_tiles, n_tiles, m_tile_size, n_tile_size, n_regressors, sek_params, gpu);

    auto d_t_cross_covariance_tiles =
        assemble_t_cross_covariance_tiles(d_cross_covariance_tiles, n_tiles, m_tiles, n_tile_size, m_tile_size, gpu);

    // Assemble placeholder matrix for diag(K_MxN * (K^-1_NxN * K_NxM))
    auto d_prior_inter_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    auto d_prediction_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    // Assemble placeholder for uncertainty
    auto d_prediction_uncertainty_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    GPRAT_END_STEP(
        assembly_timer,
        "predict_uncer_step assembly",
        d_K_tiles,
        d_alpha_tiles,
        d_cross_covariance_tiles,
        d_prediction_tiles,
        d_prior_K_tiles,
        d_prior_inter_tiles,
        d_t_cross_covariance_tiles);
    GPRAT_START_STEP(assembly_cholesky_timer);

    right_looking_cholesky_tiled(d_K_tiles, n_tile_size, n_tiles, gpu, cusolver);

    GPRAT_END_STEP(assembly_cholesky_timer, "predict_uncer_step cholesky", d_K_tiles);
    GPRAT_START_STEP(forward_timer);

    // Forward part of triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);

    GPRAT_END_STEP(forward_timer, "predict_uncer_step forward", d_alpha_tiles);
    GPRAT_START_STEP(backward_timer);

    // Backward part of the triangular solve K_NxN * alpha = y
    backward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);

    GPRAT_END_STEP(backward_timer, "predict_uncer_step backward", d_alpha_tiles);
    GPRAT_START_STEP(forward_KcK_timer);

    // Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_tiled_matrix(d_K_tiles, d_t_cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles, gpu);

    GPRAT_END_STEP(forward_KcK_timer, "predict_uncer_step forward KcK", d_t_cross_covariance_tiles);
    GPRAT_START_STEP(prediction_timer);

    // Compute predictions
    matrix_vector_tiled(
        d_cross_covariance_tiles, d_alpha_tiles, d_prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles, gpu);

    GPRAT_END_STEP(prediction_timer, "predict_uncer_step prediction", d_prediction_tiles);
    GPRAT_START_STEP(posterior_covariance_timer);

    // posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM
    symmetric_matrix_matrix_diagonal_tiled(
        d_t_cross_covariance_tiles, d_prior_inter_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles, gpu);

    GPRAT_END_STEP(posterior_covariance_timer, "predict_uncer_step posterior covariance", d_prior_inter_tiles);
    GPRAT_START_STEP(prediction_uncertainty_timer);

    // Compute predicition uncertainty
    vector_difference_tiled(
        d_prior_K_tiles, d_prior_inter_tiles, d_prediction_uncertainty_tiles, m_tile_size, m_tiles, gpu);

    GPRAT_END_STEP(
        prediction_uncertainty_timer, "predict_uncer_step prediction uncertainty", d_prediction_uncertainty_tiles);
    GPRAT_START_STEP(copyback_timer);

    // Get predictions and uncertainty to return them
    std::vector<double> prediction = copy_tiled_vector_to_host_vector(d_prediction_tiles, m_tile_size, m_tiles, gpu);
    std::vector<double> pred_var_full =
        copy_tiled_vector_to_host_vector(d_prediction_uncertainty_tiles, m_tile_size, m_tiles, gpu);

    GPRAT_END_STEP(copyback_timer, "predict_uncer_step copyback");
    GPRAT_START_STEP(destroy_timer);

    check_cuda_error(cudaFree(d_training_input));
    check_cuda_error(cudaFree(d_training_output));
    check_cuda_error(cudaFree(d_test_input));
    free_lower_tiled_matrix(d_K_tiles, n_tiles);
    free(d_alpha_tiles);
    free(d_prior_K_tiles);
    free(d_cross_covariance_tiles);
    free(d_t_cross_covariance_tiles);
    free(d_prior_inter_tiles);
    free(d_prediction_tiles);
    free(d_prediction_uncertainty_tiles);
    destroy(cusolver);

    gpu.destroy();

    GPRAT_END_STEP(destroy_timer, "predict_uncer_step ressource destroy");

    return std::vector<std::vector<double>>{ prediction, pred_var_full };
}

std::vector<std::vector<double>> predict_with_full_cov(
    const std::vector<double> &h_training_input,
    const std::vector<double> &h_training_output,
    const std::vector<double> &h_test_input,
    const SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors,
    CUDA_GPU &gpu)
{
    GPRAT_START_STEP(alloc_timer);

    gpu.create();
    cusolverDnHandle_t cusolver = create_cusolver_handle();

    GPRAT_END_STEP(alloc_timer, "predict_full_cov_step ressource allocation");
    GPRAT_START_STEP(assembly_timer);

    double *d_training_input = copy_to_device(h_training_input, gpu);
    double *d_training_output = copy_to_device(h_training_output, gpu);
    double *d_test_input = copy_to_device(h_test_input, gpu);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles =
        assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);

    auto d_alpha_tiles = assemble_alpha_tiles(d_training_output, n_tiles, n_tile_size, gpu);

    auto d_prior_K_tiles =
        assemble_prior_K_tiles_full(d_test_input, m_tiles, m_tile_size, n_regressors, sek_params, gpu);

    auto d_cross_covariance_tiles = assemble_cross_covariance_tiles(
        d_test_input, d_training_input, m_tiles, n_tiles, m_tile_size, n_tile_size, n_regressors, sek_params, gpu);

    auto d_t_cross_covariance_tiles =
        assemble_t_cross_covariance_tiles(d_cross_covariance_tiles, n_tiles, m_tiles, n_tile_size, m_tile_size, gpu);

    auto d_prediction_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    // Assemble placeholder for uncertainty
    auto d_prediction_uncertainty_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    GPRAT_END_STEP(assembly_timer,
                   "predict_full_cov_step assembly",
                   d_K_tiles,
                   d_alpha_tiles,
                   d_cross_covariance_tiles,
                   d_prediction_tiles,
                   d_prior_K_tiles,
                   d_t_cross_covariance_tiles);
    GPRAT_START_STEP(assembly_cholesky_timer);

    right_looking_cholesky_tiled(d_K_tiles, n_tile_size, n_tiles, gpu, cusolver);

    GPRAT_END_STEP(assembly_cholesky_timer, "predict_full_cov_step cholesky", d_K_tiles);
    GPRAT_START_STEP(forward_timer);

    // Forward part of triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);

    GPRAT_END_STEP(forward_timer, "predict_full_cov_step forward", d_alpha_tiles);
    GPRAT_START_STEP(backward_timer);

    // Backward part of triangular solve K_NxN * alpha = y
    backward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);

    GPRAT_END_STEP(backward_timer, "predict_full_cov_step backward", d_alpha_tiles);
    GPRAT_START_STEP(forward_KcK_timer);

    // Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_tiled_matrix(d_K_tiles, d_t_cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles, gpu);

    GPRAT_END_STEP(forward_KcK_timer, "predict_full_cov_step forward KcK", d_t_cross_covariance_tiles);
    GPRAT_START_STEP(prediction_timer);

    // Compute predictions
    matrix_vector_tiled(
        d_cross_covariance_tiles, d_alpha_tiles, d_prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles, gpu);

    GPRAT_END_STEP(prediction_timer, "predict_full_cov_step prediction", d_prediction_tiles);
    GPRAT_START_STEP(full_cov_timer);

    // posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM
    symmetric_matrix_matrix_tiled(
        d_t_cross_covariance_tiles, d_prior_K_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles, gpu);

    GPRAT_END_STEP(full_cov_timer, "predict_full_cov_step full cov", d_prior_K_tiles);
    GPRAT_START_STEP(pred_uncer_timer);

    // Compute predicition uncertainty
    matrix_diagonal_tiled(d_prior_K_tiles, d_prediction_uncertainty_tiles, m_tile_size, m_tiles, gpu);

    GPRAT_END_STEP(pred_uncer_timer, "predict_full_cov_step pred uncer", d_prediction_uncertainty_tiles);
    GPRAT_START_STEP(copyback_timer);

    // Get predictions and uncertainty to return them
    std::vector<double> prediction = copy_tiled_vector_to_host_vector(d_prediction_tiles, m_tile_size, m_tiles, gpu);
    std::vector<double> pred_var_full =
        copy_tiled_vector_to_host_vector(d_prediction_uncertainty_tiles, m_tile_size, m_tiles, gpu);

    GPRAT_END_STEP(copyback_timer, "predict_full_cov_step copyback");
    GPRAT_START_STEP(destroy_timer);

    check_cuda_error(cudaFree(d_training_input));
    check_cuda_error(cudaFree(d_training_output));
    check_cuda_error(cudaFree(d_test_input));
    free_lower_tiled_matrix(d_K_tiles, n_tiles);
    free(d_alpha_tiles);
    free_full_tiled_matrix(d_prior_K_tiles, m_tiles);
    free(d_cross_covariance_tiles);
    free(d_t_cross_covariance_tiles);
    free(d_prediction_tiles);
    free(d_prediction_uncertainty_tiles);
    destroy(cusolver);

    gpu.destroy();

    GPRAT_END_STEP(destroy_timer, "predict_full_cov_step ressource destroy");

    return std::vector<std::vector<double>>{ prediction, pred_var_full };
}

double compute_loss(const std::vector<double> &h_training_input,
                    const std::vector<double> &h_training_output,
                    const SEKParams &sek_params,
                    int n_tiles,
                    int n_tile_size,
                    int n_regressors,
                    CUDA_GPU &gpu)
{
    gpu.create();

    double *d_training_input = copy_to_device(h_training_input, gpu);
    double *d_training_output = copy_to_device(h_training_output, gpu);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles =
        assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);

    auto d_alpha_tiles = assemble_alpha_tiles(d_training_output, n_tiles, n_tile_size, gpu);

    auto d_y_tiles = assemble_y_tiles(d_training_output, n_tiles, n_tile_size, gpu);

    cusolverDnHandle_t cusolver = create_cusolver_handle();
    right_looking_cholesky_tiled(d_K_tiles, n_tile_size, n_tiles, gpu, cusolver);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);
    backward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);

    // Compute loss
    hpx::shared_future<double> loss_value =
        compute_loss_tiled(d_K_tiles, d_alpha_tiles, d_y_tiles, n_tile_size, n_tiles, gpu);

    check_cuda_error(cudaFree(d_training_input));
    check_cuda_error(cudaFree(d_training_output));

    loss_value.get();
    free_lower_tiled_matrix(d_K_tiles, n_tiles);
    free(d_alpha_tiles);
    free(d_y_tiles);
    destroy(cusolver);

    gpu.destroy();

    return loss_value.get();
}

std::vector<double>
optimize(const std::vector<double> &training_input,
         const std::vector<double> &training_output,
         int n_tiles,
         int n_tile_size,
         int n_regressors,
         const AdamParams &adam_params,
         const SEKParams &sek_params,
         std::vector<bool> trainable_params,
         CUDA_GPU &gpu)
{
    throw std::logic_error("Function not implemented for GPU");
    // return std::vector<double>>();
}

double optimize_step(const std::vector<double> &training_input,
                     const std::vector<double> &training_output,
                     int n_tiles,
                     int n_tile_size,
                     int n_regressors,
                     AdamParams &adam_params,
                     SEKParams &sek_params,
                     std::vector<bool> trainable_params,
                     int iter,
                     CUDA_GPU &gpu)
{
    throw std::logic_error("Function not implemented for GPU");
    // return 0.0;
}

std::vector<mutable_tile_data<double>>
cholesky(const std::vector<double> &h_training_input,
         const SEKParams &sek_params,
         int n_tiles,
         int n_tile_size,
         int n_regressors,
         CUDA_GPU &gpu)
{
    gpu.create();

#if GPRAT_APEX_CHOLESKY
    GPRAT_START_TIMER(assembly_cholesky_timer);
#endif

    GPRAT_START_STEP(assembly_timer);

    double *d_training_input = copy_to_device(h_training_input, gpu);
    // Assemble tiled covariance matrix on GPU.
    std::vector<hpx::shared_future<double *>> d_tiles =
        assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);

    GPRAT_END_STEP(assembly_timer, "cholesky_step assembly", d_tiles);
    GPRAT_START_STEP(cholesky_timer);

    // Compute Tiled Cholesky decomposition on device
    cusolverDnHandle_t cusolver = create_cusolver_handle();
    right_looking_cholesky_tiled(d_tiles, n_tile_size, n_tiles, gpu, cusolver);

    GPRAT_END_STEP(cholesky_timer, "cholesky_step cholesky", d_tiles);

#if GPRAT_APEX_CHOLESKY
    GPRAT_STOP_TIMER(assembly_cholesky_timer, "cholesky", d_tiles);
#endif

    // Copy tiled matrix to host
    auto h_tiles = move_lower_tiled_matrix_to_host(d_tiles, n_tile_size, n_tiles, gpu);

    cudaFree(d_training_input);
    destroy(cusolver);
    gpu.destroy();

    return h_tiles;
}

}  // end of namespace gpu

GPRAT_NS_END
