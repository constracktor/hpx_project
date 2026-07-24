#include "gprat/gprat.hpp"

#include "gprat/cpu/gp_functions.hpp"
#include "gprat/utils.hpp"

#if GPRAT_WITH_DISTRIBUTED
#include "gprat/tiled_dataset.hpp"
#endif

#if GPRAT_WITH_CUDA
#include "gprat/gpu/gp_functions.cuh"
#endif

#if GPRAT_WITH_SYCL
#include "gprat/target.hpp"

#include "gpu/sycl/sycl_gp_functions.hpp"
#endif

GPRAT_NS_BEGIN

GP_data::GP_data(const std::string &f_path, std::size_t n, std::size_t n_reg) :
    file_path(f_path),
    n_samples(n),
    n_regressors(n_reg)
{
    data = load_data(f_path, n, n_reg - 1);
}

// Generic type constructor of class GP ///////////////////////////////////////////////////////////////////////////////
GP::GP(std::vector<double> input,
       std::vector<double> output,
       std::size_t n_tiles,
       std::size_t n_tile_size,
       std::size_t n_regressors,
       const std::vector<double> &kernel_hyperparams,
       std::vector<bool> trainable_bool,
       std::shared_ptr<Target> target) :
    training_input_(std::move(input)),
    training_output_(std::move(output)),
    n_tiles_(n_tiles),
    n_tile_size_(n_tile_size),
    trainable_params_(std::move(trainable_bool)),
    target_(std::move(target)),
    n_reg(n_regressors),
    kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{ }

// CPU-type constructor of class GP ///////////////////////////////////////////////////////////////////////////////////
GP::GP(std::vector<double> input,
       std::vector<double> output,
       std::size_t n_tiles,
       std::size_t n_tile_size,
       std::size_t n_regressors,
       const std::vector<double> &kernel_hyperparams,
       std::vector<bool> trainable_bool) :
    training_input_(std::move(input)),
    training_output_(std::move(output)),
    n_tiles_(n_tiles),
    n_tile_size_(n_tile_size),
    trainable_params_(std::move(trainable_bool)),
    target_(std::make_shared<CPU>()),
    n_reg(n_regressors),
    kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{ }

/// GPU constructor ///////////////////////////////////////////////////////////////////////////////////////////////////
GP::GP(std::vector<double> input,
       std::vector<double> output,
       std::size_t n_tiles,
       std::size_t n_tile_size,
       std::size_t n_regressors,
       const std::vector<double> &kernel_hyperparams,
       std::vector<bool> trainable_bool,
       int gpu_id,
       int n_streams) :
    training_input_(std::move(input)),
    training_output_(std::move(output)),
    n_tiles_(n_tiles),
    n_tile_size_(n_tile_size),
    trainable_params_(std::move(trainable_bool)),
#if GPRAT_WITH_CUDA
    target_(std::make_shared<CUDA_GPU>(CUDA_GPU(gpu_id, n_streams))),

#elif GPRAT_WITH_SYCL
    target_(std::make_shared<SYCL_DEVICE>(SYCL_DEVICE(gpu_id, n_streams))),

#else
    target_(std::make_shared<CPU>()),

#endif
    n_reg(n_regressors),
    kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{
#if !GPRAT_WITH_CUDA && !GPRAT_WITH_SYCL
    throw std::runtime_error(
        "Cannot create GP object using CUDA or SYCL for computation. "
        "CUDA and SYCL are not available because GPRat has been compiled without CUDA and SYCL support. "
        "Remove arguments gpu_id ("
        + std::to_string(gpu_id) + ") and n_streams (" + std::to_string(n_streams)
        + ") to perform computations on the CPU.");
#endif
}

std::string GP::repr() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(12);
    oss << "Kernel_Params: [lengthscale=" << kernel_params.lengthscale << ", vertical_lengthscale="
        << kernel_params.vertical_lengthscale << ", noise_variance=" << kernel_params.noise_variance
        << ", n_regressors=" << n_reg << "], Trainable_Params: [trainable_params l=" << trainable_params_[0]
        << ", trainable_params v=" << trainable_params_[1] << ", trainable_params n=" << trainable_params_[2]
        << "], Target: [" << target_->repr() << "], n_tiles=" << n_tiles_ << ", n_tile_size=" << n_tile_size_;
    return oss.str();
}

std::vector<double> GP::get_training_input() const { return training_input_; }

std::vector<double> GP::get_training_output() const { return training_output_; }

std::vector<double> GP::predict(const std::vector<double> &test_input, std::size_t m_tiles, std::size_t m_tile_size)
{
#if GPRAT_WITH_CUDA
    if (target_->is_gpu())
    {
        return gpu::predict(
            training_input_,
            training_output_,
            test_input,
            kernel_params,
            static_cast<int>(n_tiles_),
            static_cast<int>(n_tile_size_),
            static_cast<int>(m_tiles),
            static_cast<int>(m_tile_size),
            static_cast<int>(n_reg),
            *std::dynamic_pointer_cast<CUDA_GPU>(target_));
    }
#endif
#if GPRAT_WITH_SYCL
    if (target_->is_sycl())
    {
        return sycl_backend::predict(
            training_input_,
            training_output_,
            test_input,
            kernel_params,
            static_cast<int>(n_tiles_),
            static_cast<int>(n_tile_size_),
            static_cast<int>(m_tiles),
            static_cast<int>(m_tile_size),
            static_cast<int>(n_reg),
            *std::dynamic_pointer_cast<SYCL_DEVICE>(target_));
    }
#endif

    tiled_scheduler_local scheduler;
    return cpu::predict(
        scheduler,
        training_input_,
        training_output_,
        test_input,
        kernel_params,
        n_tiles_,
        n_tile_size_,
        m_tiles,
        m_tile_size,
        n_reg);
}

// predict_with_uncertainty ///////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<double>>
GP::predict_with_uncertainty(const std::vector<double> &test_input, std::size_t m_tiles, std::size_t m_tile_size)
{
#if GPRAT_WITH_CUDA
    if (target_->is_gpu())
    {
        return gpu::predict_with_uncertainty(
            training_input_,
            training_output_,
            test_input,
            kernel_params,
            static_cast<int>(n_tiles_),
            static_cast<int>(n_tile_size_),
            static_cast<int>(m_tiles),
            static_cast<int>(m_tile_size),
            static_cast<int>(n_reg),
            *std::dynamic_pointer_cast<CUDA_GPU>(target_));
    }
#endif
#if GPRAT_WITH_SYCL
    if (target_->is_sycl())
    {
        return sycl_backend::predict_with_uncertainty(
            training_input_,
            training_output_,
            test_input,
            kernel_params,
            static_cast<int>(n_tiles_),
            static_cast<int>(n_tile_size_),
            static_cast<int>(m_tiles),
            static_cast<int>(m_tile_size),
            static_cast<int>(n_reg),
            *std::dynamic_pointer_cast<SYCL_DEVICE>(target_));
    }
#endif
    tiled_scheduler_local scheduler;
    return cpu::predict_with_uncertainty(
        scheduler,
        training_input_,
        training_output_,
        test_input,
        kernel_params,
        n_tiles_,
        n_tile_size_,
        m_tiles,
        m_tile_size,
        n_reg);
}

// predict_with_full_cov //////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<double>>
GP::predict_with_full_cov(const std::vector<double> &test_input, std::size_t m_tiles, std::size_t m_tile_size)
{
#if GPRAT_WITH_CUDA
    if (target_->is_gpu())
    {
        return gpu::predict_with_full_cov(
            training_input_,
            training_output_,
            test_input,
            kernel_params,
            static_cast<int>(n_tiles_),
            static_cast<int>(n_tile_size_),
            static_cast<int>(m_tiles),
            static_cast<int>(m_tile_size),
            static_cast<int>(n_reg),
            *std::dynamic_pointer_cast<CUDA_GPU>(target_));
    }
#endif
#if GPRAT_WITH_SYCL
    if (target_->is_sycl())
    {
        return sycl_backend::predict_with_full_cov(
            training_input_,
            training_output_,
            test_input,
            kernel_params,
            static_cast<int>(n_tiles_),
            static_cast<int>(n_tile_size_),
            static_cast<int>(m_tiles),
            static_cast<int>(m_tile_size),
            static_cast<int>(n_reg),
            *std::dynamic_pointer_cast<SYCL_DEVICE>(target_));
    }
#endif
    tiled_scheduler_local scheduler;
    return cpu::predict_with_full_cov(
        scheduler,
        training_input_,
        training_output_,
        test_input,
        kernel_params,
        n_tiles_,
        n_tile_size_,
        m_tiles,
        m_tile_size,
        n_reg);
}

std::vector<double> GP::optimize(const AdamParams &adam_params)
{
#if GPRAT_WITH_CUDA || GPRAT_WITH_SYCL
    if (target_->is_gpu() || target_->is_sycl())
    {
        std::cerr << "GP::optimize is not implemented for GPU targets.\n"
                  << "Falling back to the CPU implementation." << std::endl;
    }
#endif
    tiled_scheduler_local scheduler;
    return cpu::optimize(
        scheduler,
        training_input_,
        training_output_,
        n_tiles_,
        n_tile_size_,
        n_reg,
        adam_params,
        kernel_params,
        trainable_params_);
}

double GP::optimize_step(AdamParams &adam_params, std::size_t iter)
{
#if GPRAT_WITH_CUDA || GPRAT_WITH_SYCL
    if (target_->is_gpu() || target_->is_sycl())
    {
        std::cerr << "GP::optimize_step is not implemented for GPU targets.\n"
                  << "Falling back to the CPU implementation." << std::endl;
    }
#endif
    tiled_scheduler_local scheduler;
    return cpu::optimize_step(
        scheduler,
        training_input_,
        training_output_,
        n_tiles_,
        n_tile_size_,
        n_reg,
        adam_params,
        kernel_params,
        trainable_params_,
        iter);
}

// calculate_loss /////////////////////////////////////////////////////////////////////////////////////////////////////
double GP::calculate_loss()
{
#if GPRAT_WITH_CUDA
    if (target_->is_gpu())
    {
        return gpu::compute_loss(
            training_input_,
            training_output_,
            kernel_params,
            static_cast<int>(n_tiles_),
            static_cast<int>(n_tile_size_),
            static_cast<int>(n_reg),
            *std::dynamic_pointer_cast<CUDA_GPU>(target_));
    }
#endif
#if GPRAT_WITH_SYCL
    if (target_->is_sycl())
    {
        return sycl_backend::compute_loss(
            training_input_,
            training_output_,
            kernel_params,
            static_cast<int>(n_tiles_),
            static_cast<int>(n_tile_size_),
            static_cast<int>(n_reg),
            *std::dynamic_pointer_cast<SYCL_DEVICE>(target_));
    }
#endif
    tiled_scheduler_local scheduler;
    return cpu::calculate_loss(
        scheduler, training_input_, training_output_, kernel_params, n_tiles_, n_tile_size_, n_reg);
}

std::vector<mutable_tile_data<double>> GP::cholesky()
{
#if GPRAT_WITH_CUDA
    if (target_->is_gpu())
    {
        return gpu::cholesky(
            training_input_,
            kernel_params,
            static_cast<int>(n_tiles_),
            static_cast<int>(n_tile_size_),
            static_cast<int>(n_reg),
            *std::dynamic_pointer_cast<CUDA_GPU>(target_));
    }
#endif
#if GPRAT_WITH_SYCL
    if (target_->is_sycl())
    {
        auto raw = sycl_backend::cholesky(
            training_input_,
            kernel_params,
            static_cast<int>(n_tiles_),
            static_cast<int>(n_tile_size_),
            static_cast<int>(n_reg),
            *std::dynamic_pointer_cast<SYCL_DEVICE>(target_));
        std::vector<mutable_tile_data<double>> result;
        result.reserve(raw.size());
        for (auto &tile : raw)
        {
            if (tile.empty())
            {
                result.emplace_back();
            }
            else
            {
                mutable_tile_data<double> t(tile.size());
                std::copy(tile.begin(), tile.end(), t.begin());
                result.push_back(std::move(t));
            }
        }
        return result;
    }
#endif
    tiled_scheduler_local sched;
    return cpu::cholesky(sched, training_input_, kernel_params, n_tiles_, n_tile_size_, n_reg);
}

GPRAT_NS_END
