#include "gprat/gprat.hpp"
#include "gprat/utils.hpp"

#include "test_data.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Matchers::WithinRel;

// Boost
#include <boost/json/src.hpp>

// Standard library
#include <string>
#include <string_view>

namespace gprat::test
{

// Parameters /////////////////////////////////////////////////////////////////////////////////////

// Global test settings
constexpr std::size_t n_test = 128;
constexpr std::size_t n_train = 128;
constexpr std::size_t n_tiles = 4;
constexpr std::size_t n_reg = 8;

// CPU test settings
constexpr int OPT_ITER = 3;

// CUDA and SYCL test settings
constexpr int gpu_id = 0;
constexpr int n_streams = 4;

// Test execution /////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Generates results using the CPU for computations.
 */
gprat_results run_on_data_cpu(const std::string &train_path, const std::string &out_path, const std::string &test_path)
{
    const std::size_t tile_size = gprat::compute_train_tile_size(n_train, n_tiles);
    const auto test_tiles = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPT_ITER };

    gprat::GP_data training_input(train_path, n_train, n_reg);
    gprat::GP_data training_output(out_path, n_train, n_reg);
    gprat::GP_data test_input(test_path, n_test, n_reg);

    const std::vector<bool> trainable = { true, true, true };

    // GP constructors do not use HPX, so it is safe to construct before starting the runtime.
    gprat::GP gp_cpu(
        training_input.data, training_output.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, trainable);

    gprat::start_hpx_runtime(0, nullptr);

    gprat_results results_cpu;
    results_cpu.cholesky = to_vector(gp_cpu.cholesky());
    results_cpu.sum = gp_cpu.predict_with_uncertainty(test_input.data, test_tiles.first, test_tiles.second);
    results_cpu.full = gp_cpu.predict_with_full_cov(test_input.data, test_tiles.first, test_tiles.second);
    results_cpu.pred = gp_cpu.predict(test_input.data, test_tiles.first, test_tiles.second);
    results_cpu.losses = gp_cpu.optimize(hpar);

    gprat::stop_hpx_runtime();

    return results_cpu;
}

/**
 * @brief Generates results using a CUDA GPU or SYCL device.
 */
gprat_results run_on_data_gpu(const std::string &train_path, const std::string &out_path, const std::string &test_path)
{
    const std::size_t tile_size = gprat::compute_train_tile_size(n_train, n_tiles);
    const auto test_tiles = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data training_input(train_path, n_train, n_reg);
    gprat::GP_data training_output(out_path, n_train, n_reg);
    gprat::GP_data test_input(test_path, n_test, n_reg);

    const std::vector<bool> trainable = { true, true, true };

    gprat::GP gp_gpu(
        training_input.data,
        training_output.data,
        n_tiles,
        tile_size,
        n_reg,
        { 1.0, 1.0, 0.1 },
        trainable,
        gpu_id,
        n_streams);

    gprat::start_hpx_runtime(0, nullptr);

    gprat_results results_gpu;
    results_gpu.cholesky = to_vector(gp_gpu.cholesky());
    // NOTE: optimize and optimize_step are currently not implemented for GPU.
    // When GPU optimize is added, extend this function and update the GPU test case to verify losses.
    results_gpu.sum = gp_gpu.predict_with_uncertainty(test_input.data, test_tiles.first, test_tiles.second);
    results_gpu.full = gp_gpu.predict_with_full_cov(test_input.data, test_tiles.first, test_tiles.second);
    results_gpu.pred = gp_gpu.predict(test_input.data, test_tiles.first, test_tiles.second);

    gprat::stop_hpx_runtime();

    return results_gpu;
}

// Test cases /////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("GP CPU: results match baseline", "[integration][cpu]")
{
    const std::string root = get_data_directory("../data");

    const auto results = run_on_data_cpu(root + "/data_1024/training_input.txt",
                                         root + "/data_1024/training_output.txt",
                                         root + "/data_1024/test_input.txt");

    gprat_results expected_results;
    if (!load_or_create_expected_results(root + "/data_1024/output.json", results, expected_results))
    {
        std::cerr << "No previous results to compare to. The current results have been saved instead!\n";
        return;
    }

    double eps = std::numeric_limits<double>::epsilon() * 1'000'000;

    for (std::size_t i = 0, n = results.cholesky.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.cholesky[i].size(); j != m; ++j)
        {
            INFO("CPU cholesky " << i << " " << j);
            REQUIRE_THAT(results.cholesky[i][j], WithinRel(expected_results.cholesky[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.losses.size(); i != n; ++i)
    {
        INFO("CPU losses " << i);
        REQUIRE_THAT(results.losses[i], WithinRel(expected_results.losses[i], eps));
    }

    for (std::size_t i = 0, n = results.sum.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.sum[i].size(); j != m; ++j)
        {
            INFO("CPU sum " << i << " " << j);
            REQUIRE_THAT(results.sum[i][j], WithinRel(expected_results.sum[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.full.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.full[i].size(); j != m; ++j)
        {
            INFO("CPU full " << i << " " << j);
            REQUIRE_THAT(results.full[i][j], WithinRel(expected_results.full[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.pred.size(); i != n; ++i)
    {
        INFO("CPU pred " << i);
        REQUIRE_THAT(results.pred[i], WithinRel(expected_results.pred[i], eps));
    }
}

TEST_CASE("GP GPU: results match baseline", "[integration][gpu]")
{
    if (!gprat::compiled_with_cuda() && !gprat::compiled_with_sycl())
    {
        SKIP("GPU not compiled in — skipping GPU integration test.");
    }
    if (gprat::compiled_with_sycl() && !gprat::sycl_gpu_functional())
    {
        SKIP("SYCL GPU runtime not functional (oneMath ABI mismatch).");
    }

    const std::string root = get_data_directory("../data");

    const auto results = run_on_data_gpu(root + "/data_1024/training_input.txt",
                                         root + "/data_1024/training_output.txt",
                                         root + "/data_1024/test_input.txt");

    gprat_results expected_results;
    if (!load_or_create_expected_results(root + "/data_1024/output.json", results, expected_results))
    {
        std::cerr << "No previous results to compare to. The current results have been saved instead!\n";
        return;
    }

    double eps = std::numeric_limits<double>::epsilon() * 1'000'000;

    for (std::size_t i = 0, n = results.cholesky.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.cholesky[i].size(); j != m; ++j)
        {
            INFO("GPU cholesky " << i << " " << j);
            REQUIRE_THAT(results.cholesky[i][j], WithinRel(expected_results.cholesky[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.sum.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.sum[i].size(); j != m; ++j)
        {
            INFO("GPU sum " << i << " " << j);
            REQUIRE_THAT(results.sum[i][j], WithinRel(expected_results.sum[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.full.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.full[i].size(); j != m; ++j)
        {
            INFO("GPU full " << i << " " << j);
            REQUIRE_THAT(results.full[i][j], WithinRel(expected_results.full[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.pred.size(); i != n; ++i)
    {
        INFO("GPU pred " << i);
        REQUIRE_THAT(results.pred[i], WithinRel(expected_results.pred[i], eps));
    }
}

}  // namespace gprat::test
