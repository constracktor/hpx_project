#include "gprat_c.hpp"
#include "target.hpp"
#include "utils_c.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdlib>
#include <limits>
#include <string>
using Catch::Matchers::WithinRel;

namespace
{
// Starts the HPX runtime on construction and stops it on destruction so that
// stop_hpx_runtime() is always called even when a test assertion fails mid-test.
struct hpx_runtime_guard
{
    hpx_runtime_guard() { utils::start_hpx_runtime(0, nullptr); }

    ~hpx_runtime_guard() { utils::stop_hpx_runtime(); }
};
}  // namespace

namespace gprat::test
{

static std::string gprat_data_root()
{
    const char *env = std::getenv("GPRAT_ROOT");
    return env ? env : "../data";
}

// Macro that skips the test if no GPU (CUDA or SYCL) is available.
#define GPRAT_SKIP_IF_NO_GPU()                                                                                         \
    do {                                                                                                               \
        if (!utils::compiled_with_cuda() && !utils::compiled_with_sycl())                                              \
            SKIP("GPRat not compiled with GPU support");                                                               \
        if (gprat::gpu_count() == 0)                                                                                   \
            SKIP("No GPU detected");                                                                                   \
    } while (false)

TEST_CASE("GP: results are tile-count invariant", "[unit][gp][predict][tiling]")
{
    // Tiling is purely a scheduling/decomposition detail: predictions, uncertainty,
    // and loss for fixed (untrained) hyperparameters must not depend on n_tiles.
    const std::string root = gprat_data_root();

    constexpr int n = 128, n_reg = 8, n_test = 64;
    const double eps = std::numeric_limits<double>::epsilon() * 1'000'000;

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, 1);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    hpx_runtime_guard hpx_guard;

    // Baseline: a single tile, i.e. no decomposition at all.
    const int baseline_tile_size = utils::compute_train_tile_size(n, 1);
    const auto [baseline_m_tiles, baseline_m_tile_size] = utils::compute_test_tiles(n_test, 1, baseline_tile_size);
    gprat::GP baseline_gp(
        train_in.data, train_out.data, 1, baseline_tile_size, n_reg, { 1.0, 1.0, 0.1 }, { false, false, false });
    const auto baseline_pred =
        baseline_gp.predict_with_uncertainty(test_in.data, baseline_m_tiles, baseline_m_tile_size);
    const double baseline_loss = baseline_gp.calculate_loss();

    for (const int n_tiles : { 2, 4, 8 })
    {
        const int tile_size = utils::compute_train_tile_size(n, n_tiles);
        const auto [m_tiles, m_tile_size] = utils::compute_test_tiles(n_test, n_tiles, tile_size);

        gprat::GP gp(
            train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { false, false, false });
        const auto pred = gp.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);

        for (int i = 0; i < n_test; ++i)
        {
            INFO("n_tiles=" << n_tiles << " mean[" << i << "]");
            REQUIRE_THAT(pred[0][static_cast<std::size_t>(i)],
                         WithinRel(baseline_pred[0][static_cast<std::size_t>(i)], eps));
            INFO("n_tiles=" << n_tiles << " variance[" << i << "]");
            REQUIRE_THAT(pred[1][static_cast<std::size_t>(i)],
                         WithinRel(baseline_pred[1][static_cast<std::size_t>(i)], eps));
        }

        INFO("n_tiles=" << n_tiles << " loss");
        REQUIRE_THAT(gp.calculate_loss(), WithinRel(baseline_loss, eps));
    }
}

TEST_CASE("GP::predict_with_uncertainty: GPU matches CPU with mismatched tile sizes", "[gpu]")
{
    // Regression test: gen_tile_cross_cov_T's CUDA transpose kernel launch, and separately the
    // SYCL TransposeKernel call site plus the SYCL trsm oneMath wrapper, had width/height (resp.
    // M/N) arguments swapped -- which only produces wrong results for non-square tiles, i.e.
    // whenever the test tile size differs from the training tile size. n_test=48 does not divide
    // evenly into n_tile_size=32 here (48 % 32 != 0), so compute_test_tiles falls back to
    // m_tile_size = n_test / n_tiles = 12, genuinely different from n_tile_size = 32 -- unlike
    // e.g. n_test=64, which compute_test_tiles would keep at m_tile_size == n_tile_size and so
    // would not have caught this.
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr int n = 128, n_tiles = 4, n_reg = 8, n_test = 48;
    const int tile_size = utils::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = utils::compute_test_tiles(n_test, n_tiles, tile_size);
    REQUIRE(m_tile_size != tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, 1);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const auto cpu_unc = gp_cpu.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);
    const auto gpu_unc = gp_gpu.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);

    REQUIRE(gpu_unc[0].size() == static_cast<std::size_t>(n_test));
    REQUIRE(gpu_unc[1].size() == static_cast<std::size_t>(n_test));
    for (int i = 0; i < n_test; ++i)
    {
        const auto ui = static_cast<std::size_t>(i);
        INFO("i=" << i);
        REQUIRE(gpu_unc[1][ui] >= 0.0);
        REQUIRE_THAT(gpu_unc[0][ui], WithinRel(cpu_unc[0][ui], 1e-4));
        REQUIRE_THAT(gpu_unc[1][ui], WithinRel(cpu_unc[1][ui], 1e-4));
    }
}

}  // namespace gprat::test
