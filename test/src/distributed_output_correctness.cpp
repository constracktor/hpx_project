// Verifies the distributed compute path (tiled_scheduler_sma, the scheduler
// examples/gprat_distributed uses to spread tiles across HPX localities, and the
// gp_*_actions.cpp action dispatch it goes through) against the same data/data_1024/output.json
// baseline that GPRat_test_output_correctness checks. GPRat_test_output_correctness always uses
// gprat::GP, which hardcodes the locality-oblivious tiled_scheduler_local -- it never touches the
// distributed action code no matter how many HPX localities are running. This binary calls the
// same free functions examples/gprat_distributed/src/main.cpp calls, with tiled_scheduler_sma, so
// it actually exercises that dispatch, and is meant to be run across multiple localities via
// test/scripts/run_distributed_multi_locality.sh (see test/CMakeLists.txt).
//
// Unlike GPRat_test_output_correctness this isn't a Catch2 test: it needs to run under
// hpx::init/hpx_main so the runtime bootstraps as a real (possibly multi-locality) HPX job, and
// hpx_main only executes on locality 0 by default -- which is exactly the gating we want, since
// only locality 0 should compute and compare results while the others just service actions.
#include "gprat/cpu/adapter_cblas_fp64_actions.hpp"
#include "gprat/cpu/gp_algorithms_actions.hpp"
#include "gprat/cpu/gp_functions.hpp"
#include "gprat/cpu/gp_optimizer_actions.hpp"
#include "gprat/cpu/gp_uncertainty_actions.hpp"
#include "gprat/gprat.hpp"
#include "gprat/scheduler/sma.hpp"
#include "gprat/utils.hpp"

#include "test_data.hpp"
#include <boost/json/src.hpp>
#include <cmath>
#include <hpx/hpx_init.hpp>
#include <iostream>
#include <limits>
#include <string>

namespace
{

// Matches GPRat_test_output_correctness's CPU test settings so both can compare against the
// same data/data_1024/output.json baseline.
constexpr std::size_t n_test = 128;
constexpr std::size_t n_train = 128;
constexpr std::size_t n_tiles = 4;
constexpr std::size_t n_reg = 8;
constexpr int OPT_ITER = 3;

bool nearly_equal(double a, double b, double eps)
{
    return std::fabs(a - b) <= eps * (std::max)(std::fabs(a), std::fabs(b));
}

bool compare(
    const std::vector<double> &actual, const std::vector<double> &expected, double eps, const std::string &label)
{
    if (actual.size() != expected.size())
    {
        std::cerr << label << ": size mismatch (" << actual.size() << " vs " << expected.size() << ")\n";
        return false;
    }
    bool ok = true;
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        if (!nearly_equal(actual[i], expected[i], eps))
        {
            std::cerr << label << "[" << i << "]: " << actual[i] << " != " << expected[i] << "\n";
            ok = false;
        }
    }
    return ok;
}

bool compare(const std::vector<std::vector<double>> &actual,
             const std::vector<std::vector<double>> &expected,
             double eps,
             const std::string &label)
{
    if (actual.size() != expected.size())
    {
        std::cerr << label << ": outer size mismatch (" << actual.size() << " vs " << expected.size() << ")\n";
        return false;
    }
    bool ok = true;
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        ok = compare(actual[i], expected[i], eps, label + "[" + std::to_string(i) + "]") && ok;
    }
    return ok;
}

}  // namespace

int hpx_main(hpx::program_options::variables_map & /*vm*/)
{
    // GPRAT_TEST_DATA_DIR is baked in at configure time (CMAKE_SOURCE_DIR/data); GPRAT_ROOT, if
    // set, overrides it -- matching GPRat_test_output_correctness's own resolution so both always
    // agree on which data/data_1024/output.json baseline they're reading/writing.
    const auto root = get_data_directory(GPRAT_TEST_DATA_DIR);

    const std::size_t tile_size = gprat::compute_train_tile_size(n_train, n_tiles);
    const auto test_tiles = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPT_ITER };
    gprat::SEKParams sek_params = { 1.0, 1.0, 0.1 };
    const std::vector<bool> trainable = { true, true, true };

    gprat::GP_data training_input(root + "/data_1024/training_input.txt", n_train, n_reg);
    gprat::GP_data training_output(root + "/data_1024/training_output.txt", n_train, n_reg);
    gprat::GP_data test_input(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::tiled_scheduler_sma scheduler;

    gprat_results results;
    results.cholesky =
        to_vector(gprat::cpu::cholesky(scheduler, training_input.data, sek_params, n_tiles, tile_size, n_reg));
    results.sum = gprat::cpu::predict_with_uncertainty(
        scheduler,
        training_input.data,
        training_output.data,
        test_input.data,
        sek_params,
        n_tiles,
        tile_size,
        test_tiles.first,
        test_tiles.second,
        n_reg);
    results.full = gprat::cpu::predict_with_full_cov(
        scheduler,
        training_input.data,
        training_output.data,
        test_input.data,
        sek_params,
        n_tiles,
        tile_size,
        test_tiles.first,
        test_tiles.second,
        n_reg);
    results.pred = gprat::cpu::predict(
        scheduler,
        training_input.data,
        training_output.data,
        test_input.data,
        sek_params,
        n_tiles,
        tile_size,
        test_tiles.first,
        test_tiles.second,
        n_reg);
    results.losses = gprat::cpu::optimize(
        scheduler, training_input.data, training_output.data, n_tiles, tile_size, n_reg, hpar, sek_params, trainable);

    gprat_results expected;
    if (!load_or_create_expected_results(root + "/data_1024/output.json", results, expected))
    {
        std::cerr << "No previous results to compare to. The current results have been saved instead!\n";
        hpx::finalize();
        return 0;
    }

    const double eps = std::numeric_limits<double>::epsilon() * 1'000'000;
    bool ok = true;
    ok = compare(results.cholesky, expected.cholesky, eps, "cholesky") && ok;
    ok = compare(results.losses, expected.losses, eps, "losses") && ok;
    ok = compare(results.sum, expected.sum, eps, "sum") && ok;
    ok = compare(results.full, expected.full, eps, "full") && ok;
    ok = compare(results.pred, expected.pred, eps, "pred") && ok;

    std::cerr << (ok ? "PASS: distributed results match baseline\n"
                     : "FAIL: distributed results differ from baseline\n");

    hpx::finalize();
    return ok ? 0 : 1;
}

int main(int argc, char *argv[]) { return hpx::init(argc, argv); }
