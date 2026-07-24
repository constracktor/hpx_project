#include "gprat/cpu/adapter_cblas_fp32.hpp"
#include "gprat/cpu/adapter_cblas_fp64.hpp"
#include "gprat/cpu/gp_algorithms.hpp"
#include "gprat/gprat.hpp"
#include "gprat/hyperparameters.hpp"
#include "gprat/kernels.hpp"
#include "gprat/performance_counters.hpp"
#include "gprat/utils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <unistd.h>
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// Helper: build a tile_data from an initializer list
template <typename T>
static gprat::mutable_tile_data<T> make_tile(std::initializer_list<T> vals)
{
    gprat::mutable_tile_data<T> t(vals.size());
    std::size_t i = 0;
    for (const auto &v : vals)
    {
        t.data()[i++] = v;
    }
    return t;
}

template <typename T>
static gprat::const_tile_data<T> make_const_tile(std::initializer_list<T> vals)
{
    // const_tile_data has no mutable operator[], so build via mutable first
    gprat::mutable_tile_data<T> m = make_tile<T>(vals);
    return m;
}

namespace
{
// Starts the HPX runtime on construction and stops it on destruction so that
// stop_hpx_runtime() is always called even when a test assertion fails mid-test.
struct hpx_runtime_guard
{
    hpx_runtime_guard() { gprat::start_hpx_runtime(0, nullptr); }

    ~hpx_runtime_guard() { gprat::stop_hpx_runtime(); }
};
}  // namespace

namespace gprat::test
{

static std::string gprat_data_root()
{
    const char *env = std::getenv("GPRAT_ROOT");
    return env ? env : "../data";
}

// GP_data ///////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("GP_data: sample count", "[unit][gp_data]")
{
    const std::string root = gprat_data_root();
    const std::string path = root + "/data_1024/training_input.txt";

    constexpr std::size_t n = 64;
    constexpr std::size_t n_reg = 8;

    gprat::GP_data d(path, n, n_reg);

    REQUIRE(d.n_samples == n);
    REQUIRE(d.n_regressors == n_reg);
    // load_data allocates n_samples + (n_reg - 1) elements: data starts at offset n_reg-1
    REQUIRE(d.data.size() == n + n_reg - 1);
    REQUIRE(d.file_path == path);
}

TEST_CASE("GP_data: n_reg=1 sample count", "[unit][gp_data]")
{
    const std::string root = gprat_data_root();
    const std::string path = root + "/data_1024/training_input.txt";

    constexpr std::size_t n = 32;
    gprat::GP_data d(path, n, 1);

    REQUIRE(d.data.size() == n + 1 - 1);  // n + (n_reg - 1) with n_reg=1
}

// Tile utilities ////////////////////////////////////////////////////////////////////////////////

TEST_CASE("tile_size: divides evenly", "[unit][tiles]")
{
    REQUIRE(gprat::compute_train_tile_size(1024, 16) == 64);
    REQUIRE(gprat::compute_train_tile_size(512, 8) == 64);
    REQUIRE(gprat::compute_train_tile_size(256, 4) == 64);
}

TEST_CASE("tile_count: divides evenly", "[unit][tiles]")
{
    REQUIRE(gprat::compute_train_tiles(1024, 64) == 16);
    REQUIRE(gprat::compute_train_tiles(512, 64) == 8);
}

TEST_CASE("tile_size: throws on zero tiles", "[unit][tiles]")
{
    REQUIRE_THROWS_AS(gprat::compute_train_tile_size(1024, 0), std::runtime_error);
}

TEST_CASE("tile_count: throws on zero tile_size", "[unit][tiles]")
{
    REQUIRE_THROWS_AS(gprat::compute_train_tiles(1024, 0), std::runtime_error);
}

TEST_CASE("test_tiles: divisible n_test", "[unit][tiles]")
{
    // n_test=512, tile_size=64 → 512 % 64 == 0, so use m_tile_size=64, m_tiles=8
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(512, 16, 64);
    REQUIRE(m_tile_size == 64);
    REQUIRE(m_tiles == 8);
    REQUIRE(m_tiles * m_tile_size == 512);
}

TEST_CASE("test_tiles: non-divisible n_test", "[unit][tiles]")
{
    // n_test=100, tile_size=64 → 100 % 64 != 0, so use m_tiles=16, m_tile_size=100/16
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(100, 16, 64);
    REQUIRE(m_tiles == 16);
    REQUIRE(m_tile_size == 100 / 16);
}

TEST_CASE("tile_size and tile_count: inverses", "[unit][tiles]")
{
    constexpr std::size_t n = 1024;
    constexpr std::size_t tiles = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, tiles);
    const std::size_t recovered = gprat::compute_train_tiles(n, tile_size);
    REQUIRE(recovered == tiles);
}

// Optimizer (CPU) ////////////////////////////////////////////////////////////////////////////////

TEST_CASE("GP::optimize: loss count", "[unit][optimizer][cpu]")
{
    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128;
    constexpr std::size_t n_tiles = 4;
    constexpr std::size_t n_reg = 8;
    constexpr int opt_iter = 5;

    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });

    hpx_runtime_guard hpx_guard;
    const gprat::AdamParams params{ 0.01, 0.9, 0.999, 1e-8, opt_iter };
    const auto losses = gp.optimize(params);

    REQUIRE(losses.size() == static_cast<std::size_t>(opt_iter));
}

TEST_CASE("GP::optimize_step: finite loss", "[unit][optimizer][cpu]")
{
    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128;
    constexpr std::size_t n_tiles = 4;
    constexpr std::size_t n_reg = 8;

    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });

    hpx_runtime_guard hpx_guard;
    gprat::AdamParams params{ 0.01, 0.9, 0.999, 1e-8, 1 };
    const double loss = gp.optimize_step(params, 1);

    REQUIRE(std::isfinite(loss));
}

TEST_CASE("GP::calculate_loss: finite", "[unit][loss][cpu]")
{
    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128;
    constexpr std::size_t n_tiles = 4;
    constexpr std::size_t n_reg = 8;

    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });

    hpx_runtime_guard hpx_guard;
    const double loss = gp.calculate_loss();

    REQUIRE(std::isfinite(loss));
}

TEST_CASE("GP::optimize: loss decreases", "[unit][optimizer][cpu][fragile]")
{
    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128;
    constexpr std::size_t n_tiles = 4;
    constexpr std::size_t n_reg = 8;

    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });

    hpx_runtime_guard hpx_guard;
    const gprat::AdamParams params{ 0.1, 0.9, 0.999, 1e-8, 10 };
    const auto losses = gp.optimize(params);

    // Loss should be finite and have decreased over 10 Adam steps.
    // The decrease is not strictly guaranteed for all hyperparameter settings,
    // but with lr=0.1 and 10 steps on this dataset it is reliable in practice.
    REQUIRE(std::isfinite(losses.back()));
    REQUIRE(losses.front() > losses.back());
}

// SEKParams /////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("SEKParams: size is 3", "[unit][sek]")
{
    gprat::SEKParams p(1.0, 2.0, 0.1);
    REQUIRE(p.size() == 3);
}

TEST_CASE("SEKParams: get_param fields", "[unit][sek]")
{
    gprat::SEKParams p(1.5, 2.5, 0.3);
    REQUIRE_THAT(p.get_param(0), WithinRel(1.5, 1e-12));
    REQUIRE_THAT(p.get_param(1), WithinRel(2.5, 1e-12));
    REQUIRE_THAT(p.get_param(2), WithinRel(0.3, 1e-12));
}

TEST_CASE("SEKParams: set_param mutates", "[unit][sek]")
{
    gprat::SEKParams p(1.0, 1.0, 0.1);
    p.set_param(0, 3.0);
    p.set_param(1, 4.0);
    p.set_param(2, 0.5);
    REQUIRE_THAT(p.lengthscale, WithinRel(3.0, 1e-12));
    REQUIRE_THAT(p.vertical_lengthscale, WithinRel(4.0, 1e-12));
    REQUIRE_THAT(p.noise_variance, WithinRel(0.5, 1e-12));
}

TEST_CASE("SEKParams: get_param throws", "[unit][sek]")
{
    gprat::SEKParams p(1.0, 1.0, 0.1);
    REQUIRE_THROWS_AS(p.get_param(3), std::invalid_argument);
}

TEST_CASE("SEKParams: set_param throws", "[unit][sek]")
{
    gprat::SEKParams p(1.0, 1.0, 0.1);
    REQUIRE_THROWS_AS(p.set_param(3, 0.0), std::invalid_argument);
}

TEST_CASE("SEKParams: m_T and w_T size", "[unit][sek]")
{
    gprat::SEKParams p(1.0, 1.0, 0.1);
    REQUIRE(p.m_T.size() == 3);
    REQUIRE(p.w_T.size() == 3);
}

// AdamParams ////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("AdamParams: default values", "[unit][adam]")
{
    gprat::AdamParams p;
    REQUIRE_THAT(p.learning_rate, WithinRel(0.001, 1e-12));
    REQUIRE_THAT(p.beta1, WithinRel(0.9, 1e-12));
    REQUIRE_THAT(p.beta2, WithinRel(0.999, 1e-12));
    REQUIRE_THAT(p.epsilon, WithinRel(1e-8, 1e-12));
    REQUIRE(p.opt_iter == 0);
}

TEST_CASE("AdamParams: repr fields", "[unit][adam]")
{
    gprat::AdamParams p(0.01, 0.9, 0.999, 1e-8, 5);
    const auto s = p.repr();
    REQUIRE_THAT(s, ContainsSubstring("learning_rate"));
    REQUIRE_THAT(s, ContainsSubstring("beta1"));
    REQUIRE_THAT(s, ContainsSubstring("beta2"));
    REQUIRE_THAT(s, ContainsSubstring("epsilon"));
    REQUIRE_THAT(s, ContainsSubstring("opt_iter"));
}

// GP_data error handling /////////////////////////////////////////////////////////////////////////

TEST_CASE("GP_data: throws on missing file", "[unit][gp_data]")
{
    REQUIRE_THROWS_AS(gprat::GP_data("/nonexistent/path/file.txt", 10, 4), std::runtime_error);
}

// GP accessors and repr //////////////////////////////////////////////////////////////////////////

TEST_CASE("GP: training data round-trip", "[unit][gp]")
{
    const std::string root = gprat_data_root();

    constexpr std::size_t n = 64, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });

    REQUIRE(gp.get_training_input() == train_in.data);
    REQUIRE(gp.get_training_output() == train_out.data);
}

TEST_CASE("GP: repr fields", "[unit][gp]")
{
    const std::string root = gprat_data_root();

    constexpr std::size_t n = 64, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });

    const auto s = gp.repr();
    REQUIRE_THAT(s, ContainsSubstring("lengthscale"));
    REQUIRE_THAT(s, ContainsSubstring("n_tiles"));
}

// GP prediction shapes ///////////////////////////////////////////////////////////////////////////

TEST_CASE("GP::predict: output size", "[unit][gp][predict]")
{
    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8, n_test = 128;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });

    hpx_runtime_guard hpx_guard;
    const auto pred = gp.predict(test_in.data, m_tiles, m_tile_size);
    const auto pred_unc = gp.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);
    const auto pred_cov = gp.predict_with_full_cov(test_in.data, m_tiles, m_tile_size);

    REQUIRE(pred.size() == n_test);
    REQUIRE(pred_unc.size() == 2);
    REQUIRE(pred_unc[0].size() == n_test);
    REQUIRE(pred_unc[1].size() == n_test);
    // predict_with_full_cov returns {mean, diagonal(Sigma)} — same shape as predict_with_uncertainty
    REQUIRE(pred_cov.size() == 2);
    REQUIRE(pred_cov[0].size() == n_test);
    REQUIRE(pred_cov[1].size() == n_test);
}

TEST_CASE("GP::cholesky: tile structure", "[unit][gp][cholesky]")
{
    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });

    hpx_runtime_guard hpx_guard;
    const auto L = gp.cholesky();

    // n_tiles × n_tiles blocks stored as flat list of n_tiles^2 tiles
    REQUIRE(L.size() == n_tiles * n_tiles);
    REQUIRE(L[0].size() == tile_size * tile_size);
}

// GP trainable mask //////////////////////////////////////////////////////////////////////////////

TEST_CASE("GP::optimize: no trainable params", "[unit][optimizer][cpu]")
{
    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { false, false, false });

    hpx_runtime_guard hpx_guard;
    const gprat::AdamParams params{ 0.1, 0.9, 0.999, 1e-8, 5 };
    const auto losses = gp.optimize(params);

    // All losses should be equal — no parameters moved
    for (std::size_t i = 1; i < losses.size(); ++i)
    {
        REQUIRE_THAT(losses[i], WithinRel(losses[0], 1e-10));
    }
}

// GP kernel_params live mutation /////////////////////////////////////////////////////////////////

TEST_CASE("GP::calculate_loss: sensitive to kernel_params", "[unit][gp][loss]")
{
    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });

    hpx_runtime_guard hpx_guard;
    const double loss_before = gp.calculate_loss();
    gp.kernel_params.lengthscale = 5.0;
    const double loss_after = gp.calculate_loss();

    // Different hyperparameters must produce a different loss value.
    // We use WithinAbs to check the absolute difference is non-trivially non-zero.
    REQUIRE_THAT(std::abs(loss_before - loss_after), !WithinAbs(0.0, 1e-10));
}

// guess_good_tile_count_per_dimension ////////////////////////////////////////////////////////////

TEST_CASE("tile_count_per_dim: 1 for small n", "[unit][tiles]")
{
    // n < 2^8 = 256 → always returns 1
    REQUIRE(gprat::guess_good_tile_count_per_dimension(100) == 1);
    REQUIRE(gprat::guess_good_tile_count_per_dimension(1) == 1);
}

TEST_CASE("tile_count_per_dim: positive for medium n", "[unit][tiles]")
{
    hpx_runtime_guard hpx_guard;
    const std::size_t count = gprat::guess_good_tile_count_per_dimension(1 << 14);
    REQUIRE(count >= 1);
}

// compiled_with_cuda / compiled_with_sycl ////////////////////////////////////////////////////////

#if !GPRAT_WITH_CUDA
TEST_CASE("compiled_with_cuda: false", "[unit][target]") { REQUIRE_FALSE(gprat::compiled_with_cuda()); }
#else
TEST_CASE("compiled_with_cuda: true", "[unit][target]") { REQUIRE(gprat::compiled_with_cuda()); }
#endif

#if !GPRAT_WITH_SYCL
TEST_CASE("compiled_with_sycl: false", "[unit][target]") { REQUIRE_FALSE(gprat::compiled_with_sycl()); }
#else
TEST_CASE("compiled_with_sycl: true", "[unit][target]") { REQUIRE(gprat::compiled_with_sycl()); }
#endif

// GP GPU constructor throws without CUDA/SYCL ////////////////////////////////////////////////////

#if !GPRAT_WITH_CUDA && !GPRAT_WITH_SYCL
TEST_CASE("GP GPU: throws without CUDA/SYCL", "[unit][gp]")
{
    const std::string root = gprat_data_root();

    constexpr std::size_t n = 64, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    REQUIRE_THROWS_AS(
        (gprat::GP(
            train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1)),
        std::runtime_error);
}
#endif

// print_vector ///////////////////////////////////////////////////////////////////////////////////

TEST_CASE("print_vector: basic range", "[unit][utils]")
{
    const std::vector<double> v = { 1.0, 2.0, 3.0 };
    std::streambuf *old = std::cout.rdbuf();
    std::ostringstream buf;
    std::cout.rdbuf(buf.rdbuf());
    gprat::print_vector(v, 0, 3, ",");
    std::cout.rdbuf(old);
    REQUIRE_THAT(buf.str(), ContainsSubstring("1") && ContainsSubstring("2") && ContainsSubstring("3"));
}

TEST_CASE("print_vector: negative start", "[unit][utils]")
{
    const std::vector<double> v = { 10.0, 20.0, 30.0 };
    std::streambuf *old = std::cout.rdbuf();
    std::ostringstream buf;
    std::cout.rdbuf(buf.rdbuf());
    gprat::print_vector(v, -2, 3, " ");  // start = 3 - 2 = 1 → prints 20 30
    std::cout.rdbuf(old);
    REQUIRE_THAT(buf.str(), ContainsSubstring("20"));
}

TEST_CASE("print_vector: negative end", "[unit][utils]")
{
    const std::vector<double> v = { 10.0, 20.0, 30.0 };
    std::streambuf *old = std::cout.rdbuf();
    std::ostringstream buf;
    std::cout.rdbuf(buf.rdbuf());
    gprat::print_vector(v, 0, -1, " ");  // end = 3 + 1 - 1 = 3
    std::cout.rdbuf(old);
    REQUIRE_THAT(buf.str(), ContainsSubstring("10"));
}

TEST_CASE("print_vector: end clamped", "[unit][utils]")
{
    const std::vector<double> v = { 5.0, 6.0 };
    std::streambuf *old_out = std::cout.rdbuf();
    std::streambuf *old_err = std::cerr.rdbuf();
    std::ostringstream buf_out, buf_err;
    std::cout.rdbuf(buf_out.rdbuf());
    std::cerr.rdbuf(buf_err.rdbuf());
    gprat::print_vector(v, 0, 100, ",");  // end clamped to 2
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    REQUIRE_THAT(buf_out.str(), ContainsSubstring("5"));
}

TEST_CASE("print_vector: invalid range", "[unit][utils]")
{
    const std::vector<double> v = { 1.0, 2.0, 3.0 };
    std::streambuf *old_err = std::cerr.rdbuf();
    std::ostringstream buf;
    std::cerr.rdbuf(buf.rdbuf());
    gprat::print_vector(v, 2, 1, ",");  // start >= end → invalid
    std::cerr.rdbuf(old_err);
    REQUIRE_THAT(buf.str(), ContainsSubstring("Invalid"));
}

// fp32 BLAS adapters /////////////////////////////////////////////////////////////////////////////

TEST_CASE("fp32 BLAS: basic ops", "[unit][blas][fp32]")
{
    hpx_runtime_guard hpx_guard;

    // potrf: Cholesky of 2x2 identity → L = I
    {
        auto A = make_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        const auto L = gprat::potrf(A, 2);
        REQUIRE_THAT(static_cast<double>(L.data()[0]), WithinAbs(1.0, 1e-5));
        REQUIRE_THAT(static_cast<double>(L.data()[3]), WithinAbs(1.0, 1e-5));
    }

    // dot: 1*4 + 2*5 + 3*6 = 32
    {
        const std::vector<float> a = { 1.0f, 2.0f, 3.0f };
        const std::vector<float> b = { 4.0f, 5.0f, 6.0f };
        REQUIRE_THAT(static_cast<double>(gprat::dot(std::span<const float>(a), std::span<const float>(b), 3)),
                     WithinAbs(32.0, 1e-4));
    }

    // axpy: y -= x  (alpha = -1 by convention in gprat)
    {
        auto y = make_tile<float>({ 10.0f, 20.0f, 30.0f });
        auto x = make_const_tile<float>({ 1.0f, 2.0f, 3.0f });
        const auto r = gprat::axpy(y, x, 3);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(9.0, 1e-5));
        REQUIRE_THAT(static_cast<double>(r.data()[1]), WithinAbs(18.0, 1e-5));
        REQUIRE_THAT(static_cast<double>(r.data()[2]), WithinAbs(27.0, 1e-5));
    }

    // syrk: C -= B*B^T  (alpha = -1), C=0, B=diag(1,2) → C = -diag(1,4)
    {
        auto C = make_tile<float>({ 0.0f, 0.0f, 0.0f, 0.0f });
        auto B = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 2.0f });
        const auto r = gprat::syrk(C, B, 2);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(-1.0, 1e-5));
        REQUIRE_THAT(static_cast<double>(r.data()[3]), WithinAbs(-4.0, 1e-5));
    }

    // gemm: C -= A*B  (alpha=-1), C=0, A=I, B=diag(2,3) → C = -diag(2,3)
    {
        auto A = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto B = make_const_tile<float>({ 2.0f, 0.0f, 0.0f, 3.0f });
        auto C = make_tile<float>({ 0.0f, 0.0f, 0.0f, 0.0f });
        const auto r = gprat::gemm(A, B, C, 2, 2, 2, gprat::Blas_no_trans, gprat::Blas_no_trans);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(-2.0, 1e-5));
        REQUIRE_THAT(static_cast<double>(r.data()[3]), WithinAbs(-3.0, 1e-5));
    }

    // trsm: I * X = B → X = B
    {
        auto L = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto B = make_tile<float>({ 5.0f, 7.0f, 9.0f, 11.0f });
        const auto X = gprat::trsm(L, B, 2, 2, gprat::Blas_no_trans, gprat::Blas_left);
        REQUIRE_THAT(static_cast<double>(X.data()[0]), WithinAbs(5.0, 1e-5));
        REQUIRE_THAT(static_cast<double>(X.data()[1]), WithinAbs(7.0, 1e-5));
    }

    // trsv: I * x = b → x = b
    {
        auto L = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto b = make_tile<float>({ 3.0f, 4.0f });
        const auto x = gprat::trsv(L, b, 2, gprat::Blas_no_trans);
        REQUIRE_THAT(static_cast<double>(x.data()[0]), WithinAbs(3.0, 1e-5));
        REQUIRE_THAT(static_cast<double>(x.data()[1]), WithinAbs(4.0, 1e-5));
    }

    // gemv: I * [1,2] = [1,2]
    {
        auto A = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto x = make_const_tile<float>({ 1.0f, 2.0f });
        auto y = make_tile<float>({ 0.0f, 0.0f });
        const auto r = gprat::gemv(A, x, y, 2, 2, gprat::Blas_add, gprat::Blas_no_trans);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(1.0, 1e-5));
        REQUIRE_THAT(static_cast<double>(r.data()[1]), WithinAbs(2.0, 1e-5));
    }

    // dot_diag_syrk: r[j] += dot(col_j(A), col_j(A))
    // A = [[1,0],[2,0]] (col-major 2x2), M=2, N=2 → r[0] += 1²+2²=5, r[1] += 0
    {
        // A stored row-major 2x2: rows=[1,0],[2,0] → col 0 = [1,2], col 1 = [0,0]
        auto A = make_const_tile<float>({ 1.0f, 0.0f, 2.0f, 0.0f });
        auto r = make_tile<float>({ 0.0f, 0.0f });
        const auto out = gprat::dot_diag_syrk(A, r, 2, 2);
        REQUIRE_THAT(static_cast<double>(out.data()[0]), WithinAbs(5.0, 1e-4));  // 1² + 2²
        REQUIRE_THAT(static_cast<double>(out.data()[1]), WithinAbs(0.0, 1e-4));  // 0² + 0²
    }

    // dot_diag_gemm: r[i] += dot(row_i(A), col_i(B))
    // A=I2, B=I2 → r[i] += 1, so r = [1, 1]
    {
        auto A = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto B = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto r = make_tile<float>({ 0.0f, 0.0f });
        const auto out = gprat::dot_diag_gemm(A, B, r, 2, 2);
        REQUIRE_THAT(static_cast<double>(out.data()[0]), WithinAbs(1.0, 1e-4));
        REQUIRE_THAT(static_cast<double>(out.data()[1]), WithinAbs(1.0, 1e-4));
    }
}

// HPX runtime suspend/resume /////////////////////////////////////////////////////////////////////

TEST_CASE("hpx: suspend and resume", "[unit][hpx]")
{
    hpx_runtime_guard hpx_guard;
    // Suspend pauses HPX worker threads without stopping the runtime.
    // Resume brings them back. A loss calculation after resume confirms the
    // runtime is fully functional again.
    gprat::suspend_hpx_runtime();
    gprat::resume_hpx_runtime();

    const std::string root = gprat_data_root();
    constexpr std::size_t n = 64, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    REQUIRE(std::isfinite(gp.calculate_loss()));
}

// gpu_algorithms coverage: gen_tile_identity, gen_tile_zeros, gen_tile_output //////////////////

TEST_CASE("GP::optimize: noise-only trainable", "[unit][optimizer][cpu]")
{
    // Optimising with only noise_variance trainable triggers the identity-tile
    // assembly path in the gradient computation for the noise parameter.
    const std::string root = gprat_data_root();
    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { false, false, true });
    hpx_runtime_guard hpx_guard;
    const gprat::AdamParams params{ 0.01, 0.9, 0.999, 1e-8, 3 };
    const auto losses = gp.optimize(params);
    REQUIRE(losses.size() == 3);
    REQUIRE(std::isfinite(losses.back()));
}

// compute_error_norm /////////////////////////////////////////////////////////////////////////////

TEST_CASE("compute_error_norm: exact match", "[unit][gp_algorithms]")
{
    // Two identical tiles → error norm should be 0
    const std::size_t n_tiles = 2, tile_size = 3;
    const std::vector<double> b = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const std::vector<std::vector<double>> tiles = { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
    REQUIRE_THAT(gprat::cpu::compute_error_norm(n_tiles, tile_size, b, tiles), WithinAbs(0.0, 1e-12));
}

TEST_CASE("compute_error_norm: known residual", "[unit][gp_algorithms]")
{
    // b = [0,0], tiles = [[1,0]] → error = sqrt(1²+0²) = 1
    const std::vector<double> b = { 0.0, 0.0 };
    const std::vector<std::vector<double>> tiles = { { 1.0, 0.0 } };
    REQUIRE_THAT(gprat::cpu::compute_error_norm(1, 2, b, tiles), WithinAbs(1.0, 1e-12));
}

TEST_CASE("compute_error_norm: multi-tile", "[unit][gp_algorithms]")
{
    // b = [3,4,0,0], tiles = [[0,0],[0,0]] → error = sqrt(9+16) = 5
    const std::vector<double> b = { 3.0, 4.0, 0.0, 0.0 };
    const std::vector<std::vector<double>> tiles = { { 0.0, 0.0 }, { 0.0, 0.0 } };
    REQUIRE_THAT(gprat::cpu::compute_error_norm(2, 2, b, tiles), WithinAbs(5.0, 1e-12));
}

// guess_good_tile_count_per_dimension: large-n paths /////////////////////////////////////////////

TEST_CASE("tile_count_per_dim: positive for large n", "[unit][tiles]")
{
    // n >= 2^18: enters the min(hw_concurrency, n/256) branch when hw_concurrency >= 32,
    // or returns 16 when hw_concurrency < 32. Either way count >= 1 and count <= n/256.
    hpx_runtime_guard hpx_guard;
    const std::size_t n = 1 << 18;
    const std::size_t count = gprat::guess_good_tile_count_per_dimension(n);
    REQUIRE(count >= 1);
    REQUIRE(count <= n / 256);
}

// load_data error path ///////////////////////////////////////////////////////////////////////////

TEST_CASE("load_data: throws on short file", "[unit][utils]")
{
    // Write a file with only 2 values, then try to load 5.
    // Use a process-unique path so parallel test runners don't collide and so
    // the file is cleaned up even if REQUIRE_THROWS_AS propagates an exception.
    std::string tmp_template =
        std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/gprat_test_XXXXXX";
    std::vector<char> tmp_buf(tmp_template.begin(), tmp_template.end());
    tmp_buf.push_back('\0');
    {
        const int fd = ::mkstemp(tmp_buf.data());
        REQUIRE(fd != -1);
        ::close(fd);
    }
    const std::string tmp(tmp_buf.data());

    struct Cleanup
    {
        const std::string &path;

        ~Cleanup() { std::remove(path.c_str()); }
    } cleanup{ tmp };

    {
        std::ofstream f(tmp);
        REQUIRE(f.is_open());
        f << "1.0\n2.0\n";
    }
    REQUIRE_THROWS_AS(gprat::GP_data(tmp, 5, 1), std::runtime_error);
}

// print_vector: start clamped to 0 after negative wrap //////////////////////////////////////////

TEST_CASE("print_vector: deeply negative start", "[unit][utils]")
{
    // start=-10 on a 3-element vec → start = 3 + (-10) = -7, clamped to 0
    const std::vector<double> v = { 7.0, 8.0, 9.0 };
    std::streambuf *old = std::cout.rdbuf();
    std::ostringstream buf;
    std::cout.rdbuf(buf.rdbuf());
    gprat::print_vector(v, -10, 3, ",");
    std::cout.rdbuf(old);
    // start clamped to 0, end=3 → all elements printed
    REQUIRE_THAT(buf.str(), ContainsSubstring("7"));
}

// GPU tests (NVIDIA only) ////////////////////////////////////////////////////////////////////////
//
// Each test calls SKIP() immediately if GPRat was compiled without CUDA or if
// no NVIDIA device is detected at runtime, so they are safe to include in
// every build.  When GPRAT_WITH_CUDA=ON and a GPU is present the tests run
// in full and compare GPU results against the CPU reference.

namespace
{
// Returns the number of visible CUDA devices (0 when CUDA is absent).
int cuda_device_count()
{
#if GPRAT_WITH_CUDA
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
#else
    return 0;
#endif
}
}  // namespace

// Macro that skips the test if CUDA is unavailable or no GPU is present.
#define GPRAT_SKIP_IF_NO_GPU()                                                                                         \
    do {                                                                                                               \
        if (!gprat::compiled_with_cuda())                                                                              \
            SKIP("GPRat not compiled with CUDA support");                                                              \
        if (cuda_device_count() == 0)                                                                                  \
            SKIP("No NVIDIA GPU detected");                                                                            \
    } while (false)

TEST_CASE("GP GPU: constructor", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    // Should not throw when a real GPU is present.
    REQUIRE_NOTHROW((gprat::GP(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1)));
}

TEST_CASE("GP::predict: GPU matches CPU", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8, n_test = 64;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const auto cpu_pred = gp_cpu.predict(test_in.data, m_tiles, m_tile_size);
    const auto gpu_pred = gp_gpu.predict(test_in.data, m_tiles, m_tile_size);

    REQUIRE(cpu_pred.size() == n_test);
    REQUIRE(gpu_pred.size() == n_test);
    for (std::size_t i = 0; i < n_test; ++i)
    {
        REQUIRE_THAT(gpu_pred[i], WithinRel(cpu_pred[i], 1e-4));
    }
}

TEST_CASE("GP::predict_with_uncertainty: GPU matches CPU", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8, n_test = 64;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const auto cpu_unc = gp_cpu.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);
    const auto gpu_unc = gp_gpu.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);

    // cpu_unc[0] = mean, cpu_unc[1] = variance
    REQUIRE(gpu_unc.size() == 2);
    REQUIRE(gpu_unc[0].size() == n_test);
    REQUIRE(gpu_unc[1].size() == n_test);
    for (std::size_t i = 0; i < n_test; ++i)
    {
        REQUIRE_THAT(gpu_unc[0][i], WithinRel(cpu_unc[0][i], 1e-4));
        REQUIRE_THAT(gpu_unc[1][i], WithinRel(cpu_unc[1][i], 1e-4));
    }
}

TEST_CASE("GP::predict_with_full_cov: GPU matches CPU", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8, n_test = 64;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const auto cpu_cov = gp_cpu.predict_with_full_cov(test_in.data, m_tiles, m_tile_size);
    const auto gpu_cov = gp_gpu.predict_with_full_cov(test_in.data, m_tiles, m_tile_size);

    REQUIRE(gpu_cov.size() == 2);
    REQUIRE(gpu_cov[0].size() == n_test);
    for (std::size_t i = 0; i < n_test; ++i)
    {
        REQUIRE_THAT(gpu_cov[0][i], WithinRel(cpu_cov[0][i], 1e-4));
    }
}

TEST_CASE("GP::calculate_loss: GPU matches CPU", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const double cpu_loss = gp_cpu.calculate_loss();
    const double gpu_loss = gp_gpu.calculate_loss();

    REQUIRE(std::isfinite(gpu_loss));
    REQUIRE_THAT(gpu_loss, WithinRel(cpu_loss, 1e-4));
}

TEST_CASE("GP::cholesky: GPU tile count", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const auto cpu_L = gp_cpu.cholesky();
    const auto gpu_L = gp_gpu.cholesky();

    REQUIRE(gpu_L.size() == cpu_L.size());
    REQUIRE(gpu_L[0].size() == cpu_L[0].size());
    // Diagonal tiles of L should match CPU within tolerance
    for (std::size_t t = 0; t < n_tiles; ++t)
    {
        const std::size_t diag = t * n_tiles + t;
        for (std::size_t e = 0; e < tile_size * tile_size; ++e)
        {
            REQUIRE_THAT(gpu_L[diag].data()[e], WithinRel(cpu_L[diag].data()[e], 1e-4));
        }
    }
}

// fp64 BLAS: additional transpose/side/alpha paths ///////////////////////////////////////////////

TEST_CASE("fp64 BLAS: basic ops", "[unit][blas][fp64]")
{
    hpx_runtime_guard hpx_guard;

    // potrf: 2x2 identity → L = I
    {
        auto A = make_tile<double>({ 1.0, 0.0, 0.0, 1.0 });
        const auto L = gprat::potrf(A, 2);
        REQUIRE_THAT(static_cast<double>(L.data()[0]), WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(L.data()[3]), WithinAbs(1.0, 1e-10));
    }

    // dot: 1*4 + 2*5 + 3*6 = 32
    {
        const std::vector<double> a = { 1.0, 2.0, 3.0 };
        const std::vector<double> b = { 4.0, 5.0, 6.0 };
        REQUIRE_THAT(gprat::dot(std::span<const double>(a), std::span<const double>(b), 3), WithinAbs(32.0, 1e-10));
    }

    // axpy: y -= x
    {
        auto y = make_tile<double>({ 10.0, 20.0, 30.0 });
        auto x = make_const_tile<double>({ 1.0, 2.0, 3.0 });
        const auto r = gprat::axpy(y, x, 3);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(9.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(r.data()[2]), WithinAbs(27.0, 1e-10));
    }

    // syrk: C -= B*B^T, C=0, B=diag(1,2) → C[0,0]=-1, C[1,1]=-4
    {
        auto C = make_tile<double>({ 0.0, 0.0, 0.0, 0.0 });
        auto B = make_const_tile<double>({ 1.0, 0.0, 0.0, 2.0 });
        const auto r = gprat::syrk(C, B, 2);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(-1.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(r.data()[3]), WithinAbs(-4.0, 1e-10));
    }

    // gemm no-trans/no-trans: C -= A*B, A=I, B=diag(2,3)
    {
        auto A = make_const_tile<double>({ 1.0, 0.0, 0.0, 1.0 });
        auto B = make_const_tile<double>({ 2.0, 0.0, 0.0, 3.0 });
        auto C = make_tile<double>({ 0.0, 0.0, 0.0, 0.0 });
        const auto r = gprat::gemm(A, B, C, 2, 2, 2, gprat::Blas_no_trans, gprat::Blas_no_trans);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(-2.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(r.data()[3]), WithinAbs(-3.0, 1e-10));
    }

    // gemm with trans_A: C -= A^T*B, A=[[1,2],[0,0]], B=I → C[0,0]=-1, C[1,0]=-2
    {
        auto A = make_const_tile<double>({ 1.0, 2.0, 0.0, 0.0 });
        auto B = make_const_tile<double>({ 1.0, 0.0, 0.0, 1.0 });
        auto C = make_tile<double>({ 0.0, 0.0, 0.0, 0.0 });
        const auto r = gprat::gemm(A, B, C, 2, 2, 2, gprat::Blas_trans, gprat::Blas_no_trans);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(-1.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(r.data()[2]), WithinAbs(-2.0, 1e-10));
    }

    // gemm with trans_B: C -= A*B^T, A=I, B=[[1,0],[2,0]] → C -= [[1,2],[0,0]]
    {
        auto A = make_const_tile<double>({ 1.0, 0.0, 0.0, 1.0 });
        auto B = make_const_tile<double>({ 1.0, 0.0, 2.0, 0.0 });
        auto C = make_tile<double>({ 0.0, 0.0, 0.0, 0.0 });
        const auto r = gprat::gemm(A, B, C, 2, 2, 2, gprat::Blas_no_trans, gprat::Blas_trans);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(-1.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(r.data()[1]), WithinAbs(-2.0, 1e-10));
    }

    // trsm left no-trans: L*X = B → X = L^{-1}*B, L=I, B=[[5,7],[9,11]]
    {
        auto L = make_const_tile<double>({ 1.0, 0.0, 0.0, 1.0 });
        auto B = make_tile<double>({ 5.0, 7.0, 9.0, 11.0 });
        const auto X = gprat::trsm(L, B, 2, 2, gprat::Blas_no_trans, gprat::Blas_left);
        REQUIRE_THAT(static_cast<double>(X.data()[0]), WithinAbs(5.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(X.data()[1]), WithinAbs(7.0, 1e-10));
    }

    // trsm with trans: L^T * X = B, L=[[2,0],[1,4]], L^T=[[2,1],[0,4]]
    // Row-major B = [10,6,4,8] → col0=[10,4], col1=[6,8]
    // X col0: x1=1, x0=(10-1)/2=4.5 → X[0,0]=4.5
    // X col1: x1=2, x0=(6-2)/2=2    → X[0,1]=2.0
    {
        auto L = make_const_tile<double>({ 2.0, 0.0, 1.0, 4.0 });
        auto B = make_tile<double>({ 10.0, 6.0, 4.0, 8.0 });
        const auto X = gprat::trsm(L, B, 2, 2, gprat::Blas_trans, gprat::Blas_left);
        REQUIRE_THAT(static_cast<double>(X.data()[0]), WithinAbs(4.5, 1e-10));
        REQUIRE_THAT(static_cast<double>(X.data()[1]), WithinAbs(2.0, 1e-10));
    }

    // trsm right: X * L = B, L=I → X=B
    {
        auto L = make_const_tile<double>({ 1.0, 0.0, 0.0, 1.0 });
        auto B = make_tile<double>({ 2.0, 3.0, 4.0, 5.0 });
        const auto X = gprat::trsm(L, B, 2, 2, gprat::Blas_no_trans, gprat::Blas_right);
        REQUIRE_THAT(static_cast<double>(X.data()[0]), WithinAbs(2.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(X.data()[3]), WithinAbs(5.0, 1e-10));
    }

    // trsv no-trans: L*x = b, L=I, b=[3,4] → x=[3,4]
    {
        auto L = make_const_tile<double>({ 1.0, 0.0, 0.0, 1.0 });
        auto b = make_tile<double>({ 3.0, 4.0 });
        const auto x = gprat::trsv(L, b, 2, gprat::Blas_no_trans);
        REQUIRE_THAT(static_cast<double>(x.data()[0]), WithinAbs(3.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(x.data()[1]), WithinAbs(4.0, 1e-10));
    }

    // trsv trans: L^T*x = b, L=[[2,0],[1,4]], L^T=[[2,1],[0,4]]
    // b=[10,4]: x[1]=1, x[0]=(10-1)/2=4.5
    {
        auto L = make_const_tile<double>({ 2.0, 0.0, 1.0, 4.0 });
        auto b = make_tile<double>({ 10.0, 4.0 });
        const auto x = gprat::trsv(L, b, 2, gprat::Blas_trans);
        REQUIRE_THAT(static_cast<double>(x.data()[0]), WithinAbs(4.5, 1e-10));
        REQUIRE_THAT(static_cast<double>(x.data()[1]), WithinAbs(1.0, 1e-10));
    }

    // gemv Blas_add no-trans: b += A*x, A=I, x=[1,2], b=[3,4] → b=[4,6]
    {
        auto A = make_const_tile<double>({ 1.0, 0.0, 0.0, 1.0 });
        auto x = make_const_tile<double>({ 1.0, 2.0 });
        auto b = make_tile<double>({ 3.0, 4.0 });
        const auto r = gprat::gemv(A, x, b, 2, 2, gprat::Blas_add, gprat::Blas_no_trans);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(4.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(r.data()[1]), WithinAbs(6.0, 1e-10));
    }

    // gemv Blas_substract no-trans: b -= A*x, A=I, x=[1,2], b=[5,7] → b=[4,5]
    {
        auto A = make_const_tile<double>({ 1.0, 0.0, 0.0, 1.0 });
        auto x = make_const_tile<double>({ 1.0, 2.0 });
        auto b = make_tile<double>({ 5.0, 7.0 });
        const auto r = gprat::gemv(A, x, b, 2, 2, gprat::Blas_substract, gprat::Blas_no_trans);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(4.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(r.data()[1]), WithinAbs(5.0, 1e-10));
    }

    // gemv trans: b += A^T*x, A=[[1,0],[0,2]], x=[3,4], b=[0,0]
    // A^T = [[1,0],[0,2]] (symmetric), A^T*x = [3, 8]
    {
        auto A = make_const_tile<double>({ 1.0, 0.0, 0.0, 2.0 });
        auto x = make_const_tile<double>({ 3.0, 4.0 });
        auto b = make_tile<double>({ 0.0, 0.0 });
        const auto r = gprat::gemv(A, x, b, 2, 2, gprat::Blas_add, gprat::Blas_trans);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(3.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(r.data()[1]), WithinAbs(8.0, 1e-10));
    }

    // dot_diag_syrk fp64: r[j] += sum_i A[i,j]^2
    {
        auto A = make_const_tile<double>({ 1.0, 0.0, 2.0, 0.0 });
        auto r = make_tile<double>({ 0.0, 0.0 });
        const auto out = gprat::dot_diag_syrk(A, r, 2, 2);
        REQUIRE_THAT(static_cast<double>(out.data()[0]), WithinAbs(5.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(out.data()[1]), WithinAbs(0.0, 1e-10));
    }

    // dot_diag_gemm fp64: r[i] += dot(row_i(A), col_i(B)), A=B=I → r=[1,1]
    {
        auto A = make_const_tile<double>({ 1.0, 0.0, 0.0, 1.0 });
        auto B = make_const_tile<double>({ 1.0, 0.0, 0.0, 1.0 });
        auto r = make_tile<double>({ 0.0, 0.0 });
        const auto out = gprat::dot_diag_gemm(A, B, r, 2, 2);
        REQUIRE_THAT(static_cast<double>(out.data()[0]), WithinAbs(1.0, 1e-10));
        REQUIRE_THAT(static_cast<double>(out.data()[1]), WithinAbs(1.0, 1e-10));
    }
}

// fp32 BLAS: additional transpose/side/alpha paths //////////////////////////////////////////////

TEST_CASE("fp32 BLAS: transpose and side variants", "[unit][blas][fp32]")
{
    hpx_runtime_guard hpx_guard;

    // gemm trans_A: C -= A^T*B, A=[[1,2],[0,0]], B=I
    {
        auto A = make_const_tile<float>({ 1.0f, 2.0f, 0.0f, 0.0f });
        auto B = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto C = make_tile<float>({ 0.0f, 0.0f, 0.0f, 0.0f });
        const auto r = gprat::gemm(A, B, C, 2, 2, 2, gprat::Blas_trans, gprat::Blas_no_trans);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(-1.0, 1e-5));
        REQUIRE_THAT(static_cast<double>(r.data()[2]), WithinAbs(-2.0, 1e-5));
    }

    // gemm trans_B: C -= A*B^T, A=I, B=[[1,0],[2,0]]
    {
        auto A = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto B = make_const_tile<float>({ 1.0f, 0.0f, 2.0f, 0.0f });
        auto C = make_tile<float>({ 0.0f, 0.0f, 0.0f, 0.0f });
        const auto r = gprat::gemm(A, B, C, 2, 2, 2, gprat::Blas_no_trans, gprat::Blas_trans);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(-1.0, 1e-5));
        REQUIRE_THAT(static_cast<double>(r.data()[1]), WithinAbs(-2.0, 1e-5));
    }

    // trsm trans: L^T*X = B, L=[[2,0],[1,4]], L^T=[[2,1],[0,4]]
    // Row-major B=[10,6,4,8] → col0=[10,4], col1=[6,8]
    // X col0: x1=1, x0=4.5 → X[0,0]=4.5; X col1: x1=2, x0=2 → X[0,1]=2.0
    {
        auto L = make_const_tile<float>({ 2.0f, 0.0f, 1.0f, 4.0f });
        auto B = make_tile<float>({ 10.0f, 6.0f, 4.0f, 8.0f });
        const auto X = gprat::trsm(L, B, 2, 2, gprat::Blas_trans, gprat::Blas_left);
        REQUIRE_THAT(static_cast<double>(X.data()[0]), WithinAbs(4.5, 1e-5));
        REQUIRE_THAT(static_cast<double>(X.data()[1]), WithinAbs(2.0, 1e-5));
    }

    // trsm right: X*L = B, L=I → X=B
    {
        auto L = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto B = make_tile<float>({ 2.0f, 3.0f, 4.0f, 5.0f });
        const auto X = gprat::trsm(L, B, 2, 2, gprat::Blas_no_trans, gprat::Blas_right);
        REQUIRE_THAT(static_cast<double>(X.data()[0]), WithinAbs(2.0, 1e-5));
        REQUIRE_THAT(static_cast<double>(X.data()[3]), WithinAbs(5.0, 1e-5));
    }

    // trsv trans: L^T*x = b, L=[[2,0],[1,4]], L^T=[[2,1],[0,4]]
    // b=[10,4]: x[1]=1, x[0]=(10-1)/2=4.5
    {
        auto L = make_const_tile<float>({ 2.0f, 0.0f, 1.0f, 4.0f });
        auto b = make_tile<float>({ 10.0f, 4.0f });
        const auto x = gprat::trsv(L, b, 2, gprat::Blas_trans);
        REQUIRE_THAT(static_cast<double>(x.data()[0]), WithinAbs(4.5, 1e-5));
        REQUIRE_THAT(static_cast<double>(x.data()[1]), WithinAbs(1.0, 1e-5));
    }

    // gemv Blas_substract: b -= A*x, A=I, x=[1,2], b=[5,7] → b=[4,5]
    {
        auto A = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto x = make_const_tile<float>({ 1.0f, 2.0f });
        auto b = make_tile<float>({ 5.0f, 7.0f });
        const auto r = gprat::gemv(A, x, b, 2, 2, gprat::Blas_substract, gprat::Blas_no_trans);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(4.0, 1e-5));
        REQUIRE_THAT(static_cast<double>(r.data()[1]), WithinAbs(5.0, 1e-5));
    }

    // gemv trans: b += A^T*x, A=[[1,0],[0,2]], x=[3,4], b=[0,0]
    // A^T = [[1,0],[0,2]] (symmetric), A^T*x = [3, 8]
    {
        auto A = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 2.0f });
        auto x = make_const_tile<float>({ 3.0f, 4.0f });
        auto b = make_tile<float>({ 0.0f, 0.0f });
        const auto r = gprat::gemv(A, x, b, 2, 2, gprat::Blas_add, gprat::Blas_trans);
        REQUIRE_THAT(static_cast<double>(r.data()[0]), WithinAbs(3.0, 1e-5));
        REQUIRE_THAT(static_cast<double>(r.data()[1]), WithinAbs(8.0, 1e-5));
    }
}

// performance counters //////////////////////////////////////////////////////////////////////////

TEST_CASE("perf_counters: register", "[unit][perf]")
{
    hpx_runtime_guard hpx_guard;
    REQUIRE_NOTHROW(gprat::register_performance_counters());
}

TEST_CASE("perf_counters: tile_data tracking", "[unit][perf]")
{
    // Reset counters to a known zero state via get(..., reset=true)
    gprat::get_tile_data_allocations(true);
    gprat::get_tile_data_deallocations(true);

    gprat::track_tile_data_allocation(64);
    gprat::track_tile_data_allocation(128);
    REQUIRE(gprat::get_tile_data_allocations(false) == 2);

    gprat::track_tile_data_deallocation(64);
    REQUIRE(gprat::get_tile_data_deallocations(false) == 1);

    // reset=true clears to zero
    REQUIRE(gprat::get_tile_data_allocations(true) == 2);
    REQUIRE(gprat::get_tile_data_allocations(false) == 0);
}

TEST_CASE("perf_counters: tile_server tracking", "[unit][perf]")
{
    gprat::get_tile_server_allocations(true);
    gprat::get_tile_server_deallocations(true);

    gprat::track_tile_server_allocation(256);
    gprat::track_tile_server_allocation(512);
    gprat::track_tile_server_deallocation(256);

    REQUIRE(gprat::get_tile_server_allocations(false) == 2);
    REQUIRE(gprat::get_tile_server_deallocations(false) == 1);

    gprat::get_tile_server_allocations(true);
    gprat::get_tile_server_deallocations(true);
}

TEST_CASE("perf_counters: transmission time", "[unit][perf]")
{
    gprat::get_tile_transmission_count(true);
    gprat::get_tile_transmission_time(true);

    gprat::record_transmission_time(1000);
    gprat::record_transmission_time(2000);
    gprat::record_transmission_time(0);  // zero elapsed: count increments, time does not

    REQUIRE(gprat::get_tile_transmission_count(false) == 3);
    REQUIRE(gprat::get_tile_transmission_time(false) == 3000);

    gprat::get_tile_transmission_count(true);
    gprat::get_tile_transmission_time(true);
}

TEST_CASE("perf_counters: force_evict", "[unit][perf]")
{
    // force_evict_memory flushes CPU cache lines — verify it runs cleanly on a
    // small buffer aligned to a typical cache line (64 bytes).
    alignas(64) std::array<double, 16> buf{};
    buf.fill(3.14);
    gprat::force_evict_memory(buf.data(), sizeof(buf));
    // verify buffer contents are unchanged after eviction
    for (const auto v : buf)
    {
        REQUIRE_THAT(v, WithinAbs(3.14, 1e-15));
    }
}

TEST_CASE("perf_counters: force_evict span", "[unit][perf]")
{
    std::vector<double> data(32, 1.5);
    gprat::force_evict_memory(std::span<const double>(data));
    for (const auto v : data)
    {
        REQUIRE_THAT(v, WithinAbs(1.5, 1e-15));
    }
}

// GPU optimize and optimize_step tests //////////////////////////////////////////////////////////

TEST_CASE("GP::optimize: GPU loss count", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const gprat::AdamParams params{ 0.01, 0.9, 0.999, 1e-8, 5 };
    const auto losses = gp_gpu.optimize(params);

    REQUIRE(losses.size() == 5);
    for (const double l : losses)
    {
        REQUIRE(std::isfinite(l));
    }
}

TEST_CASE("GP::optimize: GPU losses decrease", "[gpu][cuda][fragile]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 4);

    hpx_runtime_guard hpx_guard;
    const gprat::AdamParams params{ 0.01, 0.9, 0.999, 1e-8, 10 };
    const auto losses = gp_gpu.optimize(params);

    REQUIRE(losses.size() == 10);
    REQUIRE(losses.back() < losses.front());
}

TEST_CASE("GP::optimize_step: GPU finite loss", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    gprat::AdamParams params{ 0.01, 0.9, 0.999, 1e-8, 3 };
    const double loss0 = gp_gpu.optimize_step(params, 0);
    const double loss1 = gp_gpu.optimize_step(params, 1);
    const double loss2 = gp_gpu.optimize_step(params, 2);

    REQUIRE(std::isfinite(loss0));
    REQUIRE(std::isfinite(loss1));
    REQUIRE(std::isfinite(loss2));
}

TEST_CASE("GP::optimize: GPU matches CPU", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const gprat::AdamParams params{ 0.01, 0.9, 0.999, 1e-8, 5 };
    const auto cpu_losses = gp_cpu.optimize(params);
    const auto gpu_losses = gp_gpu.optimize(params);

    REQUIRE(cpu_losses.size() == gpu_losses.size());
    for (std::size_t i = 0; i < cpu_losses.size(); ++i)
    {
        REQUIRE_THAT(gpu_losses[i], WithinRel(cpu_losses[i], 1e-3));
    }
}

TEST_CASE("GP::cholesky: GPU values", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const auto cpu_L = gp_cpu.cholesky();
    const auto gpu_L = gp_gpu.cholesky();

    REQUIRE(gpu_L.size() == cpu_L.size());
    for (std::size_t t = 0; t < cpu_L.size(); ++t)
    {
        REQUIRE(gpu_L[t].size() == cpu_L[t].size());
        for (std::size_t e = 0; e < cpu_L[t].size(); ++e)
        {
            REQUIRE_THAT(gpu_L[t].data()[e], WithinRel(cpu_L[t].data()[e], 1e-4));
        }
    }
}

TEST_CASE("GP::predict_with_uncertainty: GPU variances positive", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8, n_test = 64;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const auto gpu_unc = gp_gpu.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);

    REQUIRE(gpu_unc.size() == 2);
    for (std::size_t i = 0; i < n_test; ++i)
    {
        REQUIRE(std::isfinite(gpu_unc[0][i]));
        REQUIRE(gpu_unc[1][i] > 0.0);  // variances must be positive
    }
}

TEST_CASE("GP::optimize: GPU no trainable params", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { false, false, false }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const gprat::AdamParams params{ 0.01, 0.9, 0.999, 1e-8, 3 };
    const auto losses = gp_gpu.optimize(params);

    REQUIRE(losses.size() == 3);
    // With no trainable parameters all losses should be identical
    REQUIRE_THAT(losses[0], WithinRel(losses[1], 1e-10));
    REQUIRE_THAT(losses[0], WithinRel(losses[2], 1e-10));
}

TEST_CASE("GP GPU: training data round-trip", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 64, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    REQUIRE(gp_gpu.get_training_input() == train_in.data);
    REQUIRE(gp_gpu.get_training_output() == train_out.data);
}

// SYCL GPU tests ////////////////////////////////////////////////////////////////////////////////
// Mirror of the CUDA GPU tests above, using the same GP API.
// Optimizer tests are omitted because the SYCL optimizer stubs are not yet implemented.

#if GPRAT_WITH_SYCL

namespace
{
int sycl_device_count() { return gprat::gpu_count(); }
}  // namespace

#define GPRAT_SKIP_IF_NO_SYCL_GPU()                                                                                    \
    do {                                                                                                               \
        if (!gprat::compiled_with_sycl())                                                                              \
            SKIP("GPRat not compiled with SYCL support");                                                              \
        if (sycl_device_count() == 0)                                                                                  \
            SKIP("No SYCL GPU detected");                                                                              \
        if (!gprat::sycl_gpu_functional())                                                                             \
            SKIP("SYCL GPU runtime not functional (oneMath ABI mismatch)");                                            \
    } while (false)

TEST_CASE("GP SYCL GPU: constructor", "[gpu][sycl]")
{
    GPRAT_SKIP_IF_NO_SYCL_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    REQUIRE_NOTHROW((gprat::GP(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1)));
}

TEST_CASE("GP SYCL::predict: GPU matches CPU", "[gpu][sycl]")
{
    GPRAT_SKIP_IF_NO_SYCL_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8, n_test = 64;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const auto cpu_pred = gp_cpu.predict(test_in.data, m_tiles, m_tile_size);
    const auto gpu_pred = gp_gpu.predict(test_in.data, m_tiles, m_tile_size);

    REQUIRE(cpu_pred.size() == n_test);
    REQUIRE(gpu_pred.size() == n_test);
    for (std::size_t i = 0; i < n_test; ++i)
    {
        REQUIRE_THAT(gpu_pred[i], WithinRel(cpu_pred[i], 1e-4));
    }
}

TEST_CASE("GP SYCL::predict_with_uncertainty: GPU matches CPU", "[gpu][sycl]")
{
    GPRAT_SKIP_IF_NO_SYCL_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8, n_test = 64;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const auto cpu_unc = gp_cpu.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);
    const auto gpu_unc = gp_gpu.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);

    REQUIRE(gpu_unc.size() == 2);
    REQUIRE(gpu_unc[0].size() == n_test);
    REQUIRE(gpu_unc[1].size() == n_test);
    for (std::size_t i = 0; i < n_test; ++i)
    {
        REQUIRE_THAT(gpu_unc[0][i], WithinRel(cpu_unc[0][i], 1e-4));
        REQUIRE_THAT(gpu_unc[1][i], WithinRel(cpu_unc[1][i], 1e-4));
    }
}

TEST_CASE("GP SYCL::predict_with_full_cov: GPU matches CPU", "[gpu][sycl]")
{
    GPRAT_SKIP_IF_NO_SYCL_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8, n_test = 64;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const auto cpu_cov = gp_cpu.predict_with_full_cov(test_in.data, m_tiles, m_tile_size);
    const auto gpu_cov = gp_gpu.predict_with_full_cov(test_in.data, m_tiles, m_tile_size);

    REQUIRE(gpu_cov.size() == 2);
    REQUIRE(gpu_cov[0].size() == n_test);
    for (std::size_t i = 0; i < n_test; ++i)
    {
        REQUIRE_THAT(gpu_cov[0][i], WithinRel(cpu_cov[0][i], 1e-4));
    }
}

TEST_CASE("GP SYCL::calculate_loss: GPU matches CPU", "[gpu][sycl]")
{
    GPRAT_SKIP_IF_NO_SYCL_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const double cpu_loss = gp_cpu.calculate_loss();
    const double gpu_loss = gp_gpu.calculate_loss();

    REQUIRE(std::isfinite(gpu_loss));
    REQUIRE_THAT(gpu_loss, WithinRel(cpu_loss, 1e-4));
}

TEST_CASE("GP SYCL::cholesky: GPU tile count", "[gpu][sycl]")
{
    GPRAT_SKIP_IF_NO_SYCL_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const auto cpu_L = gp_cpu.cholesky();
    const auto gpu_L = gp_gpu.cholesky();

    REQUIRE(gpu_L.size() == cpu_L.size());
    REQUIRE(gpu_L[0].size() == cpu_L[0].size());
    for (std::size_t t = 0; t < n_tiles; ++t)
    {
        const std::size_t diag = t * n_tiles + t;
        for (std::size_t e = 0; e < tile_size * tile_size; ++e)
        {
            REQUIRE_THAT(gpu_L[diag].data()[e], WithinRel(cpu_L[diag].data()[e], 1e-4));
        }
    }
}

TEST_CASE("GP SYCL::cholesky: GPU values", "[gpu][sycl]")
{
    GPRAT_SKIP_IF_NO_SYCL_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true });
    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const auto cpu_L = gp_cpu.cholesky();
    const auto gpu_L = gp_gpu.cholesky();

    REQUIRE(gpu_L.size() == cpu_L.size());
    for (std::size_t t = 0; t < cpu_L.size(); ++t)
    {
        REQUIRE(gpu_L[t].size() == cpu_L[t].size());
        for (std::size_t e = 0; e < cpu_L[t].size(); ++e)
        {
            REQUIRE_THAT(gpu_L[t].data()[e], WithinRel(cpu_L[t].data()[e], 1e-4));
        }
    }
}

TEST_CASE("GP SYCL::predict_with_uncertainty: GPU variances positive", "[gpu][sycl]")
{
    GPRAT_SKIP_IF_NO_SYCL_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8, n_test = 64;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    hpx_runtime_guard hpx_guard;
    const auto gpu_unc = gp_gpu.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);

    REQUIRE(gpu_unc.size() == 2);
    for (std::size_t i = 0; i < n_test; ++i)
    {
        REQUIRE(std::isfinite(gpu_unc[0][i]));
        REQUIRE(gpu_unc[1][i] > 0.0);
    }
}

TEST_CASE("GP SYCL GPU: training data round-trip", "[gpu][sycl]")
{
    GPRAT_SKIP_IF_NO_SYCL_GPU();

    const std::string root = gprat_data_root();

    constexpr std::size_t n = 64, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_gpu(
        train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { true, true, true }, 0, 1);

    REQUIRE(gp_gpu.get_training_input() == train_in.data);
    REQUIRE(gp_gpu.get_training_output() == train_out.data);
}

#endif  // GPRAT_WITH_SYCL

}  // namespace gprat::test
