// GPRat
#include "gprat/gprat.hpp"
#include "gprat/utils.hpp"

// Boost
#include <boost/json/src.hpp>

// Standard library
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>

namespace gprat::example
{
struct Runtimes
{
    std::chrono::duration<double> init;
    std::chrono::duration<double> cholesky;
    std::chrono::duration<double> opt;
    std::chrono::duration<double> pred_uncer;
    std::chrono::duration<double> pred_full_cov;
    std::chrono::duration<double> pred;
};

struct GpratSettings
{
    std::string train_in_file;
    std::string train_out_file;
    std::string test_in_file;

    int train_size_start;
    int train_size_end;
    int train_size_step;

    int start_cores;
    int end_cores;

    int test_size;
    bool scale_test_with_train;

    int n_reg;
    int opt_iter;
    int loop;
    int n_tiles_start;
    int n_tiles_end;
    int step_tiles;

    bool cholesky;
};

template <typename T>
inline void extract(const boost::json::object &obj, T &t, std::string_view key)
{
    t = boost::json::value_to<T>(obj.at(key));
}

GpratSettings tag_invoke(boost::json::value_to_tag<GpratSettings>, const boost::json::value &jv)
{
    GpratSettings settings;
    const auto &obj = jv.as_object();
    extract(obj, settings.train_in_file, "TRAIN_IN_FILE");
    extract(obj, settings.train_out_file, "TRAIN_OUT_FILE");
    extract(obj, settings.test_in_file, "TEST_IN_FILE");
    extract(obj, settings.train_size_start, "TRAIN_SIZE_START");
    extract(obj, settings.train_size_end, "TRAIN_SIZE_END");
    extract(obj, settings.train_size_step, "STEP");
    extract(obj, settings.test_size, "TEST_SIZE");
    extract(obj, settings.scale_test_with_train, "SCALE_TEST_WITH_TRAIN");
    extract(obj, settings.n_reg, "N_REG");
    extract(obj, settings.opt_iter, "OPT_ITER");
    extract(obj, settings.loop, "LOOP");
    extract(obj, settings.start_cores, "START_CORES");
    extract(obj, settings.end_cores, "END_CORES");
    extract(obj, settings.n_tiles_start, "N_TILES_START");
    extract(obj, settings.n_tiles_end, "N_TILES_END");
    extract(obj, settings.step_tiles, "STEP_TILES");
    extract(obj, settings.cholesky, "CHOLESKY");

    return settings;
}

// GPU test settings
constexpr int device_id = 0;
constexpr int n_units = 1;

void append_to_output_file(
    std::string &target,
    int &core,
    int &n_tiles,
    int &n_train,
    int &n_test,
    int &n_reg,
    int &n_opt_iter,
    std::chrono::duration<double> &total_time,
    Runtimes &runtimes,
    int &l)
{
    const std::filesystem::path output_path = std::filesystem::path(GPRAT_CPP_CONFIG_PATH).parent_path() / "output.csv";
    std::ofstream outfile(output_path, std::ios::app);
    if (outfile.tellp() == 0)
    {
        outfile << "Target," << "Cores," << "N_tiles," << "N_train," << "N_test," << "N_regressor," << "Opt_iter,"
                << "Total_time," << "Init_time," << "Cholesky_time," << "Opt_Time," << "Predict_time,"
                << "Pred_uncer_time," << "Pred_Full_time," << "N_loop\n";
    }
    outfile << target << "," << core << "," << n_tiles << "," << n_train << "," << n_test << "," << n_reg << ","
            << n_opt_iter << "," << total_time.count() << "," << runtimes.init.count() << ","
            << runtimes.cholesky.count() << "," << runtimes.opt.count() << "," << runtimes.pred.count() << ","
            << runtimes.pred_uncer.count() << "," << runtimes.pred_full_cov.count() << "," << l << "\n";
    outfile.close();
}

void example_cpu(Runtimes &runtimes,
                 std::pair<std::size_t, std::size_t> &result,
                 gprat::GP_data &training_input,
                 gprat::GP_data &training_output,
                 gprat::GP_data &test_input,
                 const std::size_t n_tiles,
                 const std::size_t tile_size,
                 std::vector<bool> trainable,
                 GpratSettings &settings)
{
    gprat::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, static_cast<std::size_t>(settings.opt_iter) };

    auto start_init = std::chrono::high_resolution_clock::now();
    gprat::GP gp_cpu(training_input.data,
                     training_output.data,
                     n_tiles,
                     tile_size,
                     static_cast<std::size_t>(settings.n_reg),
                     { 1.0, 1.0, 0.1 },
                     trainable);
    auto end_init = std::chrono::high_resolution_clock::now();
    runtimes.init = end_init - start_init;

    auto start_cholesky = std::chrono::high_resolution_clock::now();
    if (settings.cholesky)
    {
        gp_cpu.cholesky();
    }
    auto end_cholesky = std::chrono::high_resolution_clock::now();
    runtimes.cholesky = settings.cholesky ? end_cholesky - start_cholesky : std::chrono::seconds(-1);

    auto start_opt = std::chrono::high_resolution_clock::now();
    if (!settings.cholesky)
    {
        gp_cpu.optimize(hpar);
    }
    auto end_opt = std::chrono::high_resolution_clock::now();
    runtimes.opt = settings.cholesky ? std::chrono::seconds(-1) : end_opt - start_opt;

    auto start_pred_uncer = std::chrono::high_resolution_clock::now();
    if (!settings.cholesky)
    {
        gp_cpu.predict_with_uncertainty(test_input.data, result.first, result.second);
    }
    auto end_pred_uncer = std::chrono::high_resolution_clock::now();
    runtimes.pred_uncer = settings.cholesky ? std::chrono::seconds(-1) : end_pred_uncer - start_pred_uncer;

    auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
    if (!settings.cholesky)
    {
        gp_cpu.predict_with_full_cov(test_input.data, result.first, result.second);
    }
    auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
    runtimes.pred_full_cov = settings.cholesky ? std::chrono::seconds(-1) : end_pred_full_cov - start_pred_full_cov;

    auto start_pred = std::chrono::high_resolution_clock::now();
    if (!settings.cholesky)
    {
        gp_cpu.predict(test_input.data, result.first, result.second);
    }
    auto end_pred = std::chrono::high_resolution_clock::now();
    runtimes.pred = settings.cholesky ? std::chrono::seconds(-1) : end_pred - start_pred;
}

void example_gpu(Runtimes &runtimes,
                 std::pair<std::size_t, std::size_t> &result,
                 gprat::GP_data &training_input,
                 gprat::GP_data &training_output,
                 gprat::GP_data &test_input,
                 const std::size_t n_tiles,
                 const std::size_t tile_size,
                 std::vector<bool> trainable,
                 std::size_t n_reg,
                 bool &cholesky)
{
    auto start_init = std::chrono::high_resolution_clock::now();
    gprat::GP gp_gpu(
        training_input.data,
        training_output.data,
        n_tiles,
        tile_size,
        n_reg,
        std::vector<double>{ 1.0, 1.0, 0.1 },
        trainable,
        device_id,
        n_units);
    auto end_init = std::chrono::high_resolution_clock::now();
    runtimes.init = end_init - start_init;

    auto start_cholesky = std::chrono::high_resolution_clock::now();
    if (cholesky)
    {
        gp_gpu.cholesky();
    }
    auto end_cholesky = std::chrono::high_resolution_clock::now();
    runtimes.cholesky = cholesky ? end_cholesky - start_cholesky : std::chrono::seconds(-1);

    // NOTE: optimization is not implemented for GPU
    runtimes.opt = std::chrono::seconds(-1);

    auto start_pred_uncer = std::chrono::high_resolution_clock::now();
    if (!cholesky)
    {
        gp_gpu.predict_with_uncertainty(test_input.data, result.first, result.second);
    }
    auto end_pred_uncer = std::chrono::high_resolution_clock::now();
    runtimes.pred_uncer = cholesky ? std::chrono::seconds(-1) : end_pred_uncer - start_pred_uncer;

    auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
    if (!cholesky)
    {
        gp_gpu.predict_with_full_cov(test_input.data, result.first, result.second);
    }
    auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
    runtimes.pred_full_cov = cholesky ? std::chrono::seconds(-1) : end_pred_full_cov - start_pred_full_cov;

    auto start_pred = std::chrono::high_resolution_clock::now();
    if (!cholesky)
    {
        gp_gpu.predict(test_input.data, result.first, result.second);
    }
    auto end_pred = std::chrono::high_resolution_clock::now();
    runtimes.pred = cholesky ? std::chrono::seconds(-1) : end_pred - start_pred;
}

}  // namespace gprat::example

int main(int argc, char *argv[])
{
    gprat::example::GpratSettings settings;

    bool use_gpu = false;

    std::ifstream ifs(GPRAT_CPP_CONFIG_PATH);
    if (!ifs.fail())
    {
        using iterator_type = std::istreambuf_iterator<char>;
        const std::string content(iterator_type{ ifs }, iterator_type{});
        settings = boost::json::value_to<gprat::example::GpratSettings>(boost::json::parse(content));

        const std::filesystem::path config_dir = std::filesystem::path(GPRAT_CPP_CONFIG_PATH).parent_path();
        auto resolve = [&](std::string &p)
        {
            if (!std::filesystem::path(p).is_absolute())
            {
                p = (config_dir / p).lexically_normal().string();
            }
        };
        resolve(settings.train_in_file);
        resolve(settings.train_out_file);
        resolve(settings.test_in_file);
    }
    else
    {
        std::cerr << "Could not read config file. Please make sure config.json is present and valid.\n";
        return 1;
    }

    if (argc > 1 && std::strcmp(argv[1], "--use-gpu") == 0)
    {
        if (!gprat::compiled_with_cuda() && !gprat::compiled_with_sycl())
        {
            std::cerr << "Error: GPU support is not available. Please compile with CUDA or SYCL support.\n";
            return 1;
        }
        else if (gprat::gpu_count() == 0)
        {
            std::cerr << "GPU support requested but GPRat found no GPUs.\n";
            return 1;
        }
        else
        {
            use_gpu = true;
            if (gprat::compiled_with_cuda())
            {
                std::cout << "Using CUDA GPU for computations.\n";
            }
            else if (gprat::compiled_with_sycl())
            {
                std::cout << "Using SYCL GPU for computations.\n";
            }
        }
    }
    else
    {
        std::cout << "Using CPU for computations.\n";
    }

    std::string target = use_gpu ? gprat::compiled_with_cuda() ? "cuda" : "sycl" : "cpu";

    // Loop over cores
    for (int core = settings.start_cores; core <= settings.end_cores; core *= 2)
    {
        std::vector<std::string> args(argv, argv + argc);
        args.erase(args.begin() + argc - 1);
        args.push_back("--hpx:threads=" + std::to_string(core));

        std::vector<char *> cstr_args;
        for (auto &arg : args)
        {
            cstr_args.push_back(const_cast<char *>(arg.c_str()));
        }

        int new_argc = static_cast<int>(cstr_args.size());
        char **new_argv = cstr_args.data();

        gprat::start_hpx_runtime(new_argc, new_argv);

        // Loop over tiles
        for (int n_tiles = settings.n_tiles_start; n_tiles <= settings.n_tiles_end; n_tiles *= settings.step_tiles)
        {
            int training_baseline = settings.train_size_start > n_tiles ? settings.train_size_start : n_tiles;

            // Loop over training sizes
            for (int train_size = training_baseline; train_size <= settings.train_size_end;
                 train_size *= settings.train_size_step)
            {
                int n_test = settings.scale_test_with_train ? train_size : settings.test_size;

                // Loop over repetitions
                for (int l = 0; l < settings.loop; l++)
                {
                    auto n_tiles_st = static_cast<std::size_t>(n_tiles);
                    auto train_size_st = static_cast<std::size_t>(train_size);
                    auto n_test_st = static_cast<std::size_t>(n_test);
                    auto n_reg_st = static_cast<std::size_t>(settings.n_reg);
                    std::size_t tile_size = gprat::compute_train_tile_size(train_size_st, n_tiles_st);
                    auto result = gprat::compute_test_tiles(n_test_st, n_tiles_st, tile_size);

                    gprat::GP_data training_input(settings.train_in_file, train_size_st, n_reg_st);
                    gprat::GP_data training_output(settings.train_out_file, train_size_st, n_reg_st);
                    gprat::GP_data test_input(settings.test_in_file, n_test_st, n_reg_st);

                    gprat::example::Runtimes runtimes;
                    std::vector<bool> trainable = { true, true, true };

                    auto start_total = std::chrono::high_resolution_clock::now();

                    if (use_gpu)
                    {
                        gprat::example::example_gpu(
                            runtimes,
                            result,
                            training_input,
                            training_output,
                            test_input,
                            n_tiles_st,
                            tile_size,
                            trainable,
                            n_reg_st,
                            settings.cholesky);
                    }
                    else
                    {
                        gprat::example::example_cpu(
                            runtimes,
                            result,
                            training_input,
                            training_output,
                            test_input,
                            n_tiles_st,
                            tile_size,
                            trainable,
                            settings);
                    }

                    auto end_total = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> total_time = end_total - start_total;

                    gprat::example::append_to_output_file(
                        target,
                        core,
                        n_tiles,
                        train_size,
                        n_test,
                        settings.n_reg,
                        settings.opt_iter,
                        total_time,
                        runtimes,
                        l);
                }
            }
        }

        gprat::stop_hpx_runtime();
    }

    return 0;
}
