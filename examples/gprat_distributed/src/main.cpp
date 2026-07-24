// All of these are necessary:
#include "gprat/cpu/adapter_cblas_fp64_actions.hpp"
#include "gprat/cpu/gp_algorithms_actions.hpp"
#include "gprat/cpu/gp_functions.hpp"
#include "gprat/cpu/gp_optimizer_actions.hpp"
#include "gprat/cpu/gp_uncertainty_actions.hpp"
#include "gprat/gprat.hpp"
#include "gprat/kernels.hpp"
#include "gprat/performance_counters.hpp"
#include "gprat/scheduler/sma.hpp"
#include "gprat/tiled_dataset.hpp"
#include "gprat/utils.hpp"

#include <filesystem>
#include <fstream>
#include <hpx/compute.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/hpx_init_params.hpp>
#include <iostream>

GPRAT_NS_BEGIN

void finish_step(const char *name, double elapsed_seconds)
{
    std::cerr << name << " done in " << elapsed_seconds << " seconds" << std::endl;
    hpx::evaluate_active_counters(true, name);
}

void run(hpx::program_options::variables_map &vm)
{
    /////////////////////
    /////// configuration
    const std::size_t START = vm["start"].as<std::size_t>();
    const std::size_t END = vm["end"].as<std::size_t>();
    const std::size_t STEP = vm["step"].as<std::size_t>();
    const std::size_t LOOP = vm["loop"].as<std::size_t>();
    const std::size_t OPT_ITER = vm["opt_iter"].as<std::size_t>();
    const std::size_t enabled = vm["enabled"].as<std::size_t>();

    const std::size_t n_test = vm["n_test"].as<std::size_t>();
    const std::size_t n_tiles = vm["tiles"].as<std::size_t>();
    const std::size_t n_reg = vm["regressors"].as<std::size_t>();

    const auto &train_path = vm["train_x_path"].as<std::string>();
    const auto &out_path = vm["train_y_path"].as<std::string>();
    const auto &test_path = vm["test_path"].as<std::string>();

    // The sweep below is multiplicative (start *= STEP); it never advances when
    // START == 0 or STEP <= 1 and would loop forever.
    if (START == 0)
    {
        throw std::runtime_error("--start must be > 0 (multiplicative training-size sweep).");
    }
    if (STEP <= 1)
    {
        throw std::runtime_error("--step must be > 1 (multiplicative training-size sweep).");
    }

    tiled_scheduler_sma scheduler;
    const auto n_localities = hpx::get_num_localities().get();

    for (std::size_t start = START; start <= END; start = start * STEP)
    {
        const auto n_train = start;
        for (std::size_t l = 0; l < LOOP; l++)
        {
            hpx::chrono::high_resolution_timer total_timer;

            // Compute tile sizes and number of predict tiles
            const auto tile_size = compute_train_tile_size(n_train, n_tiles);
            const auto result = compute_test_tiles(n_test, n_tiles, tile_size);
            /////////////////////
            ///// hyperparams
            AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPT_ITER };
            SEKParams sek_params = { 1.0, 1.0, 0.1 };
            std::vector<bool> trainable = { true, true, true };

            /////////////////////
            ////// data loading
            hpx::chrono::high_resolution_timer init_timer;
            GP_data training_input(train_path, n_train, n_reg);
            GP_data training_output(out_path, n_train, n_reg);
            GP_data test_input(test_path, n_test, n_reg);
            const auto init_time = init_timer.elapsed();
            finish_step("init", init_time);

            /////////////////////
            ///// GP

            // Start with a clean slate
            hpx::reset_active_counters();

            hpx::chrono::high_resolution_timer cholesky_timer;
            if (enabled & (1 << 0))
            {
                cpu::cholesky(scheduler, training_input.data, sek_params, n_tiles, tile_size, n_reg);
            }
            const auto cholesky_time = cholesky_timer.elapsed();
            finish_step("cholesky", cholesky_time);

            hpx::chrono::high_resolution_timer opt_timer;
            if (enabled & (1 << 1))
            {
                cpu::optimize(
                    scheduler,
                    training_input.data,
                    training_output.data,
                    n_tiles,
                    tile_size,
                    n_reg,
                    hpar,
                    sek_params,
                    trainable);
            }
            const auto opt_time = opt_timer.elapsed();
            finish_step("opt", opt_time);

            hpx::chrono::high_resolution_timer predict_timer;
            if (enabled & (1 << 2))
            {
                cpu::predict(
                    scheduler,
                    training_input.data,
                    training_output.data,
                    test_input.data,
                    sek_params,
                    n_tiles,
                    tile_size,
                    result.first,
                    result.second,
                    n_reg);
            }
            const auto predict_time = predict_timer.elapsed();
            finish_step("predict", predict_time);

            hpx::chrono::high_resolution_timer predict_with_uncertainty_timer;
            if (enabled & (1 << 3))
            {
                cpu::predict_with_uncertainty(
                    scheduler,
                    training_input.data,
                    training_output.data,
                    test_input.data,
                    sek_params,
                    n_tiles,
                    tile_size,
                    result.first,
                    result.second,
                    n_reg);
            }
            const auto predict_with_uncertainty_time = predict_with_uncertainty_timer.elapsed();
            finish_step("predict_with_uncertainty", predict_with_uncertainty_time);

            hpx::chrono::high_resolution_timer predict_with_full_cov_timer;
            if (enabled & (1 << 4))
            {
                cpu::predict_with_full_cov(
                    scheduler,
                    training_input.data,
                    training_output.data,
                    test_input.data,
                    sek_params,
                    n_tiles,
                    tile_size,
                    result.first,
                    result.second,
                    n_reg);
            }
            const auto predict_with_full_cov_time = predict_with_full_cov_timer.elapsed();
            finish_step("predict_with_full_cov", predict_with_full_cov_time);

            // Save parameters and times to a .csv file with a header
            const auto &csv_path = vm["output_csv"].as<std::string>();
            const std::string csv_header =
                "Cores,Localities,N_train,N_test,N_tiles,N_regressor,Opt_iter,Total_time,Init_time,"
                "Cholesky_time,Opt_time,Pred_Uncer_time,Pred_Full_time,Pred_time,N_loop";

            std::ofstream outfile(csv_path, std::ios::app);
            if (outfile.tellp() == 0)
            {
                // If file is empty, write the header
                outfile << csv_header << "\n";
            }
            else
            {
                // Guard against silently misaligned columns when appending to a file written
                // by an older/different version of this binary.
                std::ifstream existing(csv_path);
                std::string existing_header;
                std::getline(existing, existing_header);
                if (existing_header != csv_header)
                {
                    throw std::runtime_error("output_csv '" + csv_path
                                             + "' already exists with a different column layout. Use a different "
                                               "--output_csv path or remove the old file.");
                }
            }
            // "Cores" reports the number of HPX worker (OS) threads on this locality,
            // matching what gprat_cpp/gprat_python write in their "Cores" column.
            outfile << hpx::get_os_thread_count() << "," << n_localities << "," << n_train << "," << n_test << ","
                    << n_tiles << "," << n_reg << "," << OPT_ITER << "," << total_timer.elapsed() << "," << init_time
                    << "," << cholesky_time << "," << opt_time << "," << predict_with_uncertainty_time << ","
                    << predict_with_full_cov_time << "," << predict_time << "," << l << "\n";
            outfile.close();

            std::cerr << "====================" << std::endl;
        }
    }
    std::cerr << "DONE!" << std::endl;
}

void startup()
{
    std::cerr << "startup() called" << std::endl;

    static struct once_dummy_struct
    {
        once_dummy_struct() { register_performance_counters(); }
    } once_dummy;
}

bool check_startup(hpx::startup_function_type &startup_func, bool &pre_startup)
{
    // perform full module startup (counters will be used)
    startup_func = startup;
    pre_startup = true;
    return true;
}

GPRAT_NS_END

HPX_REGISTER_STARTUP_MODULE(GPRAT_NS::check_startup)

int hpx_main(hpx::program_options::variables_map &vm)
{
    // Debugging: dumps the full HPX runtime configuration (AGAS, logging, thread pools, etc.)
    // hpx::get_runtime().get_config().dump(0, std::cerr);
    std::cerr << "OS Threads: " << hpx::get_os_thread_count() << std::endl;
    std::cerr << "All localities: " << hpx::get_num_localities().get() << std::endl;
    std::cerr << "Root locality: " << hpx::find_root_locality() << std::endl;
    std::cerr << "This locality: " << hpx::find_here() << std::endl;
    std::cerr << "Remote localities: " << hpx::find_remote_localities().size() << std::endl;

    auto numa_domains = hpx::compute::host::numa_domains();
    std::cerr << "Local NUMA domains: " << numa_domains.size() << std::endl;
    for (const auto &domain : numa_domains)
    {
        const auto &num_pus = domain.num_pus();
        std::cerr << " Domain: " << num_pus.first << " " << num_pus.second << std::endl;
    }

    bool success = true;
    try
    {
        GPRAT_NS::run(vm);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        success = false;
    }

    // hpx::finalize() always returns 0 by design; report failure via hpx_main's own
    // return value instead, since that is what hpx::init() ultimately propagates.
    hpx::finalize();
    return success ? 0 : 1;
}

int main(int argc, char *argv[])
{
    namespace po = hpx::program_options;
    po::options_description desc("Allowed options");

    // Default to <example dir>/output.csv, matching gprat_cpp's convention of writing
    // its output next to the example sources regardless of the current working directory.
    const std::string default_output_csv = (std::filesystem::path(GPRAT_DISTRIBUTED_DIR) / "output.csv").string();

    // clang-format off
    desc.add_options()
        ("help", "produce help message")
        ("train_x_path", po::value<std::string>()->default_value("data/data_1024/training_input.txt"), "training data (x)")
        ("train_y_path", po::value<std::string>()->default_value("data/data_1024/training_output.txt"), "training data (y)")
        ("test_path", po::value<std::string>()->default_value("data/data_1024/test_input.txt"), "test data")
        ("output_csv", po::value<std::string>()->default_value(default_output_csv), "output timing reports")
        ("tiles", po::value<std::size_t>()->default_value(16), "tiles per dimension")
        ("regressors", po::value<std::size_t>()->default_value(8), "num regressors")
        ("start", po::value<std::size_t>()->default_value(128), "Starting number of training samples")
        ("end", po::value<std::size_t>()->default_value(128), "End number of training samples")
        ("step", po::value<std::size_t>()->default_value(2), "Increment of training samples")
        ("n_test", po::value<std::size_t>()->default_value(128), "Number of test samples")
        ("loop", po::value<std::size_t>()->default_value(1), "Number of iterations to be performed for each number of training samples")
        ("opt_iter", po::value<std::size_t>()->default_value(3), "Number of optimization iterations")
        ("enabled", po::value<std::size_t>()->default_value((std::numeric_limits<std::size_t>::max)()), "Bitmask of enabled steps")
    ;
    // clang-format on

    hpx::init_params init_args;
    init_args.desc_cmdline = desc;
    // If example requires to run hpx_main on all localities
    // std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};
    // init_args.cfg = cfg;
    // Run HPX main
    return hpx::init(argc, argv, init_args);
}
