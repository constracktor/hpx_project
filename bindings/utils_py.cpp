#include "gprat/target.hpp"
#include "gprat/utils.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/**
 * @brief Start HPX runtime on `n_cores` many cores with `args` as arguments
 *
 * The HPX runtime keeps running after executing this function.
 *
 * @param args List of arguments for HPX runtime
 * @param n_cores Number of cores that hpx may use for its threads
 */
void start_hpx_wrapper(std::vector<std::string> args, std::size_t n_cores)
{
    // If args is empty, set the first argument to "gprat"
    if (args.empty())
    {
        args.push_back("gprat");
    }

    // Add the --hpx:threads argument to the args vector
    args.push_back("--hpx:threads=" + std::to_string(n_cores));

    // Convert std::vector<std::string> to char* array
    std::vector<char *> argv;
    for (auto &arg : args)
    {
        argv.push_back(&arg[0]);
    }
    argv.push_back(nullptr);
    int argc = static_cast<int>(args.size());
    gprat::start_hpx_runtime(argc, argv.data());
}

/**
 * @brief Add utility functions `compute_train_tiles`,
 * `compute_train_tile_size`, `compute_test_tiles`, `print`, `start_hpx`,
 * `resume_hpx`, `suspend_hpx`, `stop_hpx` to the module
 */
void init_utils(py::module &m)
{
    m.def("compute_train_tiles",
          &gprat::compute_train_tiles,
          py::arg("n_samples"),
          py::arg("n_tile_size"),
          R"pbdoc(
          Compute the number of tiles for training data.

          Parameters:
              n_samples (int): The number of samples.
              n_tile_size (int): The size of each tile.

          Returns:
              int: Number of tiles per dimension.
          )pbdoc");

    m.def("compute_train_tile_size",
          &gprat::compute_train_tile_size,
          py::arg("n_samples"),
          py::arg("n_tiles"),
          R"pbdoc(
          Compute the tile size for training data.

          Parameters:
              n_samples (int): Number of samples.
              n_tiles (int): Number of tiles per dimension.

          Returns:
              int: Tile size
          )pbdoc");

    m.def("compute_test_tiles",
          &gprat::compute_test_tiles,
          py::arg("m_samples"),
          py::arg("n_tiles"),
          py::arg("n_tile_size"),
          R"pbdoc(
          Compute the number of tiles for test data and the respective size of test tiles.

          Parameters:
              n_test (int): The number of test samples.
              n_tiles (int): The number of tiles.
              n_tile_size (int): The size of each tile.

          Returns:
              tuple: A tuple containing the number of test tiles and the adjusted tile size.
          )pbdoc");

    m.def("print_vector",
          &gprat::print_vector,
          py::arg("vec"),
          py::arg("start") = 0,
          py::arg("end") = -1,
          py::arg("separator") = " ",
          "Print elements of a vector with optional start, end, and separator parameters");

    m.def("start_hpx", &start_hpx_wrapper, py::arg("args"), py::arg("n_cores"));  // Using the wrapper function
    m.def("resume_hpx", &gprat::resume_hpx_runtime);
    m.def("suspend_hpx", &gprat::suspend_hpx_runtime);
    m.def("stop_hpx", &gprat::stop_hpx_runtime);

    m.def("compiled_with_cuda", &gprat::compiled_with_cuda, "Check if the code was compiled with CUDA support");
    m.def("compiled_with_sycl", &gprat::compiled_with_sycl, "Check if the code was compiled with SYCL support");

    m.def("print_available_gpus", &gprat::print_available_gpus, "Print available GPUs with their properties");
    m.def("gpu_count", &gprat::gpu_count, "Return the number of available GPUs");
}
