#ifndef GPRAT_UTILS_HPP
#define GPRAT_UTILS_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include <hpx/future.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/hpx_suspend.hpp>
#include <string>
#include <vector>

GPRAT_NS_BEGIN

/**
 * @brief Compute the number of tiles for training data, given the number of
 * samples and the size of each tile.
 *
 * @param n_samples Number of samples
 * @param n_tile_size Size of each tile
 */
std::size_t compute_train_tiles(std::size_t n_samples, std::size_t n_tile_size);

/**
 * @brief Compute the number of tiles for training data, given the number of
 * samples and the size of each tile.
 *
 * @param n_samples Number of samples
 * @param n_tiles Size of each tile
 */
std::size_t compute_train_tile_size(std::size_t n_samples, std::size_t n_tiles);

/**
 * @brief Compute the number of test tiles and the size of a test tile.
 *
 * Uses n_tiles_size if n_test is divisible by n_tile_size. Otherwise uses
 * n_tiles for calculation.
 *
 * @param n_test Number of test samples
 * @param n_tiles Number of tiles
 * @param n_tile_size Size of each tile
 */
std::pair<std::size_t, std::size_t>
compute_test_tiles(std::size_t n_test, std::size_t n_tiles, std::size_t n_tile_size);

/**
 * @brief Computes a good-enough guess for the number of tiles per dimension.
 *
 * This guess is based on experiments ran on a single dual-socket 64 core machine.
 * It might not be appropriate for distributed scenarios.
 *
 * @param n Number of samples
 */
std::size_t guess_good_tile_count_per_dimension(std::size_t n);

/**
 * @brief Load data from file
 *
 * @param file_path Path to the file
 * @param n_samples Number of samples to load
 */
std::vector<double> load_data(const std::string &file_path, std::size_t n_samples, std::size_t offset);

/**
 * @brief Print a vector
 *
 * @param vec Vector to print
 * @param start Start index
 * @param end End index
 * @param separator Separator between elements
 */
void print_vector(const std::vector<double> &vec, int start, int end, const std::string &separator);

/**
 * @brief Start HPX runtime
 *
 * @param argc Number of arguments
 * @param argv Arguments as array of strings
 */
void start_hpx_runtime(int argc, char **argv);

/**
 * @brief Resume HPX runtime
 */
void resume_hpx_runtime();

/**
 * @brief Wait for all tasks to finish, and suspend the HPX runtime
 */
void suspend_hpx_runtime();

/**
 * @brief Stop HPX runtime
 */
void stop_hpx_runtime();

/**
 * @brief Returns whether the code was compiled with CUDA support.
 */
bool compiled_with_cuda();

/**
 * @brief Returns whether the code was compiled with SYCL support.
 */
bool compiled_with_sycl();

/**
 * @brief Returns whether the SYCL GPU compute backend is functional at runtime.
 *
 * Returns false if required runtime libraries (e.g. oneMath LAPACK) are
 * ABI-incompatible with the current SYCL runtime, even if compiled_with_sycl() is true.
 */
bool sycl_gpu_functional();

GPRAT_NS_END

#endif
