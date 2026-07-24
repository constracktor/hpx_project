#ifndef SYCL_GP_OPTIMIZER_H
#define SYCL_GP_OPTIMIZER_H

// GPRat
#include "gprat/hyperparameters.hpp"
#include "gprat/kernels.hpp"
#include "gprat/target.hpp"

// HPX
#include <hpx/future.hpp>

// STD library
#include <vector>

namespace gprat::sycl_backend
{

/**
 * @brief Generate a derivative tile w.r.t. lengthscale.
 *
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param grad_l_tile The gradient of the left side
 *
 * @return A quadratic tile of the derivative of l of size N x N
 */
std::vector<double> gen_tile_grad_v_trans(std::size_t N, const std::vector<double> &grad_l_tile);

/**
 * @brief Generate a derivative tile w.r.t. lengthscale.
 *
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param f_grad_l_tile The gradient of the left side
 * @param sycl_device The SYCL target for computations
 *
 * @return A quadratic tile of the derivative of l of size N x N
 */
hpx::shared_future<double *>
gen_tile_grad_l_trans(std::size_t N, const hpx::shared_future<double *> f_grad_l_tile, gprat::SYCL_DEVICE &sycl_device);

/**
 * @brief Compute negative-log likelihood on one tile.
 *
 * @param K_diag_tile The Cholesky factor L (in a diagonal tile)
 * @param alpha_tile The tiled solution of K * alpha = y
 * @param y_tile The output tile
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param sycl_device The SYCL target for computations
 *
 * @return Return l = y^T * alpha + \sum_i^N log(L_ii^2)
 */
double compute_loss(const hpx::shared_future<double *> &K_diag_tile,
                    const hpx::shared_future<double *> &alpha_tile,
                    const hpx::shared_future<double *> &y_tile,
                    std::size_t N,
                    gprat::SYCL_DEVICE &sycl_device);

/**
 * @brief Add up negative-log likelihood loss for all tiles.
 *
 * @param losses A vector contianing the loss per tile
 * @param n_tile_size The size of a tile
 * @param n_tiles The number of tiles
 *
 * @return The added up loss plus the constant factor
 */
hpx::shared_future<double>
add_losses(const std::vector<hpx::shared_future<double>> &losses, std::size_t n_tile_size, std::size_t n_tiles);

}  // namespace gprat::sycl_backend

#endif  // end of SYCL_GP_OPTIMIZER_H
