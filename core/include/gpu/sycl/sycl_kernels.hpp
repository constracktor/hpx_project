#ifndef GPRAT_SYCL_KERNELS_H
#define GPRAT_SYCL_KERNELS_H

// Includes ///////////////////////////////////////////////////////////////////////////////////////////////////////////

// GPRat
#include "gprat/kernels.hpp"

#include "sycl_utils.hpp"

// Transpose kernel ///////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Kernel to transpose a matrix.
 */
class TransposeKernel
{
  private:
    double *transposed;
    double *original;
    std::size_t width;
    std::size_t height;
    sycl::local_accessor<double, 2> local;

  public:
    /**
     * @brief Construct a TransposeKernel object
     *
     * @param transposed Pointer to the transposed output matrix.
     * @param original Pointer to the original input matrix.
     * @param width Width of the original matrix.
     * @param height Height of the original matrix.
     * @param cgh SYCL command group handler for local memory
     */
    explicit TransposeKernel(
        double *transposed, double *original, std::size_t width, std::size_t height, sycl::handler &cgh) :
        transposed(transposed),
        original(original),
        width(width),
        height(height),
        local(sycl::local_accessor<double, 2>(sycl::range<2>(WORK_GROUP_SIZE, WORK_GROUP_SIZE + 1), cgh))
    { }

    /**
     * @brief The operator() implements the actual kernel functionality.
     *
     * @param nd_item The SYCL nd_item representing the current thread.
     */
    void operator()(const sycl::nd_item<2> &nd_item) const
    {
        const std::size_t local_x = nd_item.get_local_id(1);
        const std::size_t local_y = nd_item.get_local_id(0);

        const std::size_t group_x = nd_item.get_group(1);
        const std::size_t group_y = nd_item.get_group(0);

        std::size_t xIndex = group_x * WORK_GROUP_SIZE + local_x;
        std::size_t yIndex = group_y * WORK_GROUP_SIZE + local_y;

        // Load tile into local memory
        if (xIndex < width && yIndex < height)
        {
            local[local_y][local_x] = original[yIndex * width + xIndex];
        }
        else
        {
            local[local_y][local_x] = 0.0;  // padding
        }

        nd_item.barrier(sycl::access::fence_space::local_space);

        // Compute swapped indices for transpose write-back
        std::size_t x_index_out = group_y * WORK_GROUP_SIZE + local_x;
        std::size_t y_index_out = group_x * WORK_GROUP_SIZE + local_y;

        if (x_index_out < height && y_index_out < width)
        {
            transposed[y_index_out * height + x_index_out] = local[local_x][local_y];
        }
    }
};

// GenTileCovarianceKernel ////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Class for the kernel function to compute covariance
 */
class GenTileCovarianceKernel
{
  private:
    double *d_tile;
    const double *d_input;
    std::size_t n_tile_size;
    std::size_t n_regressors;
    std::size_t tile_row;
    std::size_t tile_column;
    double lengthscale_;
    double vertical_lengthscale_;
    double noise_variance_;

  public:
    /**
     * @brief Constructor of the kernel to generate a tile of the covariance matrix
     */
    explicit GenTileCovarianceKernel(
        double *d_tile,
        const double *d_input_input,
        const std::size_t n_tile_size,
        const std::size_t n_regressors,
        const std::size_t tile_row,
        const std::size_t tile_column,
        const gprat::SEKParams sek_params) :
        d_tile(d_tile),
        d_input(d_input_input),
        n_tile_size(n_tile_size),
        n_regressors(n_regressors),
        tile_row(tile_row),
        tile_column(tile_column),
        lengthscale_(sek_params.lengthscale),
        vertical_lengthscale_(sek_params.vertical_lengthscale),
        noise_variance_(sek_params.noise_variance)
    { }

    /**
     * @brief The operator() implements the actual kernel functionality.
     *
     * @param item The SYCL item representing the current thread.
     */
    void operator()(const sycl::item<2> &item) const
    {
        const std::size_t i = item.get_id(0);
        const std::size_t j = item.get_id(1);

        const std::size_t i_global = n_tile_size * tile_row + i;
        const std::size_t j_global = n_tile_size * tile_column + j;

        double distance = 0.0;
        double z_ik_minus_z_jk = 0.0;

        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            z_ik_minus_z_jk = d_input[i_global + k] - d_input[j_global + k];
            distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
        }

        double covariance = vertical_lengthscale_ * sycl::exp(-0.5 * distance / (lengthscale_ * lengthscale_));

        if (i_global == j_global)
        {
            covariance += noise_variance_;
        }

        d_tile[i * n_tile_size + j] = covariance;
    }
};

// GenTileFullPriorCovarianceKernel ///////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Class for the kernel to generate a tile of the prior covariance matrix
 *
 */
class GenTileFullPriorCovarianceKernel
{
  private:
    double *d_tile;
    const double *d_input;
    std::size_t n_tile_size;
    std::size_t n_regressors;
    std::size_t tile_row;
    std::size_t tile_column;
    double lengthscale_;
    double vertical_lengthscale_;

  public:
    /**
     * @brief Constructor of the kernel to generate a tile of the prior covariance matrix
     */
    explicit GenTileFullPriorCovarianceKernel(
        double *d_tile,
        const double *d_input_input,
        const std::size_t n_tile_size,
        const std::size_t n_regressors,
        const std::size_t tile_row,
        const std::size_t tile_column,
        const gprat::SEKParams sek_params) :
        d_tile(d_tile),
        d_input(d_input_input),
        n_tile_size(n_tile_size),
        n_regressors(n_regressors),
        tile_row(tile_row),
        tile_column(tile_column),
        lengthscale_(sek_params.lengthscale),
        vertical_lengthscale_(sek_params.vertical_lengthscale)
    { }

    /**
     * @brief The operator() implements the actual kernel functionality.
     *
     * @param item The SYCL item representing the current thread.
     */
    void operator()(const sycl::item<2> &item) const
    {
        const std::size_t i = item.get_id(0);
        const std::size_t j = item.get_id(1);

        const std::size_t i_global = n_tile_size * tile_row + i;
        const std::size_t j_global = n_tile_size * tile_column + j;

        double distance = 0.0;
        double z_ik_minus_z_jk = 0.0;

        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            z_ik_minus_z_jk = d_input[i_global + k] - d_input[j_global + k];
            distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
        }

        const double covariance = vertical_lengthscale_ * sycl::exp(-0.5 * distance / (lengthscale_ * lengthscale_));

        d_tile[i * n_tile_size + j] = covariance;
    }
};

// GenTilePriorCovarianceKernel ///////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Class for the kernel to generate the diagonal of a diagonal tile in the prior covariance matrix
 *
 */
class GenTilePriorCovarianceKernel
{
  private:
    double *d_tile;
    const double *d_input;
    std::size_t n_tile_size;
    std::size_t n_regressors;
    std::size_t tile_row;
    std::size_t tile_column;
    double lengthscale_;
    double vertical_lengthscale_;

  public:
    /**
     * @brief Constructor of the kernel to generate the diagonal of a diagonal tile in the prior covariance matrix
     */
    explicit GenTilePriorCovarianceKernel(
        double *d_tile,
        const double *d_input_input,
        const std::size_t n_tile_size,
        const std::size_t n_regressors,
        const std::size_t tile_row,
        const std::size_t tile_column,
        const gprat::SEKParams sek_params) :
        d_tile(d_tile),
        d_input(d_input_input),
        n_tile_size(n_tile_size),
        n_regressors(n_regressors),
        tile_row(tile_row),
        tile_column(tile_column),
        lengthscale_(sek_params.lengthscale),
        vertical_lengthscale_(sek_params.vertical_lengthscale)
    { }

    /**
     * @brief The operator() implements the actual kernel functionality.
     *
     * @param id The SYCL id representing the current thread.
     */
    void operator()(const sycl::id<1> &id) const
    {
        std::size_t i_global = n_tile_size * tile_row + id;
        std::size_t j_global = n_tile_size * tile_column + id;

        double distance = 0.0;
        double z_ik_minus_z_jk = 0.0;

        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            z_ik_minus_z_jk = d_input[i_global + k] - d_input[j_global + k];
            distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
        }

        double covariance = vertical_lengthscale_ * exp(-0.5 * distance / (lengthscale_ * lengthscale_));

        d_tile[id] = covariance;
    }
};

// GenTileCrossCovarianceKernel ///////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Class for the kernel to generate a tile of the cross-covariance matrix
 */
class GenTileCrossCovarianceKernel
{
  private:
    double *d_tile;
    const double *d_row_input;
    const double *d_col_input;
    std::size_t n_row_tile_size;
    std::size_t n_column_tile_size;
    std::size_t tile_row;
    std::size_t tile_column;
    std::size_t n_regressors;
    double lengthscale_;
    double vertical_lengthscale_;

  public:
    /**
     * @brief Constructor of the kernel to generate a tile of the cross-covariance matrix
     */
    explicit GenTileCrossCovarianceKernel(
        double *d_tile,
        const double *d_row_input,
        const double *d_col_input,
        const std::size_t n_row_tile_size,
        const std::size_t n_column_tile_size,
        const std::size_t tile_row,
        const std::size_t tile_column,
        const std::size_t n_regressors,
        const gprat::SEKParams sek_params) :
        d_tile(d_tile),
        d_row_input(d_row_input),
        d_col_input(d_col_input),
        n_row_tile_size(n_row_tile_size),
        n_column_tile_size(n_column_tile_size),
        tile_row(tile_row),
        tile_column(tile_column),
        n_regressors(n_regressors),
        lengthscale_(sek_params.lengthscale),
        vertical_lengthscale_(sek_params.vertical_lengthscale)
    { }

    /**
     * @brief The operator() implements the actual kernel functionality.
     *
     * @param item The SYCL item representing the current thread.
     */
    void operator()(const sycl::item<2> &item) const
    {
        const std::size_t i = item.get_id(0);
        const std::size_t j = item.get_id(1);

        const std::size_t i_global = n_row_tile_size * tile_row + i;
        const std::size_t j_global = n_column_tile_size * tile_column + j;

        double distance = 0.0;
        double z_ik_minus_z_jk = 0.0;

        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            z_ik_minus_z_jk = d_row_input[i_global + k] - d_col_input[j_global + k];
            distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
        }

        const double covariance = vertical_lengthscale_ * sycl::exp(-0.5 * distance / (lengthscale_ * lengthscale_));

        d_tile[i * n_column_tile_size + j] = covariance;
    }
};

// GenTileOutputKernel ////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Class for the kernel to generate a tile of the output data
 */
class GenTileOutputKernel
{
  private:
    double *tile;
    const double *output;
    std::size_t row;
    std::size_t n_tile_size;

  public:
    /**
     * @brief Constructor of the kernel to generate a tile of the output data
     */
    explicit GenTileOutputKernel(double *tile, const double *output, std::size_t row, std::size_t n_tile_size) :
        tile(tile),
        output(output),
        row(row),
        n_tile_size(n_tile_size)
    { }

    /**
     * @brief The operator() implements the actual kernel functionality.
     *
     * @param id The SYCL id representing the current thread.
     */
    void operator()(const sycl::id<1> &id) const
    {
        std::size_t i_global = n_tile_size * row + id;
        tile[id] = output[i_global];
    }
};

#endif  // end of GPRAT_SYCL_KERNELS_H
