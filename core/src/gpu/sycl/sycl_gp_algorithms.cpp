#include "gpu/sycl/sycl_gp_algorithms.hpp"

#include "gp_kernels.hpp"
#include "gpu/sycl/sycl_gp_optimizer.hpp"
#include "gpu/sycl/sycl_kernels.hpp"
#include "gpu/sycl/sycl_utils.hpp"
#include "target.hpp"
#include <hpx/algorithm.hpp>

namespace gprat::sycl_backend
{

// SYCL boilerplate code //////////////////////////////////////////////////////////////////////////////////////////////

double *gen_tile_covariance(const double *d_input,
                            const std::size_t tile_row,
                            const std::size_t tile_column,
                            const std::size_t n_tile_size,
                            const std::size_t n_regressors,
                            const gprat_hyper::SEKParams sek_params,
                            gprat::SYCL_DEVICE &sycl_device)
{
    try
    {
        double *d_tile;

        sycl::queue queue = sycl_device.next_queue();

        d_tile = sycl::malloc_device<double>(n_tile_size * n_tile_size, queue);

        auto event = queue.submit(
            [&](sycl::handler &cgh)
            {
                auto kernel = GenTileCovarianceKernel(
                    d_tile, d_input, n_tile_size, n_regressors, tile_row, tile_column, sek_params);
                cgh.parallel_for(sycl::range<2>(n_tile_size, n_tile_size), kernel);
            });

        event.wait();

        return d_tile;
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
        return nullptr;
    }
}

double *gen_tile_full_prior_covariance(
    const double *d_input,
    const std::size_t tile_row,
    const std::size_t tile_columns,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::SYCL_DEVICE &sycl_device)
{
    try
    {
        double *d_tile;

        sycl::queue queue = sycl_device.next_queue();

        d_tile = sycl::malloc_device<double>(n_tile_size * n_tile_size, queue);

        auto event = queue.submit(
            [&](sycl::handler &cgh)
            {
                auto kernel = GenTileFullPriorCovarianceKernel(
                    d_tile, d_input, n_tile_size, n_regressors, tile_row, tile_columns, sek_params);
                cgh.parallel_for(sycl::range<2>(n_tile_size, n_tile_size), kernel);
            });

        event.wait();
        return d_tile;
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
        return nullptr;
    }
}

double *gen_tile_prior_covariance(
    const double *d_input,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::SYCL_DEVICE &sycl_device)
{
    try
    {
        double *d_tile;

        sycl::queue queue = sycl_device.next_queue();

        d_tile = sycl::malloc_device<double>(n_tile_size, queue);

        auto event = queue.submit(
            [&](sycl::handler &cgh)
            {
                auto kernel = GenTilePriorCovarianceKernel(
                    d_tile, d_input, n_tile_size, n_regressors, tile_row, tile_column, sek_params);
                cgh.parallel_for(sycl::range<1>(n_tile_size), kernel);
            });

        event.wait();
        return d_tile;
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
        return nullptr;
    }
}

double *gen_tile_cross_covariance(
    const double *d_row_input,
    const double *d_col_input,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const std::size_t n_row_tile_size,
    const std::size_t n_column_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::SYCL_DEVICE &sycl_device)
{
    try
    {
        double *d_tile;

        sycl::queue queue = sycl_device.next_queue();

        d_tile = sycl::malloc_device<double>(n_row_tile_size * n_column_tile_size, queue);

        auto event = queue.submit(
            [&](sycl::handler &cgh)
            {
                auto kernel = GenTileCrossCovarianceKernel(
                    d_tile,
                    d_row_input,
                    d_col_input,
                    n_row_tile_size,
                    n_column_tile_size,
                    tile_row,
                    tile_column,
                    n_regressors,
                    sek_params);
                cgh.parallel_for(sycl::range<2>(n_row_tile_size, n_column_tile_size), kernel);
            });

        event.wait();
        return d_tile;
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
        return nullptr;
    }
}

hpx::shared_future<double *> gen_tile_cross_cov_T(std::size_t n_row_tile_size,
                                                  std::size_t n_column_tile_size,
                                                  const hpx::shared_future<double *> f_cross_covariance_tile,
                                                  gprat::SYCL_DEVICE &sycl_device)
{
    try
    {
        double *transposed;

        sycl::queue queue = sycl_device.next_queue();

        transposed = sycl::malloc_device<double>(n_row_tile_size * n_column_tile_size, queue);

        double *d_cross_covariance_tile = f_cross_covariance_tile.get();

        sycl::range<2> global_range(((n_row_tile_size + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE) * WORK_GROUP_SIZE,
                                    ((n_column_tile_size + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE) * WORK_GROUP_SIZE);

        sycl::range<2> local_range(WORK_GROUP_SIZE, WORK_GROUP_SIZE);

        auto event = queue.submit(
            [&](sycl::handler &cgh)
            {
                auto kernel =
                    TransposeKernel(transposed, d_cross_covariance_tile, n_column_tile_size, n_row_tile_size, cgh);
                cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kernel);
            });

        event.wait();
        return hpx::make_ready_future(transposed);
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
        return hpx::make_ready_future<double *>(nullptr);
    }
}

double *gen_tile_output(
    const std::size_t row, const std::size_t n_tile_size, const double *d_output, gprat::SYCL_DEVICE &sycl_device)
{
    try
    {
        double *d_tile;

        sycl::queue queue = sycl_device.next_queue();

        d_tile = sycl::malloc_device<double>(n_tile_size, queue);

        auto event = queue.submit(
            [&](sycl::handler &cgh)
            {
                auto kernel = GenTileOutputKernel(d_tile, d_output, row, n_tile_size);
                cgh.parallel_for(sycl::range<1>(n_tile_size), kernel);
            });

        event.wait();
        return d_tile;
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
        return nullptr;
    }
}

double *gen_tile_zeros(std::size_t n_tile_size, gprat::SYCL_DEVICE &sycl_device)
{
    try
    {
        sycl::queue queue = sycl_device.next_queue();

        double *d_tile;
        d_tile = sycl::malloc_device<double>(n_tile_size, queue);
        queue.fill(d_tile, 0.0, n_tile_size).wait();
        return d_tile;
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
        return nullptr;
    }
}

// Standard C++ code //////////////////////////////////////////////////////////////////////////////////////////////////

double compute_error_norm(std::size_t n_tiles,
                          std::size_t n_tile_size,
                          const std::vector<double> &b,
                          const std::vector<std::vector<double>> &tiles)
{
    double error = 0.0;
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        auto a = tiles[k];
        for (std::size_t i = 0; i < n_tile_size; i++)
        {
            std::size_t i_global = n_tile_size * k + i;
            // ||a - b||_2
            error += (b[i_global] - a[i]) * (b[i_global] - a[i]);
        }
    }
    return sqrt(error);
}

// HPX boilerplate code ///////////////////////////////////////////////////////////////////////////////////////////////

std::vector<hpx::shared_future<double *>> assemble_tiled_covariance_matrix(
    const double *d_training_input,
    const std::size_t n_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::SYCL_DEVICE &sycl_device)
{
    std::vector<hpx::shared_future<double *>> d_tiles(n_tiles * n_tiles);

    for (std::size_t tile_row = 0; tile_row < n_tiles; ++tile_row)
    {
        for (std::size_t tile_column = 0; tile_column < tile_row + 1; ++tile_column)
        {
            double *result = gen_tile_covariance(
                d_training_input, tile_row, tile_column, n_tile_size, n_regressors, sek_params, std::ref(sycl_device));

            d_tiles[tile_row * n_tiles + tile_column] = hpx::make_ready_future<double *>(result);
        }
    }

    return d_tiles;
}

std::vector<hpx::shared_future<double *>> assemble_alpha_tiles(
    const double *d_output, const std::size_t n_tiles, const std::size_t n_tile_size, gprat::SYCL_DEVICE &sycl_device)
{
    std::vector<hpx::shared_future<double *>> alpha_tiles(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_tiled_alpha"),
            i,
            n_tile_size,
            d_output,
            std::ref(sycl_device));
    }

    return alpha_tiles;
}

std::vector<hpx::shared_future<double *>> assemble_cross_covariance_tiles(
    const double *d_test_input,
    const double *d_training_input,
    const std::size_t m_tiles,
    const std::size_t n_tiles,
    const std::size_t m_tile_size,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::SYCL_DEVICE &sycl_device)
{
    std::vector<hpx::shared_future<double *>> cross_covariance_tiles;
    cross_covariance_tiles.resize(m_tiles * n_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            cross_covariance_tiles[i * n_tiles + j] = hpx::async(
                [=, &sycl_device]()
                {
                    return gen_tile_cross_covariance(
                        d_test_input,
                        d_training_input,
                        i,
                        j,
                        m_tile_size,
                        n_tile_size,
                        n_regressors,
                        sek_params,
                        std::ref(sycl_device));
                });
        }
    }
    return cross_covariance_tiles;
}

std::vector<hpx::shared_future<double *>>
assemble_tiles_with_zeros(std::size_t n_tile_size, std::size_t n_tiles, gprat::SYCL_DEVICE &sycl_device)
{
    std::vector<hpx::shared_future<double *>> tiles(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        tiles[i] = hpx::async(&gen_tile_zeros, n_tile_size, std::ref(sycl_device));
    }
    return tiles;
}

std::vector<hpx::shared_future<double *>> assemble_prior_K_tiles(
    const double *d_test_input,
    const std::size_t m_tiles,
    const std::size_t m_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::SYCL_DEVICE &sycl_device)
{
    std::vector<hpx::shared_future<double *>> d_prior_K_tiles;
    d_prior_K_tiles.resize(m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        d_prior_K_tiles[i] = hpx::async(
            [=, &sycl_device]()
            {
                return gen_tile_prior_covariance(
                    d_test_input, i, i, m_tile_size, n_regressors, sek_params, std::ref(sycl_device));
            });
    }
    return d_prior_K_tiles;
}

std::vector<hpx::shared_future<double *>> assemble_prior_K_tiles_full(
    const double *d_test_input,
    const std::size_t m_tiles,
    const std::size_t m_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::SYCL_DEVICE &sycl_device)
{
    std::vector<hpx::shared_future<double *>> d_prior_K_tiles(m_tiles * m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            d_prior_K_tiles[i * m_tiles + j] = hpx::async(
                &gen_tile_full_prior_covariance,
                d_test_input,
                i,
                j,
                m_tile_size,
                n_regressors,
                sek_params,
                std::ref(sycl_device));

            if (i != j)
            {
                d_prior_K_tiles[j * m_tiles + i] = hpx::dataflow(
                    &gen_tile_grad_l_trans, m_tile_size, d_prior_K_tiles[i * m_tiles + j], std::ref(sycl_device));
            }
        }
    }
    return d_prior_K_tiles;
}

std::vector<hpx::shared_future<double *>> assemble_t_cross_covariance_tiles(
    const std::vector<hpx::shared_future<double *>> &d_cross_covariance_tiles,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    gprat::SYCL_DEVICE &sycl_device)
{
    std::vector<hpx::shared_future<double *>> d_t_cross_covariance_tiles(m_tiles * n_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            d_t_cross_covariance_tiles[j * m_tiles + i] = hpx::dataflow(
                &gen_tile_cross_cov_T,
                m_tile_size,
                n_tile_size,
                d_cross_covariance_tiles[i * n_tiles + j],
                std::ref(sycl_device));
        }
    }
    return d_t_cross_covariance_tiles;
}

std::vector<hpx::shared_future<double *>>
assemble_y_tiles(const double *d_training_output,
                 const std::size_t n_tiles,
                 const std::size_t n_tile_size,
                 gprat::SYCL_DEVICE &sycl_device)
{
    std::vector<hpx::shared_future<double *>> d_y_tiles(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        d_y_tiles[i] = hpx::async(&gen_tile_output, i, n_tile_size, d_training_output, std::ref(sycl_device));
    }
    return d_y_tiles;
}

std::vector<double> copy_tiled_vector_to_host_vector(std::vector<hpx::shared_future<double *>> &d_tiles,
                                                     std::size_t n_tile_size,
                                                     std::size_t n_tiles,
                                                     gprat::SYCL_DEVICE &sycl_device)
{
    try
    {
        std::vector<double> h_vector(n_tiles * n_tile_size);
        std::vector<sycl::queue> queues(n_tiles);

        for (std::size_t i = 0; i < n_tiles; i++)
        {
            queues[i] = sycl_device.next_queue();

            queues[i].memcpy(h_vector.data() + i * n_tile_size, d_tiles[i].get(), n_tile_size * sizeof(double));
        }

        sycl_device.sync_queues(queues);
        return h_vector;
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
        return {};
    }
}

std::vector<std::vector<double>> move_lower_tiled_matrix_to_host(
    const std::vector<hpx::shared_future<double *>> &d_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gprat::SYCL_DEVICE &sycl_device)
{
    try
    {
        std::vector<std::vector<double>> h_tiles(n_tiles * n_tiles);
        std::vector<sycl::queue> queues(n_tiles * (n_tiles + 1) / 2);

        for (std::size_t i = 0; i < n_tiles; ++i)
        {
            for (std::size_t j = 0; j <= i; ++j)
            {
                queues[i] = sycl_device.next_queue();
                h_tiles[i * n_tiles + j].resize(n_tile_size * n_tile_size);

                queues[i].memcpy(h_tiles[i * n_tiles + j].data(),
                                 d_tiles[i * n_tiles + j].get(),
                                 n_tile_size * n_tile_size * sizeof(double));

                sycl::free(d_tiles[i * n_tiles + j].get(), queues[i]);
            }
        }

        sycl_device.sync_queues(queues);
        return h_tiles;
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
        return {};
    }
}

void free_lower_tiled_matrix(const std::vector<hpx::shared_future<double *>> &d_tiles,
                             const std::size_t n_tiles,
                             gprat::SYCL_DEVICE &sycl_device)
{
    try
    {
        sycl::queue queue = sycl_device.next_queue();

        for (std::size_t i = 0; i < n_tiles; ++i)
        {
            for (std::size_t j = 0; j <= i; ++j)
            {
                sycl::free(d_tiles[i * n_tiles + j].get(), queue);
            }
        }

        queue.wait();
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
    }
}

void free_full_tiled_matrix(const std::vector<hpx::shared_future<double *>> &d_tiles,
                            const std::size_t n_tiles,
                            gprat::SYCL_DEVICE &sycl_device)
{
    try
    {
        sycl::queue queue = sycl_device.next_queue();

        for (std::size_t i = 0; i < n_tiles; ++i)
        {
            for (std::size_t j = 0; j < n_tiles; ++j)
            {
                sycl::free(d_tiles[i * n_tiles + j].get(), queue);
            }
        }

        queue.wait();
    }
    catch (const sycl::exception &e)
    {
        std::cout << "SYCL exception: " << e.what() << "\n";
    }
}

}  // namespace gprat::sycl_backend
