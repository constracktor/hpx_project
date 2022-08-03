// Disable ublas debug mode
#ifndef NDEBUG
  #define BOOST_UBLAS_NDEBUG
#endif

#define CALC_TYPE double
#define TYPE "%lf"
#include <math.h>
#include <cassert>
#include <iostream>

#include <hpx/local/chrono.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/iostream.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/triangular.hpp>

#include <boost/numeric/ublas/operation.hpp>

namespace ublas = boost::numeric::ublas;
////////////////////////////////////////////////////////////////////////////////
// GP functions to assemble K
std::vector<CALC_TYPE> compute_regressor_vector(std::size_t row, std::size_t n_regressors, std::vector<CALC_TYPE> input)
{
  std::vector<CALC_TYPE> z_row;
  z_row.resize(n_regressors);
  for (std::size_t i = 0; i < n_regressors; i++)
  {
   int index = row - n_regressors + 1 + i;
   if (index < 0)
   {
      z_row[i] = 0.0;
   }
   else
   {
     z_row[i] = input[index];
   }
  }
  return z_row;
}

CALC_TYPE compute_covariance_function(std::size_t n_regressors, CALC_TYPE* hyperparameters, std::vector<CALC_TYPE> z_i, std::vector<CALC_TYPE> z_j)
{
  // Compute the Squared Exponential Covariance Function
  // C(z_i,z_j) = vertical_lengthscale * exp(-0.5*lengthscale*(z_i-z_j)^2)
  CALC_TYPE distance = 0.0;
  for (std::size_t i = 0; i < n_regressors; i++)
  {
    distance += pow(z_i[i] - z_j[i],2);
  }
  return hyperparameters[1] * exp(-0.5 * hyperparameters[0] * distance);
}

std::vector<CALC_TYPE> gen_tile_covariance(std::size_t row, std::size_t col, std::size_t N, std::size_t n_tiles, std::size_t n_regressors, CALC_TYPE *hyperparameters, std::vector<CALC_TYPE> input)
{
   std::size_t i_global,j_global;
   CALC_TYPE covariance_function;
   std::vector<CALC_TYPE> z_i, z_j;
   // Initialize tile
   std::vector<CALC_TYPE> tile;
   tile.resize(N * N);
   for(std::size_t i = 0; i < N; i++)
   {
      i_global = N * row + i;
      z_i = compute_regressor_vector(i_global, n_regressors, input);
      for(std::size_t j = 0; j < N; j++)
      {
         j_global = N * col + j;
         z_j= compute_regressor_vector(j_global, n_regressors, input);
         // compute covariance function
         covariance_function = compute_covariance_function(n_regressors, hyperparameters, z_i, z_j);
         if (i_global==j_global)
         {
           covariance_function += hyperparameters[2];
         }
         tile[i * N + j] = covariance_function;
      }
   }
   return tile;
}

std::vector<CALC_TYPE> gen_cross_covariance(std::size_t N_row, std::size_t N_col, std::size_t n_regressors, CALC_TYPE *hyperparameters, std::vector<CALC_TYPE> z_i_input, std::vector<CALC_TYPE> z_j_input)
{
   std::size_t i_global, j_global;
   CALC_TYPE covariance_function;
   std::vector<CALC_TYPE> z_i, z_j;
   // Initialize tile
   std::vector<CALC_TYPE> tile;
   tile.resize(N_row * N_col);
   for(std::size_t i = 0; i < N_row; i++)
   {
      z_i = compute_regressor_vector(i, n_regressors, z_i_input);
      for(std::size_t j = 0; j < N_col; j++)
      {
         z_j= compute_regressor_vector(j, n_regressors, z_j_input);
         // compute covariance function
         covariance_function = compute_covariance_function(n_regressors, hyperparameters, z_i, z_j);
         tile[i * N_col + j] = covariance_function;
      }
   }
   return tile;
}

std::vector<CALC_TYPE> gen_tile_output(std::size_t row, std::size_t N, std::vector<CALC_TYPE> output)
{
   std::size_t i_global;
   // Initialize tile
   std::vector<CALC_TYPE> tile;
   tile.resize(N);
   for(std::size_t i = 0; i < N; i++)
   {
      i_global = N * row + i;
      tile[i] = output[i_global];
   }
   return tile;
}
////////////////////////////////////////////////////////////////////////////////
// BLAS operations for tiled cholkesy
// Cholesky decomposition of A -> return factorized matrix L
std::vector<CALC_TYPE> potrf(std::vector<CALC_TYPE> A,
                             std::size_t N)
{
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(N, N);
  // POTRF (compute Cholesky)
  for (size_t k=0 ; k < N; k++)
  {
    // compute squared diagonal entry
    CALC_TYPE qL_kk = A_blas(k,k) - ublas::inner_prod( ublas::project( ublas::row(L_blas, k), ublas::range(0, k) ), ublas::project( ublas::row(L_blas, k), ublas::range(0, k) ) );
    // check if positive
    if (qL_kk <= 0)
    {
      hpx::cout << qL_kk << '\n' << std::flush;
    }
    else
    {
      // set diagonal entry
      CALC_TYPE L_kk = std::sqrt( qL_kk );
      L_blas(k,k) = L_kk;
      // compute corresponding column
      ublas::matrix_column<ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> >> cLk(L_blas, k);
      ublas::project( cLk, ublas::range(k+1, N) )
        = ( ublas::project( ublas::column(A_blas, k), ublas::range(k+1, N) )
            - ublas::prod( ublas::project(L_blas, ublas::range(k+1, N), ublas::range(0, k)),
                    ublas::project(ublas::row(L_blas, k), ublas::range(0, k) ) ) ) / L_kk;
    }
  }
  // reformat to std::vector
  A = L_blas.data();
  return A;
}

// solve L * X = A^T where L triangular
std::vector<CALC_TYPE> trsm(std::vector<CALC_TYPE> L,
                            std::vector<CALC_TYPE> A,
                            std::size_t N)
{
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(N, N);
  L_blas.data() = L;
  ublas::matrix< CALC_TYPE, ublas::column_major, std::vector<CALC_TYPE> > A_blas(N, N);//use column_major because A^T
  A_blas.data() = A;
  // TRSM
  ublas::inplace_solve(L_blas, A_blas, ublas::lower_tag());
  // reformat to std::vector
  A = A_blas.data();
  return A;
}

//  A = A - B * B^T
std::vector<CALC_TYPE> syrk(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> B,
                            std::size_t N)
{
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > B_blas(N, N);
  B_blas.data() = B;
  //SYRK
  A_blas = A_blas - ublas::prod(B_blas,ublas::trans(B_blas));
  // reformat to std::vector
  A = A_blas.data();
  return A;
}

//C = C - A * B^T
std::vector<CALC_TYPE> gemm(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> B,
                            std::vector<CALC_TYPE> C,
                            std::size_t N)
{
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > B_blas(N, N);
  B_blas.data() = B;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > C_blas(N, N);
  C_blas.data() = C;
  // GEMM
  C_blas = C_blas - ublas::prod(A_blas, ublas::trans(B_blas));
  // reformat to std::vector
  C = C_blas.data();
  return C;
}

// BLAS operations for tiled triangular solve
// solve L * x = a where L lower triangular
std::vector<CALC_TYPE> trsm_l(std::vector<CALC_TYPE> L,
                              std::vector<CALC_TYPE> a,
                              std::size_t N)
{
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(N, N);
  L_blas.data() = L;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > a_blas(N);
  a_blas.data() = a;
  // TRSM
  ublas::inplace_solve(L_blas, a_blas, ublas::lower_tag());
  // reformat to std::vector
  a = a_blas.data();
  return a;
}

// solve L^T * x = a where L lower triangular
std::vector<CALC_TYPE> trsm_u(std::vector<CALC_TYPE> L,
                              std::vector<CALC_TYPE> a,
                              std::size_t N)
{
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(N, N);
  L_blas.data() = L;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > a_blas(N);
  a_blas.data() = a;
  // TRSM
  ublas::inplace_solve(ublas::trans(L_blas), a_blas, ublas::upper_tag());
  // reformat to std::vector
  a = a_blas.data();
  return a;
}

// b = b - A * a
std::vector<CALC_TYPE> gemv_l(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> a,
                            std::vector<CALC_TYPE> b,
                            std::size_t N)
{
  // convert to boost matrix and vectors
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > a_blas(N);
  a_blas.data() = a;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > b_blas(N);
  b_blas.data() = b;
  // GEMM
  b_blas = b_blas - ublas::prod(A_blas, a_blas);
  // reformat to std::vector
  b = b_blas.data();
  return b;
}

// b = b - A^T * a
std::vector<CALC_TYPE> gemv_u(std::vector<CALC_TYPE> A,
                            std::vector<CALC_TYPE> a,
                            std::vector<CALC_TYPE> b,
                            std::size_t N)
{
  // convert to boost matrix and vectors
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > A_blas(N, N);
  A_blas.data() = A;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > a_blas(N);
  a_blas.data() = a;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > b_blas(N);
  b_blas.data() = b;
  // GEMM
  b_blas = b_blas - ublas::prod(ublas::trans(A_blas), a_blas);
  // reformat to std::vector
  b = b_blas.data();
  return b;
}
////////////////////////////////////////////////////////////////////////////////
// Tiled Cholesky Algorithms
void right_looking_cholesky_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles, std::size_t N, std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::unwrapping(potrf), ft_tiles[k * n_tiles + k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::unwrapping(trsm), ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N);
    }
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // SYRK
      ft_tiles[m * n_tiles + m] = hpx::dataflow(hpx::unwrapping(syrk), ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], N);
      for (std::size_t n = k + 1; n < m; n++)
      {
        // GEMM
        ft_tiles[m * n_tiles + n] = hpx::dataflow(hpx::unwrapping(gemm), ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], N);
      }
    }
  }
}

void left_looking_cholesky_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles, std::size_t N, std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    for (std::size_t n = 0; n < k; n++)
    {
      // SYRK
      ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::unwrapping(syrk), ft_tiles[k * n_tiles + k], ft_tiles[k * n_tiles + n], N);
      for (std::size_t m = k + 1; m < n_tiles; m++)
      {
        // GEMM
        ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::unwrapping(gemm), ft_tiles[m * n_tiles + n], ft_tiles[k * n_tiles + n], ft_tiles[m * n_tiles + k], N);
      }
    }
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::unwrapping(potrf), ft_tiles[k * n_tiles + k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::unwrapping(trsm), ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N);
    }
  }
}

void top_looking_cholesky_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles, std::size_t N, std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    for (std::size_t n = 0; n < k; n++)
    {
      for (std::size_t m = 0; m < n; m++)
      {
        // GEMM
        ft_tiles[k * n_tiles + n] = hpx::dataflow(hpx::unwrapping(gemm), ft_tiles[k * n_tiles + m], ft_tiles[n * n_tiles + m], ft_tiles[k * n_tiles + n], N);
      }
      // TRSM
      ft_tiles[k * n_tiles + n] = hpx::dataflow(hpx::unwrapping(trsm), ft_tiles[n * n_tiles + n], ft_tiles[k * n_tiles + n], N);
    }
    for (std::size_t n = 0; n < k; n++)
    {
      // SYRK
      ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::unwrapping(syrk), ft_tiles[k * n_tiles + k], ft_tiles[k * n_tiles + n], N);
    }
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::unwrapping(potrf), ft_tiles[k * n_tiles + k], N);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Tiled Triangular Solve Algorithms
void forward_solve_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles, std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_rhs, std::size_t N, std::size_t n_tiles)
{
  for (std::size_t k = 0; k < n_tiles; k++)
  {
    // TRSM
    ft_rhs[k] = hpx::dataflow(hpx::unwrapping(trsm_l), ft_tiles[k * n_tiles + k], ft_rhs[k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // GEMV
      ft_rhs[m] = hpx::dataflow(hpx::unwrapping(gemv_l), ft_tiles[m * n_tiles + k], ft_rhs[k], ft_rhs[m], N);
    }
  }
}

void backward_solve_tiled(std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_tiles, std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> &ft_rhs, std::size_t N, std::size_t n_tiles)
{
  for (int k = n_tiles - 1; k >= 0; k--)
  {
    // TRSM
    ft_rhs[k] = hpx::dataflow(hpx::unwrapping(trsm_u), ft_tiles[k * n_tiles + k], ft_rhs[k], N);
    for (int m = k - 1; m >= 0; m--) // might cause problems -> int
    {
      // GEMV
      ft_rhs[m] = hpx::dataflow(hpx::unwrapping(gemv_u), ft_tiles[k * n_tiles + m], ft_rhs[k], ft_rhs[m], N);
      //ft_rhs[m] = hpx::dataflow(hpx::unwrapping(gemv), ft_tiles[m * n_tiles + k], ft_rhs[k], ft_rhs[m], N);
    }
  }
}
////////////////////////////////////////////////////////////////////////////////
// Main functions
int hpx_main(hpx::program_options::variables_map& vm)
{
  std::string cholesky = vm["cholesky"].as<std::string>();
  // GP parameters
  std::size_t n_train = vm["n_train"].as<std::size_t>();  //max 100*1000
  std::size_t n_test = vm["n_test"].as<std::size_t>();     //max 5*1000
  std::size_t n_regressors = vm["n_regressors"].as<std::size_t>();
  CALC_TYPE    hyperparameters[3];
  // initalize hyperparameters to empirical moments of the data
  hyperparameters[0] = 1.0;   // lengthscale = variance of training_output
  hyperparameters[1] = 1.0;   // vertical_lengthscale = standard deviation of training_input
  hyperparameters[2] = 0.1; // noise_variance = small value
  // tiled parameters
  std::size_t n_tiles = vm["n_tiles"].as<std::size_t>();
  std::size_t tile_size = n_train / n_tiles;
  // HPX structures
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> K_tiles;
  std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> output_tiles;
  hpx::shared_future<std::vector<CALC_TYPE>> cross_covariance;
  // data holders for assembly
  std::vector<CALC_TYPE>   training_input;
  std::vector<CALC_TYPE>   training_output;
  std::vector<CALC_TYPE>   test_input;
  std::vector<CALC_TYPE>   test_output;
  // data files
  FILE    *training_input_file;
  FILE    *training_output_file;
  FILE    *test_input_file;
  FILE    *test_output_file;
  ////////////////////////////////////////////////////////////////////////////
  // Load data
  training_input.resize(n_train);
  training_output.resize(n_train);
  test_input.resize(n_test);
  test_output.resize(n_test);
  training_input_file = fopen("../src/data/training/training_input.txt", "r");
  training_output_file = fopen("../src/data/training/training_output.txt", "r");
  test_input_file = fopen("../src/data/test/test_input_3.txt", "r");
  test_output_file = fopen("../src/data/test/test_output_3.txt", "r");
  if (training_input_file == NULL || training_output_file == NULL || test_input_file == NULL || test_output_file == NULL)
  {
    printf("Files not found!\n");
    return 1;
  }
  // load training data
  for (int i = 0; i < n_train; i++)
  {
    fscanf(training_input_file,TYPE,&training_input[i]);
    fscanf(training_output_file,TYPE,&training_output[i]);
  }
  // load test data
  for (int i = 0; i < n_test; i++)
  {
    fscanf(test_input_file,TYPE,&test_input[i]);
    fscanf(test_output_file,TYPE,&test_output[i]);
  }
  // close file streams
  fclose(training_input_file);
  fclose(training_output_file);
  fclose(test_input_file);
  fclose(test_output_file);
  // Start total timer
  hpx::chrono::high_resolution_timer t_start_total;
  //////////////////////////////////////////////////////////////////////////////
  // ASSEMBLE
  // Start timer
  hpx::chrono::high_resolution_timer t_start_assemble;
  // Assemble covariance matrix vector
  K_tiles.resize(n_tiles * n_tiles);
  // Assemble K
  for (std::size_t i = 0; i < n_tiles; i++)
  {
     for (std::size_t j = 0; j < n_tiles; ++j)
     {
        K_tiles[i * n_tiles + j] = hpx::dataflow(&gen_tile_covariance, i, j, tile_size, n_tiles, n_regressors, hyperparameters, training_input);
     }
  }
  //Assemble cross-covariacne (not tiled)
  cross_covariance = hpx::dataflow(&gen_cross_covariance, n_train, n_test, n_regressors, hyperparameters, training_input,test_input);
  //Assemble output data
  output_tiles.resize(n_tiles);
  for (std::size_t i = 0; i < n_tiles; i++)
  {
    output_tiles[i] = hpx::dataflow(&gen_tile_output, i, tile_size, training_output);
  }
  // Stop timer - wrong since not blocking
  CALC_TYPE t_elapsed_assemble = t_start_assemble.elapsed();
  //////////////////////////////////////////////////////////////////////////////
  // CHOLESKY
  // Start timer
  hpx::chrono::high_resolution_timer t_start_cholesky;
  // Compute Cholesky decomposition
  if (cholesky.compare("left") == 0)
  {
    left_looking_cholesky_tiled(K_tiles,tile_size, n_tiles);
  }
  else if (cholesky.compare("right") == 0)
  {
    right_looking_cholesky_tiled(K_tiles,tile_size, n_tiles);
  }
  else // out is set to "top" per default
  {
    top_looking_cholesky_tiled(K_tiles,tile_size, n_tiles);
  }
  /*
  // Assemble to big matrix
  std::vector<CALC_TYPE> L;
  L.resize(n_train * n_train);

  for (std::size_t k = 0; k < n_tiles; k++)
  {
     for (std::size_t l = 0; l < n_tiles; ++l)
     {
        auto tile = K_tiles[k * n_tiles + l].get();
        for(std::size_t i = 0; i < tile_size; i++)
        {
           std::size_t i_global = tile_size * k + i;
           for(std::size_t j = 0; j < tile_size; j++)
           {
              std::size_t j_global = tile_size * l + j;
              L[i_global * tile_size *n_tiles + j_global] = tile[i * tile_size + j];
           }
        }
     }
  }
  */
  forward_solve_tiled(K_tiles,output_tiles,tile_size,n_tiles);
  backward_solve_tiled(K_tiles,output_tiles,tile_size,n_tiles);
  std::vector<CALC_TYPE> alpha;
  alpha.resize(n_train);

  for (std::size_t k = 0; k < n_tiles; k++)
  {
    auto tile = output_tiles[k].get();
    for(std::size_t i = 0; i < tile_size; i++)
    {
      std::size_t i_global = tile_size * k + i;
      alpha[i_global] = tile[i];
    }
  }
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > alpha_blas(n_train);
  alpha_blas.data() = alpha;

  // Stop timer
  CALC_TYPE t_elapsed_cholesky = t_start_cholesky.elapsed();
  //////////////////////////////////////////////////////////////////////////////
  // TRIANGULAR SOLVE
  // Currently quick and dirty triangular solve
  // Start timer
  hpx::chrono::high_resolution_timer t_start_triangular;
/*
  // convert to boost matrices
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > L_blas(n_train, n_train);
  L_blas.data() = L;
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > cross_covariance_blas(n_train, n_test);
  cross_covariance_blas.data() = cross_covariance.get();
  // convert to boost vectors
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > alpha(n_train);
  alpha.data() = training_output;
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > y_test(n_test);
  y_test.data() = test_output;
  // solve triangular systems
  ublas::inplace_solve(L_blas, alpha, ublas::lower_tag() );
  ublas::inplace_solve(ublas::trans(L_blas), alpha, ublas::upper_tag());
*/
  ublas::matrix< CALC_TYPE, ublas::row_major, std::vector<CALC_TYPE> > cross_covariance_blas(n_train, n_test);
  cross_covariance_blas.data() = cross_covariance.get();
  ublas::vector< CALC_TYPE, std::vector<CALC_TYPE> > y_test(n_test);
  y_test.data() = test_output;
  // Stop timer
  CALC_TYPE t_elapsed_triangular = t_start_triangular.elapsed();
  //////////////////////////////////////////////////////////////////////////////
  // PREDICT
  // Start timer
  hpx::chrono::high_resolution_timer t_start_predict;
  // make predictions
  alpha_blas = ublas::prod(ublas::trans(cross_covariance_blas), alpha_blas);
  // compute error
  CALC_TYPE error = ublas::norm_2(alpha_blas - y_test);
  // Stop timer
  CALC_TYPE t_elapsed_predict = t_start_predict.elapsed();
  //////////////////////////////////////////////////////////////////////////////
  // Stop global timer
  CALC_TYPE t_elapsed_total = t_start_total.elapsed();
  // print output information
  printf("%ld;%lf;%lf;%lf;%lf;%lf;%lf;%ld;%ld;%ld;%s;\n", n_tiles, t_elapsed_total, t_elapsed_assemble, t_elapsed_cholesky, t_elapsed_triangular, t_elapsed_predict, error / n_test, n_train, n_test, n_regressors, cholesky.c_str());
  return hpx::local::finalize();    // Handles HPX shutdown
}

int main(int argc, char* argv[])
{
    // hpx
    hpx::program_options::options_description desc_commandline;
    hpx::local::init_params init_args;
    ////////////////////////////////////////////////////////////////////////////
    // Setup input arguments
    desc_commandline.add_options()
        ("n_train", hpx::program_options::value<std::size_t>()->default_value(1 * 1000),
         "Number of training samples (max 100 000)")
        ("n_test", hpx::program_options::value<std::size_t>()->default_value(1 * 1000),
         "Number of test samples (max 5 000)")
        ("n_regressors", hpx::program_options::value<std::size_t>()->default_value(100),
        "Number of delayed input regressors")
        ("n_tiles", hpx::program_options::value<std::size_t>()->default_value(10),
         "Number of tiles per dimension -> n_tiles * n_tiles total")
        ("cholesky", hpx::program_options::value<std::string>()->default_value("top"),
         "Choose between left- right- or top-looking tiled Cholesky decomposition")
    ;
    hpx::program_options::variables_map vm;
    hpx::program_options::store(hpx::program_options::parse_command_line(argc, argv, desc_commandline), vm);
    ////////////////////////////////////////////////////////////////////////////
    // Run HPX
    init_args.desc_cmdline = desc_commandline;
    return hpx::local::init(hpx_main, argc, argv, init_args);
}
