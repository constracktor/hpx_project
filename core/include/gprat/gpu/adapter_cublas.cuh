#ifndef GPRAT_GPU_ADAPTER_CUBLAS_HPP
#define GPRAT_GPU_ADAPTER_CUBLAS_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include "gprat/target.hpp"

#include <hpx/future.hpp>
#include <hpx/modules/async_cuda.hpp>

#include <cusolverDn.h>

GPRAT_NS_BEGIN

// Constants, compatible with cuBLAS

/**
 * @brief BLAS operation types
 *
 * @see cublasOperation_t
 */
typedef enum BLAS_TRANSPOSE {
    Blas_no_trans = 0,  // CUBLAS_OP_N
    Blas_trans = 1      // CUBLAS_OP_T
} BLAS_TRANSPOSE;

/**
 * @brief BLAS side types
 *
 * @see cublasSideMode_t
 */
typedef enum BLAS_SIDE {
    Blas_left = 0,  // CUBLAS_SIDE_LEFT
    Blas_right = 1  // CUBLAS_SIDE_RIGHT
} BLAS_SIDE;

/**
 * @brief BLAS types for alpha scalar
 */
typedef enum BLAS_ALPHA { Blas_add = 1, Blas_substract = -1 } BLAS_ALPHA;

// BLAS level 3 operations

/**
 * @brief In-place Cholesky decomposition of A
 *
 * @param cusolver cuSolver handle, already created
 * @param stream CUDA stream, already created
 * @param f_A matrix to be factorized
 * @param N matrix dimension
 *
 * @return factorized, lower triangular matrix f_L, in-place update of f_A
 */
hpx::shared_future<double *>
potrf(cusolverDnHandle_t cusolver, cudaStream_t stream, hpx::shared_future<double *> f_A, const std::size_t N);

/**
 * @brief In-place solve A(^T) * X = B or X * A(^T) = B for lower triangular A
 *
 * @param cublas cuBLAS handle, already created
 * @param stream CUDA stream, already created
 * @param f_A lower triangular matrix
 * @param f_B right hand side matrix
 * @param M number of rows
 * @param N number of columns
 * @param transpose_A whether to transpose A
 * @param side_A whether to use A on the left or right side
 *
 * @return solution matrix f_X, in-place update of f_B
 */
hpx::shared_future<double *>
trsm(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_B,
     const std::size_t M,
     const std::size_t N,
     const BLAS_TRANSPOSE transpose_A,
     const BLAS_SIDE side_A);

/**
 * @brief Symmetric rank-k update: C = C - A * A^T
 *
 * @param cublas cuBLAS handle, already created
 * @param stream CUDA stream, already created
 * @param f_A matrix
 * @param f_C Symmetric matrix
 * @param N matrix dimension
 *
 * @return updated matrix f_A, inplace update
 */
hpx::shared_future<double *>
syrk(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_C,
     const std::size_t N);

/**
 * @brief General matrix-matrix multiplication: C = C - A(^T) * B(^T)
 *
 * @param cublas cuBLAS handle, already created
 * @param stream CUDA stream, already created
 * @param f_A Left update matrix
 * @param f_B Right update matrix
 * @param f_C Base matrix
 * @param M Number of rows of matrix A
 * @param N Number of columns of matrix B
 * @param K Number of columns of matrix A / rows of matrix B
 * @param transpose_A whether to transpose left matrix A
 * @param transpose_B whether to transpose right matrix B
 *
 * @return updated matrix f_C, in-place update
 */
hpx::shared_future<double *>
gemm(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_B,
     hpx::shared_future<double *> f_C,
     const std::size_t M,
     const std::size_t N,
     const std::size_t K,
     const BLAS_TRANSPOSE transpose_A,
     const BLAS_TRANSPOSE transpose_B);

// BLAS level 2 operations

/**
 * @brief In-place solve A(^T) * x = b where A lower triangular
 *
 * @param cublas cuBLAS handle, already created
 * @param stream CUDA stream, already created
 * @param f_A lower triangular matrix
 * @param f_a right hand side vector
 * @param N matrix dimension
 * @param transpose_A whether to transpose A
 *
 * @return solution vector f_x, in-place update of b
 */
hpx::shared_future<double *>
trsv(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_b,
     const std::size_t N,
     const BLAS_TRANSPOSE transpose_A);

/**
 * @brief General matrix-vector multiplication: y = y - A(^T) * x
 *
 * @param cublas cuBLAS handle, already created
 * @param stream CUDA stream, already created
 * @param f_A update matrix
 * @param f_x update vector
 * @param f_y base vector
 * @param N matrix dimension
 * @param alpha add or substract update to base vector
 * @param transpose_A transpose update matrix
 *
 * @return updated vector f_y, in-place update
 */
hpx::shared_future<double *>
gemv(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_x,
     hpx::shared_future<double *> f_y,
     const std::size_t M,
     const std::size_t N,
     const BLAS_ALPHA alpha,
     const BLAS_TRANSPOSE transpose_A);

/**
 * @brief General matrix rank-1 update: A = A - x*y^T
 *
 * @param cublas cuBLAS handle, already created
 * @param stream CUDA stream, already created
 * @param f_A base matrix
 * @param f_x first update vector
 * @param f_y second update vector
 * @param N matrix dimension
 *
 * @return vector f_b, in-place update
 */
hpx::shared_future<double *>
ger(cublasHandle_t cublas,
    cudaStream_t stream,
    hpx::shared_future<double *> f_A,
    hpx::shared_future<double *> f_x,
    hpx::shared_future<double *> f_y,
    const std::size_t N);

/**
 * @brief Vector update with diagonal SYRK: r = r + diag(A^T * A)
 *
 * @param stream CUDA stream, already created
 * @param f_A update matrix
 * @param f_r base vector
 * @param M number of rows of A
 * @param N number of columns of A
 *
 * @return vector f_r, in-place update
 */
hpx::shared_future<double *>
dot_diag_syrk(cublasHandle_t cublas,
              cudaStream_t stream,
              hpx::shared_future<double *> f_A,
              hpx::shared_future<double *> f_r,
              const std::size_t M,
              const std::size_t N);

/**
 * @brief Vector update with diagonal GEMM: r = r + diag(A * B)
 *
 * @param stream CUDA stream, already created
 * @param f_A first update matrix, of size NxN
 * @param f_B second update matrix, of size NxM
 * @param f_r base vector
 * @param M first matrix dimension
 * @param N second matrix dimension
 *
 * @return updated vector f_r, in-place update
 */
hpx::shared_future<double *>
dot_diag_gemm(cudaStream_t stream,
              hpx::shared_future<double *> f_A,
              hpx::shared_future<double *> f_B,
              hpx::shared_future<double *> f_r,
              const std::size_t M,
              const std::size_t N);

// BLAS level 1 operations

/**
 * @brief Dot product: a * b
 *
 * @param cublas cuBLAS handle, already created
 * @param stream CUDA stream, already created
 * @param f_a left vector
 * @param f_b right vector
 * @param N vector length
 * @return f_a * f_b
 */
hpx::shared_future<double *>
dot(cublasHandle_t cublas,
    cudaStream_t stream,
    hpx::shared_future<double *> f_a,
    hpx::shared_future<double *> f_b,
    const std::size_t N);

// Helper functions

/**
 * @brief Return inverse of cublasOperation_t: transpose or no transpose
 *
 * @see BLAS_TRANSPOSE
 */
inline cublasOperation_t opposite(cublasOperation_t op) { return (op == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N; }

/**
 * @brief Return inverse of cublasSideMode_t: left or right side
 *
 * @see BLAS_SIDE
 */
inline cublasSideMode_t opposite(cublasSideMode_t side)
{
    return (side == CUBLAS_SIDE_LEFT) ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
}

GPRAT_NS_END

#endif
