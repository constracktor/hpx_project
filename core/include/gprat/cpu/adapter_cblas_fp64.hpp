#ifndef GPRAT_CPU_ADAPTER_CBLAS_FP64_HPP
#define GPRAT_CPU_ADAPTER_CBLAS_FP64_HPP

#pragma once

#include "gprat/cpu/blas_enums.hpp"
#include "gprat/detail/config.hpp"
#include "gprat/tile_data.hpp"

#include <span>

GPRAT_NS_BEGIN

// typedef enum BLAS_UPLO { Blas_upper = 121,
//                          Blas_lower = 122 } BLAS_UPLO;

// typedef enum BLAS_ORDERING { Blas_row_major = 101,
//                              Blas_col_major = 102 } BLAS_ORDERING;

// BLAS level 3 operations

/**
 * @brief FP64 In-place Cholesky decomposition of A
 * @param A matrix to be factorized
 * @param N matrix dimension
 * @return factorized, lower triangular matrix L
 */
mutable_tile_data<double> potrf(const mutable_tile_data<double> &A, int N);

/**
 * @brief FP64 In-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
 * @param L Cholesky factor matrix
 * @param A right hand side matrix
 * @param N first dimension
 * @param M second dimension
 * @return solution matrix X
 */
mutable_tile_data<double>
trsm(const const_tile_data<double> &L,
     const mutable_tile_data<double> &A,
     int N,
     int M,
     BLAS_TRANSPOSE transpose_L,
     BLAS_SIDE side_L);

/**
 * @brief FP64 Symmetric rank-k update: A = A - B * B^T
 * @param A Base matrix
 * @param B Symmetric update matrix
 * @param N matrix dimension
 * @return updated matrix A
 */
mutable_tile_data<double> syrk(const mutable_tile_data<double> &A, const const_tile_data<double> &B, int N);

/**
 * @brief FP64 General matrix-matrix multiplication: C = C - A(^T) * B(^T)
 * @param C Base matrix
 * @param B Right update matrix
 * @param A Left update matrix
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @param K third matrix dimension
 * @param transpose_A transpose left matrix
 * @param transpose_B transpose right matrix
 * @return updated matrix X
 */
mutable_tile_data<double>
gemm(const const_tile_data<double> &A,
     const const_tile_data<double> &B,
     const mutable_tile_data<double> &C,
     int N,
     int M,
     int K,
     BLAS_TRANSPOSE transpose_A,
     BLAS_TRANSPOSE transpose_B);

// BLAS level 2 operations

/**
 * @brief FP64 In-place solve L(^T) * x = a where L lower triangular
 * @param L Cholesky factor matrix
 * @param a right hand side vector
 * @param N matrix dimension
 * @param transpose_L transpose Cholesky factor
 * @return solution vector x
 */
mutable_tile_data<double>
trsv(const const_tile_data<double> &L, const mutable_tile_data<double> &a, int N, BLAS_TRANSPOSE transpose_L);

/**
 * @brief FP64 General matrix-vector multiplication: b = b - A(^T) * a
 * @param A update matrix
 * @param a update vector
 * @param b base vector
 * @param N matrix dimension
 * @param alpha add or substract update to base vector
 * @param transpose_A transpose update matrix
 * @return updated vector b
 */
mutable_tile_data<double>
gemv(const const_tile_data<double> &A,
     const const_tile_data<double> &a,
     const mutable_tile_data<double> &b,
     int N,
     int M,
     BLAS_ALPHA alpha,
     BLAS_TRANSPOSE transpose_A);

/**
 * @brief FP64 Vector update with diagonal SYRK: r = r + diag(A^T * A)
 * @param A update matrix
 * @param r base vector
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @return updated vector r
 */
mutable_tile_data<double>
dot_diag_syrk(const const_tile_data<double> &A, const mutable_tile_data<double> &r, int N, int M);

/**
 * @brief FP64 Vector update with diagonal GEMM: r = r + diag(A * B)
 * @param A first update matrix
 * @param B second update matrix
 * @param r base vector
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @return updated vector r
 */
mutable_tile_data<double>
dot_diag_gemm(const const_tile_data<double> &A,
              const const_tile_data<double> &B,
              const mutable_tile_data<double> &r,
              int N,
              int M);

// BLAS level 1 operations

/**
 * @brief FP64 AXPY: y - x
 * @param f_y left vector
 * @param f_x right vector
 * @param N vector length
 * @return y - x
 */
mutable_tile_data<double> axpy(const mutable_tile_data<double> &y, const const_tile_data<double> &x, int N);

/**
 * @brief FP64 Dot product: a * b
 * @param a left vector
 * @param b right vector
 * @param N vector length
 * @return a * b
 */
double dot(std::span<const double> a, std::span<const double> b, int N);

GPRAT_NS_END

#endif
