#ifndef ADAPTER_ONEMATH_H
#define ADAPTER_ONEMATH_H

// INCLUDES ///////////////////////////////////////////////////////////////////////////////////////////////////////////

// GRPat
#include "gprat/target.hpp"

#include "sycl_utils.hpp"

// SYCL
#include <sycl/sycl.hpp>

// oneMath
#include <oneapi/math.hpp>

// BLAS LEVEL 3 OPERATIONS ////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief In-place Cholesky decomposition of f_A
 *
 * @param queue SYCL queue
 * @param f_A matrix to be factorized
 * @param N matrix dimension
 *
 * @return factorized, lower triangular matrix f_L, in-place update of f_A
 */
double *potrf(sycl::queue queue, double *f_A, const std::size_t N);

/**
 * @brief In-place solve A(^T) * X = B or X * A(^T) = B for lower triangular A
 *
 * @param queue SYCL queue
 * @param f_A lower triangular matrix
 * @param f_B right hand side matrix
 * @param M number of rows
 * @param N number of columns
 * @param is_A_transposed whether to transpose A
 * @param is_right whether to use A on the left or right side
 *
 * @return solution matrix f_X, in-place update of f_B
 */
double *trsm(sycl::queue queue,
             double *f_A,
             double *f_B,
             const std::size_t M,
             const std::size_t N,
             const oneapi::math::transpose is_transposed,
             const oneapi::math::side is_right);

/**
 * @brief Symmetric rank-k update: C = C - A * A^T
 *
 * @param queue SYCL queue
 * @param f_A matrix
 * @param f_C Symmetric matrix
 * @param N matrix dimension
 *
 * @return updated matrix f_A, in-place update
 */
double *syrk(sycl::queue queue, double *f_A, double *f_C, const std::size_t N);

/**
 * @brief General matrix-matrix multiplication: C = C - A(^T) * B(^T)
 *
 * @param queue SYCL queue
 * @param f_A Left update matrix
 * @param f_B Right update matrix
 * @param f_C Base matrix
 * @param M Number of rows of matrix A
 * @param N Number of columns of matrix B
 * @param K Number of columns of matrix A / rows of matrix B
 * @param is_A_transposed whether to transpose left matrix A
 * @param is_B_transposed whether to transpose right matrix B
 *
 * @return updated matrix f_C, in-place update
 */
double *gemm(sycl::queue queue,
             double *f_A,
             double *f_B,
             double *f_C,
             const std::size_t M,
             const std::size_t N,
             const std::size_t K,
             const oneapi::math::transpose is_A_transposed,
             const oneapi::math::transpose is_B_transposed);

// BLAS LEVEL 2 OPERATIONS ////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief In-place solve A(^T) * x = b where A lower triangular
 *
 * @param queue SYCL queue
 * @param f_A lower triangular matrix
 * @param f_a right hand side vector
 * @param N matrix dimension
 * @param is_A_transposed whether to transpose A
 *
 * @return solution vector f_x, in-place update of b
 */
double *
trsv(sycl::queue queue, double *f_A, double *f_b, const std::size_t N, const oneapi::math::transpose is_A_transposed);

/**
 * @brief General matrix-vector multiplication: y = y - A(^T) * x
 *
 * @param queue SYCL queue
 * @param f_A update matrix
 * @param f_x update vector
 * @param f_y base vector
 * @param N matrix dimension
 * @param alpha add or substract update to base vector
 * @param is_A_transposed transpose update matrix
 *
 * @return updated vector f_y, in-place update
 */
double *gemv(sycl::queue queue,
             double *f_A,
             double *f_x,
             double *f_y,
             const std::size_t M,
             const std::size_t N,
             const double alpha,
             const oneapi::math::transpose is_A_transposed);

/**
 * @brief General matrix rank-1 update: A = A - x*y^T
 *
 * @param queue SYCL queue
 * @param f_A base matrix
 * @param f_x first update vector
 * @param f_y second update vector
 * @param N matrix dimension
 *
 * @return vector f_b, in-place update
 */
double *ger(sycl::queue queue, double *f_A, double *f_x, double *f_y, const std::size_t N);

/**
 * @brief Vector update with diagonal SYRK: r = r + diag(A^T * A)
 *
 * @param queue SYCL queue
 * @param f_A update matrix
 * @param f_r base vector
 * @param M number of rows of A
 * @param N number of columns of A
 *
 * @return vector f_r, in-place update
 */
double *dot_diag_syrk(sycl::queue queue, double *f_A, double *f_r, const std::size_t M, const std::size_t N);

/**
 * @brief Kernel class for vector update with diagonal SYRK
 */
class DotDiagSyrkKernel
{
  private:
    double *d_A;
    double *d_r;
    std::size_t M;
    std::size_t N;

  public:
    /**
     * @brief Constructor for DotDiagSyrkKernel for vector update with diagonal SYRK
     *
     * @param A update matrix
     * @param r base vector
     * @param M number of rows of A
     * @param N number of columns of A
     */
    explicit DotDiagSyrkKernel(double *A, double *r, const std::size_t M, const std::size_t N);

    /**
     * @brief The operator() of DotDiagSyrkKernel implements the actual kernel
     *
     * @param id global SYCL id of the kernel
     */
    void operator()(const sycl::id<1> &id) const;
};

/**
 * @brief Vector update with diagonal GEMM: r = r + diag(A * B)
 *
 * @param queue SYCL queue
 * @param f_A first update matrix, of size NxN
 * @param f_B second update matrix, of size NxM
 * @param f_r base vector
 * @param M first matrix dimension
 * @param N second matrix dimension
 *
 * @return updated vector f_r, in-place update
 */
double *
dot_diag_gemm(sycl::queue queue, double *f_A, double *f_B, double *f_r, const std::size_t M, const std::size_t N);

/**
 * @brief Kernel class for vector update with diagonal GEMM
 */
class DotDiagGemmKernel
{
  private:
    double *A;
    double *B;
    double *r;
    std::size_t M;
    std::size_t N;

  public:
    /**
     * @brief Constructor for DotDiagGemmKernel for vector update with diagonal GEMM
     *
     * @param A first update matrix, of size NxN
     * @param B second update matrix, of size NxM
     * @param r base vector
     * @param M first matrix dimension
     * @param N second matrix dimension
     */
    explicit DotDiagGemmKernel(double *A, double *B, double *r, const std::size_t M, const std::size_t N);

    /**
     * @brief The operator() of DotDiagGemmKernel implements the actual kernel
     *
     * @param id global SYCL id of the kernel
     */
    void operator()(const sycl::id<1> &id) const;
};

// BLAS LEVEL 1 OPERATIONS ////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Dot product: a * b
 *
 * @param queue SYCL queue
 * @param f_a left vector
 * @param f_b right vector
 * @param N vector length
 * @return f_a * f_b
 */
double *dot(sycl::queue queue, double *f_a, double *f_b, const std::size_t N);

// HELPER FUNCTIONS ///////////////////////////////////////////////////////////////////////////////////////////////////

inline oneapi::math::transpose invert_transpose_operator(oneapi::math::transpose op)
{
    return (op == oneapi::math::transpose::nontrans)
               ? oneapi::math::transpose::trans
               : oneapi::math::transpose::nontrans;
}

inline oneapi::math::side invert_side_operator(oneapi::math::side op)
{
    return (op == oneapi::math::side::left) ? oneapi::math::side::right : oneapi::math::side::left;
}

#endif  // end of ADAPTER_ONEMATH_H
