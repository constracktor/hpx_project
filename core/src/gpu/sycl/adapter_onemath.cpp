#include "gpu/sycl/adapter_onemath.hpp"

// BLAS LEVEL 3 OPERATIONS ////////////////////////////////////////////////////////////////////////////////////////////

double *potrf(sycl::queue queue, double *f_A, const std::size_t N)
{
    std::int64_t scratchpad_size = oneapi::math::lapack::potrf_scratchpad_size<double>(
        queue, oneapi::math::uplo::upper, static_cast<std::int64_t>(N), static_cast<std::int64_t>(N));

    double *scratchpad = sycl::malloc_device<double>(static_cast<std::size_t>(scratchpad_size), queue);

    // row-major POTRF
    // A = potrf(A)
    // for LOWER part of symmetric positive semi-definite matrix A

    // column-major cuBLAS POTRF for row-major stored A
    // for UPPER part of symmetric positive semi-definite matrix A

    try
    {
        oneapi::math::lapack::potrf(
            queue,
            oneapi::math::uplo::upper,
            static_cast<std::int64_t>(N),
            f_A,
            static_cast<std::int64_t>(N),
            scratchpad,
            scratchpad_size);

        queue.wait();
    }
    catch (...)
    {
        sycl::free(scratchpad, queue);
        throw;
    }

    sycl::free(scratchpad, queue);

    return f_A;
}

double *trsm(sycl::queue queue,
             double *f_A,
             double *f_B,
             const std::size_t M,
             const std::size_t N,
             const oneapi::math::transpose is_transposed,
             const oneapi::math::side is_right)
{
    // TRSM constants
    const double alpha = 1.0;

    // row-major TRSM solves for X
    //
    // for side_A == Blas_right:
    //   op(A) * X = alpha * B
    //     A^T * X = B
    //
    // for side_A == Blas_left:
    //   X * op(A) = alpha * B
    //     X * A^T = B
    //
    // for op: transpose_A

    // column-major cuBLAS TRSM for row-major stored A & B
    // for X on opposite side (opposite of side_A)

    oneapi::math::blas::column_major::trsm(
        queue,
        invert_side_operator(is_right),
        oneapi::math::uplo::upper,
        is_transposed,
        oneapi::math::diag::nonunit,
        static_cast<std::int64_t>(M),
        static_cast<std::int64_t>(N),
        alpha,
        f_A,
        static_cast<std::int64_t>(M),
        f_B,
        static_cast<std::int64_t>(N));

    queue.wait();

    return f_B;
}

double *syrk(sycl::queue queue, double *f_A, double *f_C, const std::size_t N)
{
    // SYRK constants
    const double alpha = -1.0;
    const double beta = 1.0;

    // row-major SYRK
    // C = alpha * op(A) * op(A)^T + beta * C
    //     C = - A * A^T + C
    // for LOWER part of symmetric matrix C
    // for op: NO transpose:

    // column-major cuBLAS SYRK for row-major stored A & C
    // C = - op(A) * op(A)^T + fm(C)
    //   = - A^T * A - C
    // for UPPER part of symmetric matrix C
    // for op: TRANSPOSE

    oneapi::math::blas::column_major::syrk(
        queue,
        oneapi::math::uplo::upper,
        oneapi::math::transpose::trans,
        static_cast<std::int64_t>(N),
        static_cast<std::int64_t>(N),
        alpha,
        f_A,
        static_cast<std::int64_t>(N),
        beta,
        f_C,
        static_cast<std::int64_t>(N));

    queue.wait();

    return f_C;
}

double *gemm(sycl::queue queue,
             double *f_A,
             double *f_B,
             double *f_C,
             const std::size_t M,
             const std::size_t N,
             const std::size_t K,
             const oneapi::math::transpose is_A_transposed,
             const oneapi::math::transpose is_B_transposed)
{
    // row-major GEMM
    // C = alpha * op(A) * op(B) + beta * C
    //   = op(A) * op(B) - C
    // for op(A): transpose_A
    // for op(B): transpose_B

    // column-major cuBLAS GEMM for row-major stored A, B, C
    // C = alpha * op(B) * op(A) + beta * C
    //   = op(B) * op(A) - C
    // for inverted ordering of matrices A, B

    oneapi::math::blas::column_major::gemm(
        queue,
        is_B_transposed,
        is_A_transposed,
        static_cast<std::int64_t>(N),
        static_cast<std::int64_t>(M),
        static_cast<std::int64_t>(K),
        -1.0,
        f_B,
        static_cast<std::int64_t>(N),
        f_A,
        static_cast<std::int64_t>(K),
        1.0,
        f_C,
        static_cast<std::int64_t>(N));

    queue.wait();

    return f_C;
}

// BLAS LEVEL 2 OPERATIONS ////////////////////////////////////////////////////////////////////////////////////////////

double *
trsv(sycl::queue queue, double *f_A, double *f_b, const std::size_t N, const oneapi::math::transpose is_A_transposed)
{
    // row-major TRSV solves for x
    // op(A) * x = b
    // for op: transpose_A
    // for LOWER part of lower triangular matrix A

    // column-major cuBLAS TRSV for row-major stored A
    // for op: opposite of transpose_A
    // for UPPER part of lower triangular matrix A

    oneapi::math::blas::column_major::trsv(
        queue,
        oneapi::math::uplo::upper,
        invert_transpose_operator(is_A_transposed),
        oneapi::math::diag::nonunit,
        static_cast<std::int64_t>(N),
        f_A,
        static_cast<std::int64_t>(N),
        f_b,
        1);

    queue.wait();

    return f_b;
}

double *gemv(sycl::queue queue,
             double *f_A,
             double *f_x,
             double *f_y,
             const std::size_t M,
             const std::size_t N,
             const double alpha,
             const oneapi::math::transpose is_A_transposed)
{
    // GEMV constants
    // const double alpha_value = alpha;
    // const double beta = 1.0;

    // row-major GEMV
    // y = alpha * op(A) * x + beta * y
    //   = alpha * op(A) * x + y
    // for MxN matrix A
    // for vector x
    // for vector y

    // column-major cuBLAS GEMV for row-major stored A (and x,y)
    // for op: opposite of transpose_A

    oneapi::math::blas::column_major::gemv(
        queue,
        invert_transpose_operator(is_A_transposed),
        static_cast<std::int64_t>(N),
        static_cast<std::int64_t>(M),
        alpha,
        f_A,
        static_cast<std::int64_t>(N),
        f_x,
        1,
        1.0,
        f_y,
        1);

    queue.wait();

    return f_y;
}

double *ger(sycl::queue queue, double *f_A, double *f_x, double *f_y, const std::size_t N)
{
    // GER constants
    const double alpha = -1.0;

    // row-major GER
    // A = alpha * x*y^T + A
    //   = -x*y^T + A

    // column-major cuBLAS GER for row-major stored A (and x,y)
    // A = alpha * y*x^T + A
    //   = -y*x^T + A
    // for opposite order of x,y

    oneapi::math::blas::column_major::ger(
        queue,
        static_cast<std::int64_t>(N),
        static_cast<std::int64_t>(N),
        alpha,
        f_y,
        1,
        f_x,
        1,
        f_A,
        static_cast<std::int64_t>(N));

    queue.wait();

    return f_A;
}

DotDiagSyrkKernel::DotDiagSyrkKernel(double *d_A, double *d_r, const std::size_t M, const std::size_t N) :
    d_A(d_A),
    d_r(d_r),
    M(M),
    N(N)
{ }

void DotDiagSyrkKernel::operator()(const sycl::id<1> &id) const
{
    double dot_product = 0.0;

    for (std::size_t i = 0; i < M; ++i)
    {
        dot_product += d_A[i * N + id] * d_A[i * N + id];
    }

    d_r[id] += dot_product;
}

double *dot_diag_syrk(sycl::queue queue, double *f_A, double *f_r, const std::size_t M, const std::size_t N)
{
    // r = r + diag(A^T * A)

    auto event = queue.submit(
        [&](sycl::handler &cgh)
        {
            auto kernel = DotDiagSyrkKernel(f_A, f_r, M, N);
            cgh.parallel_for(sycl::range<1>(N), kernel);
        });
    event.wait();

    return f_r;
}

DotDiagGemmKernel::DotDiagGemmKernel(double *A, double *B, double *r, const std::size_t M, const std::size_t N) :
    A(A),
    B(B),
    r(r),
    M(M),
    N(N)
{ }

void DotDiagGemmKernel::operator()(const sycl::id<1> &id) const
{
    double dot_product = 0.0;

    for (std::size_t i = 0; i < M; ++i)
    {
        dot_product += A[i * N + id] * B[id * M + i];
    }

    r[id] += dot_product;
}

double *
dot_diag_gemm(sycl::queue queue, double *f_A, double *f_B, double *f_r, const std::size_t M, const std::size_t N)
{
    // r = r + diag(A * B)
    auto event = queue.submit(
        [&](sycl::handler &cgh)
        {
            auto kernel = DotDiagGemmKernel(f_A, f_B, f_r, M, N);
            cgh.parallel_for(sycl::range<1>(N), kernel);
        });
    event.wait();

    return f_r;
}

// BLAS LEVEL 1 OPERATIONS ////////////////////////////////////////////////////////////////////////////////////////////

double *dot(sycl::queue queue, double *f_a, double *f_b, const std::size_t N)
{
    // Use shared USM so the result is readable from the host without an explicit copy.
    double *result = sycl::malloc_shared<double>(1, queue);
    *result = 0.0;

    oneapi::math::blas::column_major::dot(queue, static_cast<std::int64_t>(N), f_a, 1, f_b, 1, result);

    queue.wait();

    return result;
}
