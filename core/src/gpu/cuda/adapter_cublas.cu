#include "gprat/gpu/adapter_cublas.cuh"

GPRAT_NS_BEGIN

// frequently used names
using hpx::cuda::experimental::check_cuda_error;

// BLAS level 3 operations

hpx::shared_future<double *>
potrf(cusolverDnHandle_t cusolver, cudaStream_t stream, hpx::shared_future<double *> f_A, const std::size_t N)
{
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);

    int *d_info = nullptr;
    check_cuda_error(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    size_t workspaceInBytesOnDevice;
    void *d_work = nullptr;
    size_t workspaceInBytesOnHost;
    void *h_work = nullptr;

    double *d_A = f_A.get();

    cusolverDnXpotrf_bufferSize(
        cusolver,
        params,
        CUBLAS_FILL_MODE_UPPER,
        N,
        CUDA_R_64F,
        d_A,
        N,
        CUDA_R_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost);

    check_cuda_error(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

    if (workspaceInBytesOnHost > 0)
    {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr)
        {
            check_cuda_error(cudaFree(d_work));
            check_cuda_error(cudaFree(d_info));
            cusolverDnDestroyParams(params);
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    // row-major POTRF
    // A = potrf(A)
    // for LOWER part of symmetric positive semi-definite matrix A

    // column-major cuBLAS POTRF for row-major stored A
    // for UPPER part of symmetric positive semi-definite matrix A

    try
    {
        cusolverDnXpotrf(
            cusolver,
            params,
            CUBLAS_FILL_MODE_UPPER,
            N,
            CUDA_R_64F,
            d_A,
            N,
            CUDA_R_64F,
            d_work,
            workspaceInBytesOnDevice,
            h_work,
            workspaceInBytesOnHost,
            d_info);
        check_cuda_error(cudaStreamSynchronize(stream));
    }
    catch (...)
    {
        check_cuda_error(cudaFree(d_work));
        if (h_work != nullptr)
        {
            free(h_work);
        }
        check_cuda_error(cudaFree(d_info));
        cusolverDnDestroyParams(params);
        throw;
    }

    check_cuda_error(cudaFree(d_work));
    if (h_work != nullptr)
    {
        free(h_work);
    }
    check_cuda_error(cudaFree(d_info));
    cusolverDnDestroyParams(params);

    return hpx::make_ready_future(d_A);
}

hpx::shared_future<double *>
trsm(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_B,
     const std::size_t M,
     const std::size_t N,
     const BLAS_TRANSPOSE transpose_A,
     const BLAS_SIDE side_A)
{
    // TRSM constants
    const double alpha = 1.0;
    double *d_A = f_A.get();
    double *d_B = f_B.get();

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

    cublasDtrsm(
        cublas,
        opposite(static_cast<cublasSideMode_t>(side_A)),
        CUBLAS_FILL_MODE_UPPER,
        static_cast<cublasOperation_t>(transpose_A),
        CUBLAS_DIAG_NON_UNIT,
        N,
        M,
        &alpha,
        d_A,
        M,
        d_B,
        N);

    check_cuda_error(cudaStreamSynchronize(stream));
    return hpx::make_ready_future(d_B);
}

hpx::shared_future<double *>
syrk(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_C,
     const std::size_t N)
{
    // SYRK constants
    const double alpha = -1.0;
    const double beta = 1.0;
    double *d_A = f_A.get();
    double *d_C = f_C.get();

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

    cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, N, N, &alpha, d_A, N, &beta, d_C, N);

    check_cuda_error(cudaStreamSynchronize(stream));
    return hpx::make_ready_future(d_C);
}

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
     const BLAS_TRANSPOSE transpose_B)
{
    const double alpha = -1.0;
    const double beta = 1.0;
    double *d_A = f_A.get();
    double *d_B = f_B.get();
    double *d_C = f_C.get();

    // row-major GEMM
    // C = alpha * op(A) * op(B) + beta * C
    //   = op(A) * op(B) - C
    // for op(A): transpose_A
    // for op(B): transpose_B

    // column-major cuBLAS GEMM for row-major stored A, B, C
    // C = alpha * op(B) * op(A) + beta * C
    //   = op(B) * op(A) - C
    // for inverted ordering of matrices A, B

    cublasDgemm(
        cublas,
        static_cast<cublasOperation_t>(transpose_B),
        static_cast<cublasOperation_t>(transpose_A),
        N,
        M,
        K,
        &alpha,
        d_B,
        N,
        d_A,
        K,
        &beta,
        d_C,
        N);

    check_cuda_error(cudaStreamSynchronize(stream));
    return hpx::make_ready_future(d_C);
}

// BLAS level 2 operations

hpx::shared_future<double *>
trsv(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_b,
     const std::size_t N,
     const BLAS_TRANSPOSE transpose_A)
{
    double *d_A = f_A.get();
    double *d_b = f_b.get();

    // row-major TRSV solves for x
    // op(A) * x = b
    // for op: transpose_A
    // for LOWER part of lower triangular matrix A

    // column-major cuBLAS TRSV for row-major stored A
    // for op: opposite of transpose_A
    // for UPPER part of lower triangular matrix A

    cublasDtrsv(cublas,
                CUBLAS_FILL_MODE_UPPER,
                opposite(static_cast<cublasOperation_t>(transpose_A)),
                CUBLAS_DIAG_NON_UNIT,
                N,
                d_A,
                N,
                d_b,
                1);

    check_cuda_error(cudaStreamSynchronize(stream));

    // return solution vector x
    return hpx::make_ready_future(d_b);
}

hpx::shared_future<double *>
gemv(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_x,
     hpx::shared_future<double *> f_y,
     const std::size_t M,
     const std::size_t N,
     const BLAS_ALPHA alpha,
     const BLAS_TRANSPOSE transpose_A)
{
    auto d_A = f_A.get();
    auto d_x = f_x.get();
    auto d_y = f_y.get();

    const double alpha_value = static_cast<double>(alpha);
    const double beta = 1.0;

    // row-major GEMV
    // y = alpha * op(A) * x + beta * y
    //   = alpha * op(A) * x + y
    // for MxN matrix A
    // for vector x
    // for vector y

    // column-major cuBLAS GEMV for row-major stored A (and x,y)
    // for op: opposite of transpose_A

    cublasDgemv(
        cublas,
        opposite(static_cast<cublasOperation_t>(transpose_A)),
        N,
        M,
        &alpha_value,
        d_A,
        N,
        d_x,
        1,
        &beta,
        d_y,
        1);

    check_cuda_error(cudaStreamSynchronize(stream));
    // return updated vector b
    return hpx::make_ready_future(d_y);
}

hpx::shared_future<double *>
ger(cublasHandle_t cublas,
    cudaStream_t stream,
    hpx::shared_future<double *> f_A,
    hpx::shared_future<double *> f_x,
    hpx::shared_future<double *> f_y,
    const std::size_t N)
{
    double *d_A = f_A.get();
    double *d_x = f_x.get();
    double *d_y = f_y.get();
    const double alpha = -1.0;

    // row-major GER
    // A = alpha * x*y^T + A
    //   = -x*y^T + A

    // column-major cuBLAS GER for row-major stored A (and x,y)
    // A = alpha * y*x^T + A
    //   = -y*x^T + A
    // for opposite order of x,y

    cublasDger(cublas, N, N, &alpha, d_y, 1, d_x, 1, d_A, N);

    check_cuda_error(cudaStreamSynchronize(stream));
    // Return updated A
    return hpx::make_ready_future(d_A);
}

__global__ void
dot_diag_syrk_kernel(cublasHandle_t cublas, double *d_A, double *d_r, const std::size_t M, const std::size_t N)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N)
    {
        double dot_product = 0.0;
        for (int i = 0; i < M; ++i)
        {
            dot_product += d_A[i * N + j] * d_A[i * N + j];
        }
        d_r[j] += dot_product;
    }
}

hpx::shared_future<double *>
dot_diag_syrk(cublasHandle_t cublas,
              cudaStream_t stream,
              hpx::shared_future<double *> f_A,
              hpx::shared_future<double *> f_r,
              const std::size_t M,
              const std::size_t N)
{
    auto d_A = f_A.get();
    auto d_r = f_r.get();

    // r = r + diag(A^T * A)
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    dot_diag_syrk_kernel<<<numBlocks, blockSize, 0, stream>>>(cublas, d_A, d_r, M, N);
    cudaStreamSynchronize(stream);

    return hpx::make_ready_future(d_r);
}

__global__ void dot_diag_gemm_kernel(double *A, double *B, double *r, const std::size_t M, const std::size_t N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            r[i] += A[i * N + j] * B[j * M + i];
        }
    }
}

hpx::shared_future<double *>
dot_diag_gemm(cudaStream_t stream,
              hpx::shared_future<double *> f_A,
              hpx::shared_future<double *> f_B,
              hpx::shared_future<double *> f_r,
              const std::size_t M,
              const std::size_t N)
{
    int blockSize = 256;
    int numBlocks = (M + blockSize - 1) / blockSize;

    double *d_A = f_A.get();
    double *d_B = f_B.get();
    double *d_r = f_r.get();

    // r = r + diag(A * B)
    dot_diag_gemm_kernel<<<numBlocks, blockSize, 0, stream>>>(d_A, d_B, d_r, M, N);
    cudaStreamSynchronize(stream);

    return hpx::make_ready_future(d_r);
}

// BLAS level 1 operations

hpx::shared_future<double *>
dot(cublasHandle_t cublas,
    cudaStream_t stream,
    hpx::shared_future<double *> f_a,
    hpx::shared_future<double *> f_b,
    const std::size_t N)
{
    double *result;
    check_cuda_error(cudaMalloc(&result, sizeof(double)));
    check_cuda_error(cudaMemset(result, 0, sizeof(double)));

    double *d_a = f_a.get();
    double *d_b = f_b.get();

    // `result` is a device pointer, but the handle (shared with host-scalar BLAS calls)
    // defaults to CUBLAS_POINTER_MODE_HOST. Switch to DEVICE mode for this call and
    // restore the previous mode afterwards so other users of the handle are unaffected.
    cublasPointerMode_t prev_mode;
    cublasGetPointerMode(cublas, &prev_mode);
    cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
    cublasDdot(cublas, N, d_a, 1, d_b, 1, result);
    cublasSetPointerMode(cublas, prev_mode);
    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(result);
}

GPRAT_NS_END
