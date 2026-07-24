// Shared templated implementation for fp32 and fp64 CBLAS adapters.
// Included directly by adapter_cblas_fp32.cpp and adapter_cblas_fp64.cpp.
// Not a public header — do not include from elsewhere.

#pragma once

#include "gprat/performance_counters.hpp"
#include "gprat/tile_data.hpp"

#ifdef HPX_HAVE_MODULE_PERFORMANCE_COUNTERS
#include <hpx/performance_counters/manage_counter_type.hpp>
#endif

#ifdef GPRAT_ENABLE_MKL
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

#include <span>
#include <type_traits>

GPRAT_NS_BEGIN

namespace detail
{

// BLAS level 3 ///////////////////////////////////////////////////////////////

template <typename T>
mutable_tile_data<T> potrf_impl(const mutable_tile_data<T> &A, const int N)
{
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_TIME_FUNCTION(&potrf);
    if constexpr (std::is_same_v<T, float>)
    {
        LAPACKE_spotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
    }
    else
    {
        LAPACKE_dpotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
    }
    return A;
}

template <typename T>
mutable_tile_data<T>
trsm_impl(const const_tile_data<T> &L,
          const mutable_tile_data<T> &A,
          const int N,
          const int M,
          const BLAS_TRANSPOSE transpose_L,
          const BLAS_SIDE side_L)
{
    GPRAT_BENCHMARK_FORCE_EVICT(L.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_TIME_FUNCTION(&trsm);
    const T alpha = T(1);
    if constexpr (std::is_same_v<T, float>)
    {
        cblas_strsm(
            CblasRowMajor,
            static_cast<CBLAS_SIDE>(side_L),
            CblasLower,
            static_cast<CBLAS_TRANSPOSE>(transpose_L),
            CblasNonUnit,
            N,
            M,
            alpha,
            L.data(),
            N,
            A.data(),
            M);
    }
    else
    {
        cblas_dtrsm(
            CblasRowMajor,
            static_cast<CBLAS_SIDE>(side_L),
            CblasLower,
            static_cast<CBLAS_TRANSPOSE>(transpose_L),
            CblasNonUnit,
            N,
            M,
            alpha,
            L.data(),
            N,
            A.data(),
            M);
    }
    return A;
}

template <typename T>
mutable_tile_data<T> syrk_impl(const mutable_tile_data<T> &A, const const_tile_data<T> &B, const int N)
{
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(B.as_span());
    GPRAT_TIME_FUNCTION(&syrk);
    const T alpha = T(-1);
    const T beta = T(1);
    if constexpr (std::is_same_v<T, float>)
    {
        cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, N, alpha, B.data(), N, beta, A.data(), N);
    }
    else
    {
        cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, N, alpha, B.data(), N, beta, A.data(), N);
    }
    return A;
}

template <typename T>
mutable_tile_data<T>
gemm_impl(const const_tile_data<T> &A,
          const const_tile_data<T> &B,
          const mutable_tile_data<T> &C,
          const int N,
          const int M,
          const int K,
          const BLAS_TRANSPOSE transpose_A,
          const BLAS_TRANSPOSE transpose_B)
{
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(B.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(C.as_span());
    GPRAT_TIME_FUNCTION(&gemm);
    const T alpha = T(-1);
    const T beta = T(1);
    if constexpr (std::is_same_v<T, float>)
    {
        cblas_sgemm(
            CblasRowMajor,
            static_cast<CBLAS_TRANSPOSE>(transpose_A),
            static_cast<CBLAS_TRANSPOSE>(transpose_B),
            K,
            M,
            N,
            alpha,
            A.data(),
            K,
            B.data(),
            M,
            beta,
            C.data(),
            M);
    }
    else
    {
        cblas_dgemm(
            CblasRowMajor,
            static_cast<CBLAS_TRANSPOSE>(transpose_A),
            static_cast<CBLAS_TRANSPOSE>(transpose_B),
            K,
            M,
            N,
            alpha,
            A.data(),
            K,
            B.data(),
            M,
            beta,
            C.data(),
            M);
    }
    return C;
}

// BLAS level 2 ///////////////////////////////////////////////////////////////

template <typename T>
mutable_tile_data<T>
trsv_impl(const const_tile_data<T> &L, const mutable_tile_data<T> &a, const int N, const BLAS_TRANSPOSE transpose_L)
{
    GPRAT_BENCHMARK_FORCE_EVICT(L.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(a.as_span());
    GPRAT_TIME_FUNCTION(&trsv);
    if constexpr (std::is_same_v<T, float>)
    {
        cblas_strsv(CblasRowMajor,
                    CblasLower,
                    static_cast<CBLAS_TRANSPOSE>(transpose_L),
                    CblasNonUnit,
                    N,
                    L.data(),
                    N,
                    a.data(),
                    1);
    }
    else
    {
        cblas_dtrsv(CblasRowMajor,
                    CblasLower,
                    static_cast<CBLAS_TRANSPOSE>(transpose_L),
                    CblasNonUnit,
                    N,
                    L.data(),
                    N,
                    a.data(),
                    1);
    }
    return a;
}

template <typename T>
mutable_tile_data<T>
gemv_impl(const const_tile_data<T> &A,
          const const_tile_data<T> &a,
          const mutable_tile_data<T> &b,
          const int N,
          const int M,
          const BLAS_ALPHA alpha,
          const BLAS_TRANSPOSE transpose_A)
{
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(a.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(b.as_span());
    GPRAT_TIME_FUNCTION(&gemv);
    const T beta = T(1);
    if constexpr (std::is_same_v<T, float>)
    {
        cblas_sgemv(
            CblasRowMajor,
            static_cast<CBLAS_TRANSPOSE>(transpose_A),
            N,
            M,
            alpha,
            A.data(),
            M,
            a.data(),
            1,
            beta,
            b.data(),
            1);
    }
    else
    {
        cblas_dgemv(
            CblasRowMajor,
            static_cast<CBLAS_TRANSPOSE>(transpose_A),
            N,
            M,
            alpha,
            A.data(),
            M,
            a.data(),
            1,
            beta,
            b.data(),
            1);
    }
    return b;
}

template <typename T>
mutable_tile_data<T>
dot_diag_syrk_impl(const const_tile_data<T> &A, const mutable_tile_data<T> &r, const int N, const int M)
{
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(r.as_span());
    GPRAT_TIME_FUNCTION(&dot_diag_syrk);
    auto r_p = r.data();
    auto A_p = A.data();
    for (std::size_t j = 0; j < static_cast<std::size_t>(M); ++j)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            r_p[j] += cblas_sdot(N, &A_p[j], M, &A_p[j], M);
        }
        else
        {
            r_p[j] += cblas_ddot(N, &A_p[j], M, &A_p[j], M);
        }
    }
    return r;
}

template <typename T>
mutable_tile_data<T> dot_diag_gemm_impl(
    const const_tile_data<T> &A, const const_tile_data<T> &B, const mutable_tile_data<T> &r, const int N, const int M)
{
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(B.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(r.as_span());
    GPRAT_TIME_FUNCTION(&dot_diag_gemm);
    auto r_p = r.data();
    auto A_p = A.data();
    auto B_p = B.data();
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            r_p[i] += cblas_sdot(M, &A_p[i * static_cast<std::size_t>(M)], 1, &B_p[i], N);
        }
        else
        {
            r_p[i] += cblas_ddot(M, &A_p[i * static_cast<std::size_t>(M)], 1, &B_p[i], N);
        }
    }
    return r;
}

// BLAS level 1 ///////////////////////////////////////////////////////////////

template <typename T>
mutable_tile_data<T> axpy_impl(const mutable_tile_data<T> &y, const const_tile_data<T> &x, const int N)
{
    GPRAT_BENCHMARK_FORCE_EVICT(y.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(x.as_span());
    GPRAT_TIME_FUNCTION(&axpy);
    if constexpr (std::is_same_v<T, float>)
    {
        cblas_saxpy(N, T(-1), x.data(), 1, y.data(), 1);
    }
    else
    {
        cblas_daxpy(N, T(-1), x.data(), 1, y.data(), 1);
    }
    return y;
}

template <typename T>
T dot_impl(std::span<const T> a, std::span<const T> b, const int N)
{
    GPRAT_BENCHMARK_FORCE_EVICT(a);
    GPRAT_BENCHMARK_FORCE_EVICT(b);
    GPRAT_TIME_FUNCTION(&dot);
    if constexpr (std::is_same_v<T, float>)
    {
        return cblas_sdot(N, a.data(), 1, b.data(), 1);
    }
    else
    {
        return cblas_ddot(N, a.data(), 1, b.data(), 1);
    }
}

}  // namespace detail

GPRAT_NS_END
