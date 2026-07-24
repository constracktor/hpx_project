#include "gprat/cpu/adapter_cblas_fp64.hpp"

#include "adapter_cblas_impl.hpp"

#ifdef HPX_HAVE_MODULE_PERFORMANCE_COUNTERS
#include <hpx/performance_counters/manage_counter_type.hpp>
#endif

GPRAT_NS_BEGIN

mutable_tile_data<double> potrf(const mutable_tile_data<double> &A, const int N) { return detail::potrf_impl(A, N); }

mutable_tile_data<double>
trsm(const const_tile_data<double> &L,
     const mutable_tile_data<double> &A,
     const int N,
     const int M,
     const BLAS_TRANSPOSE transpose_L,
     const BLAS_SIDE side_L)
{
    return detail::trsm_impl(L, A, N, M, transpose_L, side_L);
}

mutable_tile_data<double> syrk(const mutable_tile_data<double> &A, const const_tile_data<double> &B, const int N)
{
    return detail::syrk_impl(A, B, N);
}

mutable_tile_data<double>
gemm(const const_tile_data<double> &A,
     const const_tile_data<double> &B,
     const mutable_tile_data<double> &C,
     const int N,
     const int M,
     const int K,
     const BLAS_TRANSPOSE transpose_A,
     const BLAS_TRANSPOSE transpose_B)
{
    return detail::gemm_impl(A, B, C, N, M, K, transpose_A, transpose_B);
}

mutable_tile_data<double> trsv(
    const const_tile_data<double> &L, const mutable_tile_data<double> &a, const int N, const BLAS_TRANSPOSE transpose_L)
{
    return detail::trsv_impl(L, a, N, transpose_L);
}

mutable_tile_data<double>
gemv(const const_tile_data<double> &A,
     const const_tile_data<double> &a,
     const mutable_tile_data<double> &b,
     const int N,
     const int M,
     const BLAS_ALPHA alpha,
     const BLAS_TRANSPOSE transpose_A)
{
    return detail::gemv_impl(A, a, b, N, M, alpha, transpose_A);
}

mutable_tile_data<double>
dot_diag_syrk(const const_tile_data<double> &A, const mutable_tile_data<double> &r, const int N, const int M)
{
    return detail::dot_diag_syrk_impl(A, r, N, M);
}

mutable_tile_data<double>
dot_diag_gemm(const const_tile_data<double> &A,
              const const_tile_data<double> &B,
              const mutable_tile_data<double> &r,
              const int N,
              const int M)
{
    return detail::dot_diag_gemm_impl(A, B, r, N, M);
}

mutable_tile_data<double> axpy(const mutable_tile_data<double> &y, const const_tile_data<double> &x, const int N)
{
    return detail::axpy_impl(y, x, N);
}

double dot(std::span<const double> a, std::span<const double> b, const int N) { return detail::dot_impl(a, b, N); }

#ifdef HPX_HAVE_MODULE_PERFORMANCE_COUNTERS
namespace detail
{
void register_fp64_performance_counters()
{
#define GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(name, fn_expr)                                                              \
    hpx::performance_counters::install_counter_type(                                                                   \
        name "/time",                                                                                                  \
        get_and_reset_function_elapsed<fn_expr>,                                                                       \
        #fn_expr,                                                                                                      \
        "",                                                                                                            \
        hpx::performance_counters::counter_type::monotonically_increasing);                                            \
    hpx::performance_counters::install_counter_type(                                                                   \
        name "/calls",                                                                                                 \
        get_and_reset_function_calls<fn_expr>,                                                                         \
        #fn_expr,                                                                                                      \
        "",                                                                                                            \
        hpx::performance_counters::counter_type::monotonically_increasing)

    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/potrf64", &potrf);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/trsm64", &trsm);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/syrk64", &syrk);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/gemm64", &gemm);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/trsv64", &trsv);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/gemv64", &gemv);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/dot_diag_syrk64", &dot_diag_syrk);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/dot_diag_gemm64", &dot_diag_gemm);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/axpy64", &axpy);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/dot64", &dot);

#undef GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR
}
}  // namespace detail
#endif

GPRAT_NS_END
