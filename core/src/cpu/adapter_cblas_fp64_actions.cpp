#include "gprat/cpu/adapter_cblas_fp64_actions.hpp"

#include <hpx/include/performance_counters.hpp>

GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::potrf, GPRAT_NS::cpu::potrf_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::trsm, GPRAT_NS::cpu::trsm_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::syrk, GPRAT_NS::cpu::syrk_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::gemm, GPRAT_NS::cpu::gemm_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::trsv, GPRAT_NS::cpu::trsv_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::gemv, GPRAT_NS::cpu::gemv_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::dot_diag_syrk, GPRAT_NS::cpu::dot_diag_syrk_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::dot_diag_gemm, GPRAT_NS::cpu::dot_diag_gemm_distributed_action)
GPRAT_DEFINE_PLAIN_ACTION_FOR(&GPRAT_NS::axpy, GPRAT_NS::cpu::axpy_distributed_action)

GPRAT_NS_BEGIN

namespace cpu
{
hpx::future<tile_handle<double>> potrf_distributed(const tile_handle<double> &A, int N)
{
    return hpx::dataflow(
        hpx::launch::async,
        [A, N](hpx::future<mutable_tile_data<double>> &&tile) { return A.set_async(potrf(tile.get(), N)); },
        A.get_async());
}

hpx::future<tile_handle<double>> trsm_distributed(
    const tile_handle<double> &L,
    const tile_handle<double> &A,
    int N,
    int M,
    BLAS_TRANSPOSE transpose_L,
    BLAS_SIDE side_L)
{
    return hpx::dataflow(
        hpx::launch::async,
        [A, N, M, transpose_L, side_L](
            hpx::future<mutable_tile_data<double>> &&Ld, hpx::future<mutable_tile_data<double>> &&Ad)
        { return A.set_async(trsm(Ld.get(), Ad.get(), N, M, transpose_L, side_L)); },
        L.get_async(),
        A.get_async());
}

hpx::future<tile_handle<double>> syrk_distributed(const tile_handle<double> &A, const tile_handle<double> &B, int N)
{
    return hpx::dataflow(
        hpx::launch::async,
        [A, N](hpx::future<mutable_tile_data<double>> &&Ad, hpx::future<mutable_tile_data<double>> &&Bd)
        { return A.set_async(syrk(Ad.get(), Bd.get(), N)); },
        A.get_async(),
        B.get_async());
}

hpx::future<tile_handle<double>> gemm_distributed(
    const tile_handle<double> &A,
    const tile_handle<double> &B,
    const tile_handle<double> &C,
    int N,
    int M,
    int K,
    BLAS_TRANSPOSE transpose_A,
    BLAS_TRANSPOSE transpose_B)
{
    return hpx::dataflow(
        hpx::launch::async,
        [C, N, M, K, transpose_A, transpose_B](hpx::future<mutable_tile_data<double>> &&Ad,
                                               hpx::future<mutable_tile_data<double>> &&Bd,
                                               hpx::future<mutable_tile_data<double>> &&Cd)
        { return C.set_async(gemm(Ad.get(), Bd.get(), Cd.get(), N, M, K, transpose_A, transpose_B)); },
        A.get_async(),
        B.get_async(),
        C.get_async());
}

hpx::future<tile_handle<double>>
trsv_distributed(const tile_handle<double> &L, const tile_handle<double> &a, int N, BLAS_TRANSPOSE transpose_L)
{
    return hpx::dataflow(
        hpx::launch::async,
        [a, N, transpose_L](hpx::future<mutable_tile_data<double>> &&Ld, hpx::future<mutable_tile_data<double>> &&ad)
        { return a.set_async(trsv(Ld.get(), ad.get(), N, transpose_L)); },
        L.get_async(),
        a.get_async());
}

hpx::future<tile_handle<double>> gemv_distributed(
    const tile_handle<double> &A,
    const tile_handle<double> &a,
    const tile_handle<double> &b,
    int N,
    int M,
    BLAS_ALPHA alpha,
    BLAS_TRANSPOSE transpose_A)
{
    return hpx::dataflow(
        hpx::launch::async,
        [b, N, M, alpha, transpose_A](hpx::future<mutable_tile_data<double>> &&Ad,
                                      hpx::future<mutable_tile_data<double>> &&ad,
                                      hpx::future<mutable_tile_data<double>> &&bd)
        { return b.set_async(gemv(Ad.get(), ad.get(), bd.get(), N, M, alpha, transpose_A)); },
        A.get_async(),
        a.get_async(),
        b.get_async());
}

hpx::future<tile_handle<double>>
dot_diag_syrk_distributed(const tile_handle<double> &A, const tile_handle<double> &r, int N, int M)
{
    return hpx::dataflow(
        hpx::launch::async,
        [r, N, M](hpx::future<mutable_tile_data<double>> &&Ad, hpx::future<mutable_tile_data<double>> &&rd)
        { return r.set_async(dot_diag_syrk(Ad.get(), rd.get(), N, M)); },
        A.get_async(),
        r.get_async());
}

hpx::future<tile_handle<double>> dot_diag_gemm_distributed(
    const tile_handle<double> &A, const tile_handle<double> &B, const tile_handle<double> &r, int N, int M)
{
    return hpx::dataflow(
        hpx::launch::async,
        [r, N, M](hpx::future<mutable_tile_data<double>> &&Ad,
                  hpx::future<mutable_tile_data<double>> &&Bd,
                  hpx::future<mutable_tile_data<double>> &&rd)
        { return r.set_async(dot_diag_gemm(Ad.get(), Bd.get(), rd.get(), N, M)); },
        A.get_async(),
        B.get_async(),
        r.get_async());
}

hpx::future<tile_handle<double>> axpy_distributed(const tile_handle<double> &y, const tile_handle<double> &x, int N)
{
    return hpx::dataflow(
        hpx::launch::async,
        [y, N](hpx::future<mutable_tile_data<double>> &&yd, hpx::future<mutable_tile_data<double>> &&xd)
        { return y.set_async(axpy(yd.get(), xd.get(), N)); },
        y.get_async(),
        x.get_async());
}
}  // namespace cpu

GPRAT_NS_END
