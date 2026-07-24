#ifndef GPRAT_CPU_ADAPTER_CBLAS_FP64_ACTIONS_HPP
#define GPRAT_CPU_ADAPTER_CBLAS_FP64_ACTIONS_HPP

#pragma once

#include "gprat/cpu/adapter_cblas_fp64.hpp"
#include "gprat/detail/actions.hpp"
#include "gprat/detail/config.hpp"
#include "gprat/tiled_dataset.hpp"

#include <hpx/actions_base/plain_action.hpp>

GPRAT_NS_BEGIN

namespace cpu
{

hpx::future<tile_handle<double>> potrf_distributed(const tile_handle<double> &A, int N);
hpx::future<tile_handle<double>> trsm_distributed(
    const tile_handle<double> &L,
    const tile_handle<double> &A,
    int N,
    int M,
    BLAS_TRANSPOSE transpose_L,
    BLAS_SIDE side_L);
hpx::future<tile_handle<double>> syrk_distributed(const tile_handle<double> &A, const tile_handle<double> &B, int N);
hpx::future<tile_handle<double>> gemm_distributed(
    const tile_handle<double> &A,
    const tile_handle<double> &B,
    const tile_handle<double> &C,
    int N,
    int M,
    int K,
    BLAS_TRANSPOSE transpose_A,
    BLAS_TRANSPOSE transpose_B);

hpx::future<tile_handle<double>>
trsv_distributed(const tile_handle<double> &L, const tile_handle<double> &a, int N, BLAS_TRANSPOSE transpose_L);
hpx::future<tile_handle<double>> gemv_distributed(
    const tile_handle<double> &A,
    const tile_handle<double> &a,
    const tile_handle<double> &b,
    int N,
    int M,
    BLAS_ALPHA alpha,
    BLAS_TRANSPOSE transpose_A);

hpx::future<tile_handle<double>>
dot_diag_syrk_distributed(const tile_handle<double> &A, const tile_handle<double> &r, int N, int M);
hpx::future<tile_handle<double>> dot_diag_gemm_distributed(
    const tile_handle<double> &A, const tile_handle<double> &B, const tile_handle<double> &r, int N, int M);
hpx::future<tile_handle<double>> axpy_distributed(const tile_handle<double> &y, const tile_handle<double> &x, int N);

// This just gives us the action type (that we want in the correct namespace)
HPX_DEFINE_PLAIN_DIRECT_ACTION(potrf_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(trsm_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(syrk_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gemm_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(trsv_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gemv_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(dot_diag_syrk_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(dot_diag_gemm_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(axpy_distributed);

}  // namespace cpu

GPRAT_NS_END

GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::potrf, GPRAT_NS::cpu::potrf_distributed_action, "POTRF");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::trsm, GPRAT_NS::cpu::trsm_distributed_action, "TRSM");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::syrk, GPRAT_NS::cpu::syrk_distributed_action, "SYRK");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::gemm, GPRAT_NS::cpu::gemm_distributed_action, "GEMM");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::trsv, GPRAT_NS::cpu::trsv_distributed_action, "TRSV");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::gemv, GPRAT_NS::cpu::gemv_distributed_action, "GEMV");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::dot_diag_syrk,
                               GPRAT_NS::cpu::dot_diag_syrk_distributed_action,
                               "dot diag(SYRK)");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::dot_diag_gemm,
                               GPRAT_NS::cpu::dot_diag_gemm_distributed_action,
                               "dot diag(GEMM)");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::axpy, GPRAT_NS::cpu::axpy_distributed_action, "axpy");

#endif
