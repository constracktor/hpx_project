#ifndef GPRAT_CPU_GP_UNCERTAINTY_ACTIONS_HPP
#define GPRAT_CPU_GP_UNCERTAINTY_ACTIONS_HPP

#pragma once

#include "gprat/cpu/gp_uncertainty.hpp"
#include "gprat/detail/actions.hpp"
#include "gprat/detail/config.hpp"
#include "gprat/tiled_dataset.hpp"

GPRAT_NS_BEGIN

namespace cpu
{
hpx::future<tile_handle<double>> get_matrix_diagonal_distributed(const tile_handle<double> &A, std::size_t M);
HPX_DEFINE_PLAIN_DIRECT_ACTION(get_matrix_diagonal_distributed);
}  // namespace cpu

GPRAT_NS_END

GPRAT_DECLARE_PLAIN_ACTION_FOR(&GPRAT_NS::cpu::get_matrix_diagonal,
                               GPRAT_NS::cpu::get_matrix_diagonal_distributed_action,
                               "get_matrix_diagonal");

#endif
