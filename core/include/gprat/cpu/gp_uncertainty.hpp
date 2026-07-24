#ifndef GPRAT_CPU_GP_UNCERTAINTY_HPP
#define GPRAT_CPU_GP_UNCERTAINTY_HPP

#pragma once

#include "gprat/detail/config.hpp"
#include "gprat/tile_data.hpp"

GPRAT_NS_BEGIN

namespace cpu
{

/**
 * @brief Extract diagonal elements of the matrix A.
 *
 * @param A The matrix
 * @param M The rumber of rows in the matrix
 *
 * @return Diagonal element vector of the matrix A of size M
 */
mutable_tile_data<double> get_matrix_diagonal(const const_tile_data<double> &A, std::size_t M);

}  // end of namespace cpu

GPRAT_NS_END

#endif  // end of CPU_GP_UNCERTAINTY_H
