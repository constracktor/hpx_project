#include "gprat/cpu/gp_uncertainty.hpp"

#include "gprat/tile_data.hpp"

GPRAT_NS_BEGIN

namespace cpu
{

mutable_tile_data<double> get_matrix_diagonal(const const_tile_data<double> &A, std::size_t M)
{
    mutable_tile_data<double> tile(M);
    for (std::size_t i = 0; i < M; ++i)
    {
        tile.data()[i] = A.data()[i * M + i];
    }
    return tile;
}

}  // end of namespace cpu

GPRAT_NS_END
