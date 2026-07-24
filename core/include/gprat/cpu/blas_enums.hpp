#ifndef GPRAT_CPU_BLAS_ENUMS_HPP
#define GPRAT_CPU_BLAS_ENUMS_HPP

#pragma once

#include "gprat/detail/config.hpp"

GPRAT_NS_BEGIN

// Constants that are compatible with CBLAS
typedef enum BLAS_TRANSPOSE { Blas_no_trans = 111, Blas_trans = 112 } BLAS_TRANSPOSE;

typedef enum BLAS_SIDE { Blas_left = 141, Blas_right = 142 } BLAS_SIDE;

typedef enum BLAS_ALPHA { Blas_add = 1, Blas_substract = -1 } BLAS_ALPHA;

GPRAT_NS_END

#endif
