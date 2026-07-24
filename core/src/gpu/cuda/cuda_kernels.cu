#include "gprat/gpu/cuda_kernels.cuh"

#include "gprat/gpu/cuda_utils.cuh"

GPRAT_NS_BEGIN

__global__ void transpose(double *transposed, double *original, std::size_t width, std::size_t height)
{
    __shared__ double block[BLOCK_SIZE][BLOCK_SIZE + 1];

    std::size_t xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    std::size_t yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (xIndex < width && yIndex < height)
    {
        std::size_t index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = original[index_in];
    }

    __syncthreads();

    xIndex = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    if (xIndex < height && yIndex < width)
    {
        std::size_t index_out = yIndex * height + xIndex;
        transposed[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

GPRAT_NS_END
