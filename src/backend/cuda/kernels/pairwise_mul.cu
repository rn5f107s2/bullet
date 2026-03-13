/*
Computes
N = len(input_vector)
output_vector = input_vector[:N] * input_vector[N:]
(and gradients thereof)
*/
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int N = 8;

constexpr size_t threadsPerBlock = static_cast<size_t>(1024);

__global__ void pairwiseMulKernel(
    const size_t tensorSize,
    const float* inp,
    float* out) {
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= tensorSize * 2)
        return;

    if (tid % N >= (N / 2))
        return;

    const float* thisInp = inp + 2 * tensorSize * blockIdx.y + tid;
    // tid % N is never >= N / 2 so this should be fine
    float* thisOut = out + tensorSize * blockIdx.y + (tid / N) + (tid % N);

    thisOut[0] = thisInp[0] * thisInp[N / 2];
}

extern "C" void pairwiseMul(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const float* input,
    float* output) {
    const size_t grid_x = (outputSize + threadsPerBlock - 1) / threadsPerBlock;
    const dim3 grid(grid_x, batchSize);

    pairwiseMulKernel<<<grid, threadsPerBlock>>>(outputSize, input, output);
}

__global__ void pairwiseMulBackwardKernel(
    const size_t tensorSize,
    const float* inp,
    float* out) {
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= tensorSize * 2)
        return;

    if (tid % N >= (N / 2))
        return;

    const float* thisInp = inp + tensorSize * blockIdx.y + (tid / N) + (tid % N);
    float* thisOut = out + 2 * tensorSize * blockIdx.y + tid;

    const float gradIn = thisInp[0];
    const float valLeft = thisOut[0];
    const float valRight = thisOut[N / 2];
    const float gradLeft = gradIn * valRight;
    const float gradRight = gradIn * valLeft;

    thisOut[0] = gradLeft;
    thisOut[N / 2] = gradRight;
}

extern "C" void backpropPairwiseMul(
    const size_t batchSize,
    const size_t inputSize,
    const size_t outputSize,
    const float* input,
    float* output) {
    const size_t grid_x = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
    const dim3 grid(grid_x, batchSize);

    pairwiseMulBackwardKernel<<<grid, threadsPerBlock>>>(inputSize, input, output);
}