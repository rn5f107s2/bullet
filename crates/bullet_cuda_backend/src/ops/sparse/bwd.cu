#ifndef STUFF
#define INV_DERIV 1.0F
#define BIAS_ARG
#define DECL_MAXY 1
#define DECL_M 1
#define DECL_NNZ 1
#define BIAS_BACKPROP
#endif

constexpr int MaximumBlocksY = DECL_MAXY;
constexpr int m = DECL_M;
constexpr int nnz = DECL_NNZ;

__device__ float op([[maybe_unused]] float x) {
    return INV_DERIV;
}

constexpr int N = 16;
constexpr int HL = 64 * N;

extern "C" __global__ void kernel(
    const int k,
    const int* X,
    const float* Y,
    const float* Yg,
    float* Ag
    BIAS_ARG)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= k * N)
        return;

    const int* thisInput = X + k * blockIdx.y;
    const float* thisErrors  = Yg + m * blockIdx.y;
    const float* thisOutputs = Y + m * blockIdx.y; 

    int index = elem / N;

    const int feat = thisInput[index];

    const int featSqOur = feat % 64;
    const int featPcOur = feat % 64;

    const int ourIndex = featPcOur * HL + featSqOur * N + elem % N;

    float ourError = *(thisErrors + ourIndex) * op(thisOutputs[ourIndex]);

    for (int i = 0; i < nnz; i++) {
        const int j = thisInput[i];

        if (j == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(j) * HL * 12 + ourIndex;

        if (ourError != 0.0F)
            atomicAdd(&Ag[ourIdx], ourError);
    }
}
