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

constexpr int N  = 16;
constexpr int HL = 64 * N; 

__device__ float op([[maybe_unused]] float x) {
    return INV_DERIV;
}

extern "C" __global__ void kernel(
    const int k,
    const int* X,
    const float* Y,
    const float* Yg,
    float* Ag
    BIAS_ARG)
{
    const int loc = MaximumBlocksY * blockIdx.z + blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    const int elem = m * loc + row;

    if (elem >= 768 * N)
        return;

    const int* tX = X + nnz * loc;
    const int offset = m * loc;

    const float tE = op(Y[offset + row]) * Yg[offset + row];

    int index = elem / N;

    const int feat = X[index];

    const int featSq = feat % 64;
    const int featPc = feat / 64;

    const int ourIndex = featPc * HL + featSq * N + elem % N;

    for (int i = 0; i < nnz; i++) {
        const int j = tX[i];

        if (j == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(j) * HL * 12 + ourIndex;

        if (tE != 0.0F)
            atomicAdd(&Ag[ourIdx], tE);
    }
}
