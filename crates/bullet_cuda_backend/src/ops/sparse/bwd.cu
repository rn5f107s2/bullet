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

constexpr int N = 16;
constexpr int HL = N * 64;

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

    if (row >= nnz * N || loc >= k)
        return;

    const int* tX = X + nnz * loc;
    const int offset = m * loc;

    int index = row / N;

    const int feat = tX[index];

    if (feat == -1)
        return;

    const int sq = feat / 64;
    const int pc = feat % 64;

    const int idx = row % N;

    const int nRow = pc * HL + sq * N + idx;

    const float tE = op(Y[offset + nRow]) * Yg[offset + nRow];

    if (isnan(Y[offset + nRow])) {
        int* f = reinterpret_cast<int*>(0xDEADBEEF);
        *f = 0;
    }

    for (int i = 0; i < nnz; i++) {
        const int j = tX[i];

        if (j == -1)
            break;

        if (tE != 0.0F)
            atomicAdd(&Ag[j * m + nRow], tE);
    }
}
