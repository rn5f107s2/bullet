#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

struct Feat {
    int32_t our;
    int32_t opp;
};

constexpr int N  = 8;
constexpr int HL = 64 * N; 

__global__ void SingleSparseAffineForwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const Feat* inputs,
    float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const size_t inputIdx = inputSize * blockIdx.y;
    const Feat* thisInput = inputs + inputSize * blockIdx.y;
    float* thisOutput = outputs + outputSize * blockIdx.y + elem;

    float ourElementVal = biases[elem];

    for (size_t i = 0; i < inputSize; i++) {
        const Feat inp = thisInput[i];

        if (inp.our == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp.our) * outputSize + elem;
        ourElementVal += weights[ourIdx];
    }

    thisOutput[0] = ourElementVal;
}

__global__ void SingleSparseAffineBackwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const Feat* inputs,
    const float* errors,
    const float* output,
    const float ftRegularisation)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;

    const Feat* thisInput = inputs + inputSize * blockIdx.y;
    const float* thisErrors = errors + outputSize * blockIdx.y;
    const float* thisOutput = output + 2 * outputSize * blockIdx.y;

    float ourError = thisErrors[elem];

    // Idea from Jay (Beserk author).
    if (ftRegularisation != 0.0F)
    {
            const float* thisOutput = output + 2 * outputSize * blockIdx.y;
            ourError += ftRegularisation * (thisOutput[elem] > 0.0F);
    }

    atomicAdd(&biasesGrad[elem], ourError);

    for (size_t i = 0; i < inputSize; i++) {
        const Feat inp = thisInput[i];

        if (inp.our == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp.our) * outputSize + elem;
        atomicAdd(&weightsGrad[ourIdx], ourError);
    }
}

__global__ void sparseAffineForwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const Feat* inputs,
    float* outputs)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= outputSize)
        return;
    
    *(outputs + 2 * outputSize * blockIdx.y + elem             ) = 0;
    *(outputs + 2 * outputSize * blockIdx.y + elem + outputSize) = 0;

    if (elem >= inputSize * N)
        return;

    int index = elem / N;

    const size_t inputIdx = inputSize * blockIdx.y;
    const Feat* thisInput = inputs + inputSize * blockIdx.y;

    const Feat feat = thisInput[index];

    const int featSqOur = feat.our % 64;
    const int featSqOpp = feat.opp % 64;
    const int featPcOur = feat.our / 64;
    const int featPcOpp = feat.opp / 64;

    const int ourIndex = featPcOur * HL + featSqOur * N + elem % N;
    const int oppIndex = featPcOpp * HL + featSqOpp * N + elem % N;

    float ourElementVal = 0;
    float oppElementVal = 0;

    float* ourOutput = outputs + 2 * outputSize * blockIdx.y + ourIndex;
    float* oppOutput = outputs + 2 * outputSize * blockIdx.y + oppIndex + outputSize;

    if (feat.our == -1) {
        *ourOutput = *oppOutput = 0;
        return;
    }

    for (size_t i = 0; i < inputSize; i++) {
        const Feat inp = thisInput[i];

        if (inp.our == -1)
            break;

        // idx * L1_SIZE * 12 + bucketSq * 4 + L1_SIZE * bucketPc;

        const size_t ourIdx = static_cast<size_t>(inp.our) * HL * 12 + ourIndex;
        const size_t oppIdx = static_cast<size_t>(inp.opp) * HL * 12 + oppIndex;

        ourElementVal += weights[ourIdx];
        oppElementVal += weights[oppIdx];
    }

    // if (foundOur)
    //     if (foundOpp) printf("Element: %d our: 1 opp: 1\n", elem);
    //     else printf("Element: %d our: 1 opp: 0\n", elem);
    // else
    //     if (foundOpp) printf("Element: %d our: 0 opp: 1\n", elem);
    //     else printf("Element: %d our: 0 opp: 0\n", elem);

    *ourOutput = ourElementVal;
    *oppOutput = oppElementVal;
}

__global__ void sparseAffineBackwardKernel(
    const size_t inputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const Feat* inputs,
    const float* errors,
    const float* output,
    const float ftRegularisation)
{
    const size_t elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= inputSize * N)
        return;

    const Feat* thisInput = inputs + inputSize * blockIdx.y;
    const float* thisErrors = errors + 2 * outputSize * blockIdx.y;

    int index = elem / N;

    const Feat feat = thisInput[index];

    const int featSqOur = feat.our % 64;
    const int featSqOpp = feat.opp % 64;
    const int featPcOur = feat.our / 64;
    const int featPcOpp = feat.opp / 64;

    const int ourIndex = featPcOur * HL + featSqOur * N + elem % N;
    const int oppIndex = featPcOpp * HL + featSqOpp * N + elem % N;

    float ourError = *(thisErrors + ourIndex);
    float oppError = *(thisErrors + oppIndex + outputSize);

    // Idea from Jay (Beserk author).
    if (ftRegularisation != 0.0F)
    {
            const float* thisOutput = output + 2 * outputSize * blockIdx.y;
            ourError += ftRegularisation * (thisOutput[elem] > 0.0F);
            oppError += ftRegularisation * (thisOutput[elem + outputSize] > 0.0F);
    }

    for (size_t i = 0; i < inputSize; i++) {
        const Feat inp = thisInput[i];

        if (inp.our == -1)
            break;

        const size_t ourIdx = static_cast<size_t>(inp.our) * HL * 12 + ourIndex;
        const size_t oppIdx = static_cast<size_t>(inp.opp) * HL * 12 + oppIndex;
        atomicAdd(&weightsGrad[ourIdx], ourError);
        atomicAdd(&weightsGrad[oppIdx], oppError);
    }
}

extern "C" void singleSparseAffineForward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const Feat* inputs,
    float* outputs)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    SingleSparseAffineForwardKernel<<<grid, threads>>>(
        maxInputSize,
        outputSize,
        weights,
        biases,
        inputs,
        outputs
    );
}

extern "C" void singleSparseAffineBackward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const Feat* inputs,
    const float* errors,
    const float* output,
    const float ftRegularisation)
{
    const size_t numChunks = (maxInputSize * N + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? maxInputSize * N : 1024;

    SingleSparseAffineBackwardKernel<<<grid, threads>>>(
        maxInputSize,
        outputSize,
        weightsGrad,
        biasesGrad,
        inputs,
        errors,
        output,
        ftRegularisation
    );
}

extern "C" void sparseAffineForward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    const float* weights,
    const float* biases,
    const Feat* inputs,
    float* outputs)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    sparseAffineForwardKernel<<<grid, threads>>>(
        maxInputSize,
        outputSize,
        weights,
        biases,
        inputs,
        outputs
    );
}

extern "C" void sparseAffineBackward(
    const size_t batchSize,
    const size_t maxInputSize,
    const size_t outputSize,
    float* weightsGrad,
    float* biasesGrad,
    const Feat* inputs,
    const float* errors,
    const float* output,
    const float ftRegularisation)
{
    const size_t numChunks = (outputSize + static_cast<size_t>(1023)) / static_cast<size_t>(1024);

    dim3 grid(numChunks, batchSize);

    const size_t threads = (numChunks == 1) ? outputSize : 1024;

    sparseAffineBackwardKernel<<<grid, threads>>>(
        maxInputSize,
        outputSize,
        weightsGrad,
        biasesGrad,
        inputs,
        errors,
        output,
        ftRegularisation
    );
}
