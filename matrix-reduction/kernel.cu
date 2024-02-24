#define BLOCK_SIZE 512

__global__ void naiveReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // NAIVE REDUCTION IMPLEMENTATION

    __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    partialSum[t] = (start + t < size) ? in[start + t] : 0.0f;
    partialSum[blockDim.x + t] = (start + blockDim.x + t < size) ? in[start + blockDim.x + t] : 0.0f;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();
        if (t % stride == 0) {
            partialSum[2*t] += partialSum[2*t + stride];
        }
    }

    if (t == 0) {
        out[blockIdx.x] = partialSum[0];
    }
}

__global__ void optimizedReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // OPTIMIZED REDUCTION IMPLEMENTATION

    __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    partialSum[t] = (start + t < size) ? in[start + t] : 0.0f;
    partialSum[blockDim.x + t] = (start + blockDim.x + t < size) ? in[start + blockDim.x + t] : 0.0f;

    for (unsigned int stride = blockDim.x; stride > 0; stride >>= 1)
    {
        __syncthreads();
        if (t < stride) {
            float temp = partialSum[t + stride];
            __syncthreads();  // Ensure all threads have updated partialSum
            partialSum[t] += temp;
        }
    }

    if (t == 0) {
        out[blockIdx.x] = partialSum[0];
    }
}
