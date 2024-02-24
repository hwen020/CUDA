#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {
    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float Cvalue = 0.0;

    // Loop over the tiles of the input matrices
    for (int t = 0; t < (k - 1) / TILE_SIZE + 1; ++t) {
        // Load one tile of A and B into shared memory
        if (row < m && t * TILE_SIZE + threadIdx.x < k) {
            As[threadIdx.y][threadIdx.x] = A[row * k + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < n && t * TILE_SIZE + threadIdx.y < k) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        // Multiply the two tiles together
        for (int i = 0; i < TILE_SIZE; ++i) {
            Cvalue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the final result to C
    if (row < m && col < n) {
        C[row * n + col] = Cvalue;
    }
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C) {
    // Initialize thread block and kernel grid dimensions
    const unsigned int BLOCK_SIZE = TILE_SIZE;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n - 1) / BLOCK_SIZE + 1, (m - 1) / BLOCK_SIZE + 1);

    // Invoke CUDA kernel
    mysgemm<<<gridSize, blockSize>>>(m, n, k, A, B, C);

    // Check for kernel launch errors
    cudaError_t cudaKernelError = cudaGetLastError();
    if (cudaKernelError != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(cudaKernelError));
        exit(1);
    }

    // Synchronize the device after kernel launch
    cudaError_t cudaSyncError = cudaDeviceSynchronize();
    if (cudaSyncError != cudaSuccess) {
        fprintf(stderr, "CUDA synchronization error: %s\n", cudaGetErrorString(cudaSyncError));
        exit(1);
    }
}
