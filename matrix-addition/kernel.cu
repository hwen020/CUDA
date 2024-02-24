#include <stdio.h>

__global__ void matAdd(int dim, const float* A, const float* B, float* C) {
    // Calculate the global indices for the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within the matrix dimensions
    if (row < dim && col < dim) {
        int idx = row * dim + col;
        C[idx] = A[idx] + B[idx];
    }
}

void basicMatAdd(int dim, const float* A, const float* B, float* C) {
    // Define block and grid dimensions
    dim3 blockDim(16, 16);  // 16x16 thread block
    dim3 gridDim((dim + 15) / 16, (dim + 15) / 16);  // Calculate grid dimensions

    // Launch the CUDA kernel
    matAdd<<<gridDim, blockDim>>>(dim, A, B, C);

    // Ensure all GPU operations are completed
    cudaDeviceSynchronize();
}
