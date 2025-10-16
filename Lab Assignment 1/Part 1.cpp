#include <stdio.h>

// __global__ means this function runs on the GPU and can be called from the CPU.
__global__ void helloKernel() {
    // Calculate the unique global ID for each thread
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d\n", threadID);
}

int main() {
    // Launch the kernel on the GPU.
    // <<< GridSize, BlockSize >>>
    // We are launching 2 blocks, and each block has 8 threads.
    helloKernel<<<2, 8>>>();

    // cudaDeviceSynchronize() waits for the GPU to finish its work
    // before the CPU continues. This is important for printf.
    cudaDeviceSynchronize();

    return 0;
}