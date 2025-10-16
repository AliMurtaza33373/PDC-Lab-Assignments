#include <stdio.h>
#include <stdlib.h>

// --- Part 2: Kernels ---
// Kernel 1: Computes C[i] = A[i] + B[i]
__global__ void kernel1_add(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// Kernel 2: Computes D[i] = C[i] * C[i]
__global__ void kernel2_square(float* c, float* d, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        d[tid] = c[tid] * c[tid];
    }
}


// --- Part 6: Bonus Challenge (Reduction) ---
// Kernel to sum all elements in an array using shared memory and atomics
__global__ void reductionKernel(float* input, float* output_sum, int n) {
    // Dynamically allocate shared memory. The size is passed during kernel launch.
    extern __shared__ float sdata[];

    // Each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = 0.0f; // Pad with 0 for elements outside the array
    }
    
    // Make sure all threads in the block have finished loading
    __syncthreads();

    // Perform the reduction in shared memory
    // Each iteration, half the threads become inactive
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // Wait for all threads to finish their addition for this stride
        __syncthreads();
    }

    // The first thread of each block adds its partial sum to the global total
    // We use atomicAdd to prevent a race condition between blocks
    if (tid == 0) {
        atomicAdd(output_sum, sdata[0]);
    }
}


int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    // --- Part 1: Memory Allocation ---
    // Allocate arrays on the CPU (host)
    float *h_A, *h_B, *h_C, *h_D, *h_D_streams;
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);
    h_D = (float*)malloc(bytes);        // For serial kernel results
    h_D_streams = (float*)malloc(bytes); // For streamed kernel results

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)i * 2.0f;
    }

    // Allocate memory on the GPU (device)
    float *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_D, bytes);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Set up thread hierarchy for most kernels
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("--- Part 2: Running Kernels Serially on Default Stream ---\n");
    kernel1_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    kernel2_square<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_D, N);
    
    // --- Part 4: Synchronization ---
    // We MUST wait for the GPU to finish before we copy data back.
    // If you comment this out, the CPU might copy the data before it's ready!
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost);
    printf("Serial execution finished. First result D[0] = %f, Last result D[1023] = %f\n\n", h_D[0], h_D[1023]);


    printf("--- Part 3: Running Kernels on Different Streams ---\n");
    // Create two streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Launch kernels on their respective streams
    // These may run at the same time if the GPU has resources! >w<
    kernel1_add<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, N);
    kernel2_square<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_C, d_D, N);
    
    // We still need to synchronize the whole device before accessing the result
    cudaDeviceSynchronize();
    cudaMemcpy(h_D_streams, d_D, bytes, cudaMemcpyDeviceToHost);
    printf("Streamed execution finished. First result D[0] = %f, Last result D[1023] = %f\n\n", h_D_streams[0], h_D_streams[1023]);


    printf("--- Part 5: Comparing Thread Hierarchy ---\n");
    // Launching with 1 block, N threads
    printf("Launching with <<<1, %d>>>...\n", N);
    kernel1_add<<<1, N>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Wait to ensure it finishes before the next launch

    // Launching with N/32 blocks, 32 threads
    int blocks = N / 32;
    int threads = 32;
    printf("Launching with <<<%d, %d>>>...\n", blocks, threads);
    kernel1_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    printf("Both launch configurations completed successfully!\n\n");
    

    printf("--- Part 6: Bonus - Parallel Reduction ---\n");
    float h_final_sum = 0.0f;
    float* d_final_sum;
    cudaMalloc(&d_final_sum, sizeof(float));
    // Important: Initialize the sum on the GPU to zero
    cudaMemset(d_final_sum, 0, sizeof(float));

    // Shared memory size for reduction kernel
    size_t shared_mem_size = threadsPerBlock * sizeof(float);
    
    reductionKernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_D, d_final_sum, N);
    cudaDeviceSynchronize();

    // Copy the final sum back to the host
    cudaMemcpy(&h_final_sum, d_final_sum, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum of all elements in D is: %.0f\n", h_final_sum);


    // --- Cleanup ---
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_D_streams);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_final_sum);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}