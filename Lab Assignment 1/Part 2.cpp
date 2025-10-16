#include <stdio.h>
#include <stdlib.h>
#include <time.h> // For CPU timing

// CUDA Kernel for vector addition
__global__ void addVectors(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 10000000; // 10 million elements
    size_t size = N * sizeof(float);

    // --- Host (CPU) memory allocation ---
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c_cpu = (float*)malloc(size); // CPU result
    float *h_c_gpu = (float*)malloc(size); // GPU result

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)i * 2.0f;
    }

    // --- CPU Calculation & Timing ---
    clock_t start_cpu = clock();
    for (int i = 0; i < N; i++) {
        h_c_cpu[i] = h_a[i] + h_b[i];
    }
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("CPU Execution Time: %f seconds\n", cpu_time);


    // --- GPU Calculation & Timing ---
    // Device (GPU) memory allocation
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Set up thread configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Use CUDA events for accurate GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    // Launch the kernel
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop_gpu);

    // Wait for the GPU to finish
    cudaEventSynchronize(stop_gpu);

    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);
    double gpu_time = gpu_time_ms / 1000.0;
    printf("GPU Execution Time: %f seconds\n", gpu_time);

    // Copy result back from device (GPU) to host (CPU)
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    // --- Final Results ---
    printf("\nSpeedup (CPU Time / GPU Time): %.2fx\n", cpu_time / gpu_time);

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}