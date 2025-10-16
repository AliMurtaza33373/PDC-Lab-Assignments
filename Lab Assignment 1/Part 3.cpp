#include <stdio.h>
#include <time.h>

// These lines are required to use the stb_image library
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// GPU kernel to invert pixel values
__global__ void invertKernel(unsigned char* input_image, unsigned char* output_image, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output_image[i] = 255 - input_image[i];
    }
}

int main() {
    int width, height, channels;
    const char* input_filename = "input.png";

    // Load image using stb_image
    unsigned char* h_image = stbi_load(input_filename, &width, &height, &channels, 0);
    if (h_image == NULL) {
        printf("Error: Could not load image '%s'\n", input_filename);
        return 1;
    }
    printf("Image loaded: %d x %d with %d channels\n", width, height, channels);

    int image_size = width * height * channels;
    size_t bytes = image_size * sizeof(unsigned char);

    // Allocate memory for output images
    unsigned char* h_output_cpu = (unsigned char*)malloc(bytes);
    unsigned char* h_output_gpu = (unsigned char*)malloc(bytes);

    // --- CPU Inversion ---
    clock_t start_cpu = clock();
    for (int i = 0; i < image_size; i++) {
        h_output_cpu[i] = 255 - h_image[i];
    }
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("CPU Inversion Time: %f seconds\n", cpu_time);
    stbi_write_png("output_cpu.png", width, height, channels, h_output_cpu, width * channels);


    // --- GPU Inversion ---
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_image, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (image_size + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    invertKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, image_size);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);
    double gpu_time = gpu_time_ms / 1000.0;
    printf("GPU Inversion Time: %f seconds\n", gpu_time);

    cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost);
    stbi_write_png("output_gpu.png", width, height, channels, h_output_gpu, width * channels);
    
    printf("\nSpeedup: %.2fx\n", cpu_time / gpu_time);
    printf("CPU and GPU output images saved as 'output_cpu.png' and 'output_gpu.png'.\n");


    // Cleanup
    stbi_image_free(h_image);
    free(h_output_cpu);
    free(h_output_gpu);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}