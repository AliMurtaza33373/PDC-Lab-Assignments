from numba import cuda
import numpy as np
from PIL import Image
import time
import math

# Define the CUDA kernel for image inversion
@cuda.jit
def invert_kernel(input_image, output_image):
    """
    CUDA kernel to perform pixel inversion on a 2D image.
    Each thread handles one pixel.
    """
    # Calculate the global x and y coordinates for the current thread
    y, x = cuda.grid(2)

    # Get the image dimensions from its shape
    height, width = input_image.shape

    # Boundary check: ensure the thread is within the image dimensions
    if y < height and x < width:
        output_image[y, x] = 255 - input_image[y, x]

def main():
    # Load a grayscale image
    try:
        img = Image.open('input.jpg').convert('L') # 'L' mode is for grayscale
        input_array = np.array(img)
        print(f"Image loaded successfully: {input_array.shape[1]}x{input_array.shape[0]} pixels.")
    except FileNotFoundError:
        print("Error: 'input.jpg' not found. Please place it in the same directory.")
        return

    # Create an empty array for the output
    output_array = np.empty_like(input_array)

    # Block configurations to test
    block_configs = [(8, 8), (16, 16), (32, 32)]

    # Copy data to the GPU device once
    d_input = cuda.to_device(input_array)
    d_output = cuda.to_device(output_array)

    
    for block_dim in block_configs:
        # Calculate grid dimensions based on image size and block size
        grid_dim = (
            math.ceil(input_array.shape[0] / block_dim[0]),
            math.ceil(input_array.shape[1] / block_dim[1])
        )

        # Time the kernel execution
        start_time = time.time()
        
        # Launch the kernel
        invert_kernel[grid_dim, block_dim](d_input, d_output)
        
        # Synchronize the GPU to ensure the kernel is finished before stopping the timer
        cuda.synchronize()
        
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000 # in milliseconds
        print(f"Block Size: {block_dim}\t Execution Time: {execution_time:.4f} ms")

    # Copy the result back from GPU to CPU to save it
    final_output = d_output.copy_to_host()
    Image.fromarray(final_output).save('output_gpu.jpg')
    print("Inverted image saved as 'output_gpu.jpg'")

if __name__ == "__main__":
    main()
