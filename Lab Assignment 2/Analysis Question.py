# Get the current GPU device
device = cuda.get_current_device()

# A common, safe, and efficient 2D block size is (16, 16) or (32, 32)
# We can choose one based on device capability, but (16,16) is a very safe default.
# It gives 256 threads, which is a multiple of 32 and works well on most GPUs.
block_dim = (16, 16) 

# Check if we can use a larger, potentially faster block size
# Max threads per block is a key hardware limit
if device.MAX_THREADS_PER_BLOCK >= 1024:
    block_dim = (32, 32) # 1024 threads, often faster if kernel is simple

# Calculate grid dimensions dynamically
grid_dim = (
    math.ceil(input_array.shape[0] / block_dim[0]),
    math.ceil(input_array.shape[1] / block_dim[1])
)

print(f"Using generic block size: {block_dim} for this device.")
# Launch kernel
# invert_kernel[grid_dim, block_dim](d_input, d_output)