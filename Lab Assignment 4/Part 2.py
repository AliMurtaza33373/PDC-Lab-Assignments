import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import pandas as pd

# --- Model and Helper Functions ---

class SimpleCNN(nn.Module):
    """A simple Convolutional Neural Network for MNIST."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_for_batch_size(model, device, train_loader, epochs=2):
    """Trains a model and returns total time and peak memory usage."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Reset CUDA memory stats to get accurate peak usage for this run
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    total_start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    total_time = time.time() - total_start_time
    
    # Get peak memory allocated in Megabytes (MB)
    peak_memory = 0
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    
    return total_time, peak_memory

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("This script requires a GPU to measure memory usage. Exiting.")
        exit()

    print(f"Using device: {device}")

    # Prepare dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 1. Train with different batch sizes
    batch_sizes = [16, 64, 256, 1024]
    results = []

    print("\nStarting training with different batch sizes...")
    for bs in batch_sizes:
        print(f"--- Training with Batch Size: {bs} ---")
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        model = SimpleCNN()
        time_taken, mem_used = train_for_batch_size(model, device, train_loader)
        
        print(f"Total Time: {time_taken:.2f}s")
        print(f"Peak GPU Memory: {mem_used:.2f} MB")
        results.append({'batch_size': bs, 'time': time_taken, 'memory': mem_used})

    # Convert results to a pandas DataFrame for easy plotting
    df = pd.DataFrame(results)

    # 3. Plot the results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Total Training Time (s)', color=color)
    ax1.plot(df['batch_size'], df['time'], color=color, marker='o', label='Training Time')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create a second y-axis for memory usage
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Peak GPU Memory (MB)', color=color)
    ax2.plot(df['batch_size'], df['memory'], color=color, marker='x', linestyle='--', label='GPU Memory')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Effect of Batch Size on Training Time and GPU Memory')
    plt.savefig('part2_batch_size_effects.png')
    
    print("\nPlot saved as 'part2_batch_size_effects.png'")