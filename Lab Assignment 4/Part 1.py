import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os

def get_device():
    """Checks for and returns the available computing device."""
    if torch.cuda.is_available():
        print("CUDA (GPU) is available. Using GPU.")
        return torch.device("cuda")
    else:
        print("CUDA (GPU) not available. Using CPU.")
        return torch.device("cpu")

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
        # Conv layer 1
        x = self.pool(self.relu(self.conv1(x)))
        # Conv layer 2
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten the tensor
        x = x.view(-1, 64 * 7 * 7)
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, device, train_loader, epochs=3):
    """Trains the model on the specified device and measures performance."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    total_start_time = time.time()
    
    print(f"\n--- Training on {device.type.upper()} for {epochs} epochs ---")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        for data, target in train_loader:
            # Move data to the selected device
            data, target = data.to(device), target.to(device)
            
            # Forward pass -> backward pass -> optimizer step
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} - Training time: {epoch_time:.2f}s")

    total_time = time.time() - total_start_time
    
    print(f"Total Training Time on {device.type.upper()}: {total_time:.2f}s")
    return total_time

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Choose and prepare the dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 3. Train the model on CPU
    cpu_device = torch.device("cpu")
    cpu_model = SimpleCNN()
    cpu_time = train_model(cpu_model, cpu_device, train_loader)

    # 3. Train the model on GPU
    gpu_device = get_device()
    gpu_time = -1 # Default value if no GPU
    
    if gpu_device.type == 'cuda':
        print("\nTo monitor GPU utilization, open a new terminal and run:")
        print("watch -n 0.5 nvidia-smi")
        input("Press Enter to start GPU training...")
        
        gpu_model = SimpleCNN() # Create a new model instance for GPU
        gpu_time = train_model(gpu_model, gpu_device, train_loader)

        # 5. Compute and display speedup
        print("\n--- Results ---")
        speedup = cpu_time / gpu_time
        print(f"CPU Total Time: {cpu_time:.2f}s")
        print(f"GPU Total Time: {gpu_time:.2f}s")
        print(f"Speedup Ratio (CPU/GPU): {speedup:.2f}x")
    else:
        print("\nSkipping speedup calculation as no GPU was found.")