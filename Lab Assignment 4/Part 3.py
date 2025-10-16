import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import pandas as pd

# --- Model Definitions ---

class SmallCNN(nn.Module):
    """Small model: 1 Conv layer, 1 FC layer."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, 10) # MNIST is 28x28 -> 14x14 after pooling

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc1(x)
        return x

class MediumCNN(nn.Module):
    """Medium model: 2 Conv layers, 2 FC layers."""
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

class LargeCNN(nn.Module):
    """Large model: 3 Conv layers with BatchNorm, 2 FC layers with Dropout."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 256 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_and_measure(model, device, train_loader, epochs=2):
    """Helper function to train a model and return performance metrics."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
    
    peak_memory = 0
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    
    return total_time, peak_memory

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("This script requires a GPU. Exiting.")
        exit()

    print(f"Using device: {device}")

    # Prepare dataset with a fixed batch size
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # 1. Train models of increasing complexity
    models_to_test = {
        'Small': SmallCNN(),
        'Medium': MediumCNN(),
        'Large': LargeCNN()
    }
    
    results = []
    print("\nStarting training with different model complexities...")
    print("Monitor GPU utilization with 'watch -n 0.5 nvidia-smi'")

    for name, model in models_to_test.items():
        print(f"\n--- Training Model: {name} ---")
        time_taken, mem_used = train_and_measure(model, device, train_loader)
        
        # Count number of parameters
        num_params = sum(p.numel() for p in model.parameters()) / 1e6 # in millions
        
        print(f"Total Time: {time_taken:.2f}s")
        print(f"Peak GPU Memory: {mem_used:.2f} MB")
        print(f"Number of Parameters: {num_params:.2f}M")
        
        results.append({
            'Model Complexity': name,
            'Training Time (s)': time_taken,
            'Peak GPU Memory (MB)': mem_used,
            'Parameters (M)': num_params
        })

    # 3. Record and display results in a table
    df = pd.DataFrame(results)
    print("\n\n--- Performance Comparison ---")
    print(df.to_string(index=False))