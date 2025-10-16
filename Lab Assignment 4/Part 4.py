import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import pandas as pd

# --- Model and Helper Functions ---

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

def train_with_workers(model, device, train_loader, epochs=2):
    """Trains a model for a fixed number of epochs and returns total time."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    return total_time

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("This script requires a GPU to demonstrate data loading bottlenecks. Exiting.")
        exit()

    print(f"Using device: {device}")

    # Prepare dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 2. Compare performance for different num_workers
    num_workers_list = [0, 2, 4, 8]
    results = []

    print("\nStarting training with different numbers of data loader workers...")
    print("Monitor GPU utilization with 'watch -n 0.5 nvidia-smi'")

    for workers in num_workers_list:
        # For num_workers > 0, pin_memory=True can speed up CPU-to-GPU transfers
        use_pin_memory = workers > 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=256,
            shuffle=True,
            num_workers=workers,
            pin_memory=use_pin_memory
        )
        
        print(f"\n--- Training with num_workers = {workers} ---")
        model = MediumCNN()
        time_taken = train_with_workers(model, device, train_loader)
        print(f"Total Time: {time_taken:.2f}s")
        results.append({'num_workers': workers, 'time': time_taken})

    # Convert results to a DataFrame and display
    df = pd.DataFrame(results)
    print("\n\n--- Performance Comparison ---")
    print(df.to_string(index=False))

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(df['num_workers'], df['time'], marker='o')
    plt.title('Effect of `num_workers` on Training Time')
    plt.xlabel('Number of Worker Processes')
    plt.ylabel('Total Training Time (s)')
    plt.xticks(num_workers_list)
    plt.grid(True)
    plt.savefig('part4_num_workers_effect.png')
    
    print("\nPlot saved as 'part4_num_workers_effect.png'")