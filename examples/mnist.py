import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    """A small fully-connected MNIST classifier.

    ``hidden`` sets the width of the single hidden layer (128 in the training
    example below); examples/benchmark.py sweeps it to scale the model up.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torchvulkan as torchvk

    batch_size = 32
    learning_rate = 0.001
    epochs = 3
    device = torch.device('vulkan' if torchvk.is_available() else 'cpu')

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"\repoch [{epoch+1}/{epochs}] | batch {batch_idx+1}/{len(train_loader)} | loss: {loss.item():.4f}", end='')

    print("Training complete!")

if __name__ == "__main__":
    main()