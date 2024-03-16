import torch
import torch.nn as nn
import torch.optim as optim

def print_tensor_memory(tensor, name):
    # Calculate and print memory usage for the given tensor
    print(f"{name}: {tensor.numel() * tensor.element_size()} bytes")

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        print_tensor_memory(x, "Input Tensor")
        x = self.fc1(x)
        print_tensor_memory(x, "After fc1")
        x = torch.relu(x)
        print_tensor_memory(x, "After ReLU")
        x = self.fc2(x)
        print_tensor_memory(x, "Output Tensor")
        return x

def train(model, device):
    dataset = [(torch.randn(10).to(device), torch.randint(0, 2, (1,)).to(device)) for _ in range(10)]
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for data, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels.squeeze())
        print_tensor_memory(loss, "Loss Tensor")
        loss.backward()
        optimizer.step()
        
    if device.type == 'cuda':
        print(torch.cuda.memory_summary(device=device))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)
    train(model, device)
