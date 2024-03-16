import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import time
# Define a simple neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the training function to be run by each process
def train(rank, model, sync_event, device):
    print(f"the sub process run in device: {device}")
    dataset = [(torch.randn(10).to(device), torch.randint(0, 2, (1,)).to(device)) for _ in range(100)]
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for i, (data, labels) in enumerate(data_loader):
        # Simulate some training steps by adding a sleep function
        time.sleep(0.1)
        
        if rank % 2 == 0:  # Even rank processes
            sync_event.wait()  # Wait for the signal to continue
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
        else:  # Odd rank processes
            optimizer.zero_grad()
            outputs = model(data)
            sync_event.set()
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()  # Update the model parameters
        print(f"Rank {rank} has finished iteration {i}, loss: {loss.item()}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"the main process start at device: {device}")
    num_processes = 2
    model = SimpleNet().to(device)
    model.share_memory()  # Allow model parameters to be shared between processes
    mp.set_start_method('spawn')
    # Create a pair of events for synchronization between processes
    sync_event = mp.Event()

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(rank, model, sync_event, device))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
