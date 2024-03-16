import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.distributed as dist
import torch.multiprocessing as mp

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def train(rank, world_size):
    # Initialize the distributed environment.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print(f"rank: {rank}, world_size: {world_size}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Create the model and move it to the GPU with id "rank".
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Define a loss function and optimizer.
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Dummy dataset: input and target.
    inputs = torch.randn(20, 10).to(rank)
    targets = torch.randn(20, 5).to(rank)

    # Training loop.
    for epoch in range(2):  # loop over the dataset multiple times
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Cleanup.
    dist.destroy_process_group()

def main():
    world_size = 4  # Number of GPUs you want to use.
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
