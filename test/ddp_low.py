import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import argparse

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

class ToyDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, index):
        return torch.randn(10), torch.randn(5)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataset = ToyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for epoch in range(10):
        sampler.set_epoch(epoch)
        for data, target in dataloader:
            optimizer.zero_grad()
            output = ddp_model(data.to(rank))
            loss = criterion(output, target.to(rank))
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    #print(f"rank: {rank}, before grad param: {param.grad}")
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    #print(f"rank: {rank}, after grad param: {param.grad}")
                    param.grad.data /= world_size
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed training example.')
    parser.add_argument('--rank', type=int, required=True, help='Rank of the process')
    parser.add_argument('--world_size', type=int, required=True, help='Number of processes in the distributed setup')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    demo_basic(args.rank, args.world_size)
