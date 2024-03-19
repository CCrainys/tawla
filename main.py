import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import argparse
import torch.multiprocessing as mp

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

def setup(masterip, masterport, rank, world_size):
    init_method = f'tcp://{masterip}:{masterport}'
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)

def create_independent_groups(world_size):
    all_ranks = range(world_size)

    group1 = dist.new_group(ranks=all_ranks)
    group2 = dist.new_group(ranks=all_ranks)
    
    return group1, group2

def cleanup():
    dist.destroy_process_group()

# The function to be run in each process
def train(rank, prank, model, world_size, sync_event):
    if prank % 2 == 0:
        setup("localhost", 12355, rank, world_size)
    else:
        setup("localhost", 12356, rank, world_size)
    #group1, group2 = create_independent_groups(world_size)
    dataset = [(torch.randn(10).to(rank), torch.randn(5).to(rank)) for _ in range(100)]
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for i, (data, labels) in enumerate(data_loader):
        if prank % 2 == 0:  # Even rank processes
            sync_event.wait()  # Wait for the signal to continue
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    #print(f"rank: {rank}, before grad param: {param.grad}")
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    #print(f"rank: {rank}, after grad param: {param.grad}")
                    param.grad.data /= world_size
            optimizer.step()
        else:  # Odd rank processes
            optimizer.zero_grad()
            outputs = model(data)
            sync_event.set()
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    #print(f"rank: {rank}, before grad param: {param.grad}")
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    #print(f"rank: {rank}, after grad param: {param.grad}")
                    param.grad.data /= world_size
            optimizer.step()  # Update the model parameters
        print(f"Rank-prank: {rank}-{prank} has finished iteration {i}, loss: {loss.item()}")
    
    cleanup()

def ddp_basic(rank, world_size):
    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)

    model.share_memory()
    mp.set_start_method('spawn')
    sync_event = mp.Event()

    processes = []
    num_processes = 2
    for prank in range(num_processes):
        p = mp.Process(target=train, args=(rank, prank, model, world_size, sync_event))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed training example.')
    parser.add_argument('--rank', type=int, required=True, help='Rank of the process')
    parser.add_argument('--world_size', type=int, required=True, help='Number of processes in the distributed setup')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    ddp_basic(args.rank, args.world_size)
