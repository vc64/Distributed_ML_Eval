import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models import LogisticRegression, MLP
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

os.environ["USE_LIBUV"] = "0"

def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", 29500))
    
    store = dist.TCPStore(
        host_name=master_addr,
        port=master_port,
        world_size=world_size,
        is_master=(rank == 0),
        use_libuv=False,
    )

    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )

def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["USE_LIBUV"] = "0"
    os.environ["GLOO_SOCKET_IFNAME"] = "Loopback Pseudo-Interface 1"

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    print(f"Initialized rank {rank} of {world_size}")

    X = torch.randn(1000, 54)
    y = torch.randint(0, 7, (1000,))
    dataset = TensorDataset(X, y)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, shuffle=False)

    model = LogisticRegression(54, 7)
    # model = MLP(54, 7)
    dist_model = DDP(model)

    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(dist_model.parameters(), lr=0.01)

    for epoch in range(5):
        sampler.set_epoch(epoch)
        for batch_x, batch_y in loader:
            optim.zero_grad()
            outputs = dist_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optim.step()

        if rank == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    # workaround, since torchrun was requiring libuv which doesnt come with windows pytorch
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)