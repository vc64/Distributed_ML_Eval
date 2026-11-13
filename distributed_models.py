import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models import LogisticRegression, MLP

def setup_distributed():
    # local_rank = int(os.environ["LOCAL_RANK"])
    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    return local_rank

dist.init_process_group()

local_rank = setup_distributed()
model = LogisticRegression().to(local_rank)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
print(model)