import os
import torch
from typing import Optional

def get_device() -> torch.device:
    # Respect CUDA_VISIBLE_DEVICES set by Slurm/cluster
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def set_seed(seed: Optional[int]):
    if seed is None:
        return
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
