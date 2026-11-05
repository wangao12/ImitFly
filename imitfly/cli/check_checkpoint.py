"""Validate a separately supplied v51 checkpoint without running physics."""
import argparse
import hashlib
from pathlib import Path
import torch
from imitfly.checkpoints import load_policy
from imitfly.contracts import SENSOR_DIMS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint",type=Path)
    a=p.parse_args()
    torch.set_num_threads(2)
    model,c=load_policy(a.checkpoint)
    t=c["sequence_length"]
    x=[torch.zeros(1,t,1,32,32) for _ in range(2)]
    x += [torch.zeros(1,t,d) for d in SENSOR_DIMS.values()]
    with torch.no_grad():
        mu,ls=model(x)
    if mu.shape != (1,12) or not torch.isfinite(mu).all() or not torch.isfinite(ls).all():
        raise ValueError("Invalid policy output")
    print({"sha256":hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
           "parameters":sum(p.numel() for p in model.parameters()),"config":c})

if __name__ == "__main__":
    main()
