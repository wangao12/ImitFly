"""Offline distribution distillation; no environment or teacher in the training loop."""
import argparse
import json
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from imitfly.data import EpisodeDataset, WalkingEpisodeDataset
from imitfly.models import make_policy
from imitfly.losses import gaussian_kl_loss, gaussian_nll_loss
from imitfly.checkpoints import atomic_save, rng_state, restore_rng


def prepare_batch(batch, device):
    if isinstance(batch[0], dict):
        inputs = {k:v.to(device) for k,v in batch[0].items()}
    else:
        inputs = [v.to(device) for v in batch[:-2]]
    return inputs, batch[-2].to(device), batch[-1].to(device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]/"configs/imitfly_12_demos.json")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    a=p.parse_args()
    c=json.loads(a.config.read_text())
    loss_functions = {"kl": gaussian_kl_loss, "nll": gaussian_nll_loss}
    criterion = loss_functions[c.get("loss", "kl")]
    if a.output.exists() and not a.resume:
        raise FileExistsError("Use a new output directory or explicitly --resume")
    device="cuda" if a.device=="auto" and torch.cuda.is_available() else a.device
    if device=="auto": device="cpu"
    if device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    print(f"device={device}")
    seed=c.get("seed",0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(c.get("cpu_threads", 2))
    generator=torch.Generator().manual_seed(seed)
    files=[a.data / name for name in c["train_episodes"]]
    valid_files=[a.data / name for name in c.get("validation_episodes",[])]
    if set(f.resolve() for f in files) & set(f.resolve() for f in valid_files):
        raise ValueError("Training and validation episodes overlap")
    dataset_type = WalkingEpisodeDataset if c["model"] == "walking" else EpisodeDataset
    dataset=dataset_type(files,c["sequence_length"])
    loader=DataLoader(dataset,batch_size=c["batch_size"],shuffle=True,
                      num_workers=0,generator=generator)
    validation=DataLoader(dataset_type(valid_files,c["sequence_length"]),
                          batch_size=c["batch_size"]) if valid_files else None
    a.output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.output).free < 512*1024**2:
        raise OSError("Less than 512 MiB free; refusing training")
    # Record exact data bytes so a resumed run cannot silently change its dataset.
    import hashlib
    data_hashes={str(f.relative_to(a.data)):hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in files+valid_files}
    model=make_policy(c["model"],c["sequence_length"]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=c["learning_rate"])
    start=0
    if a.resume:
        ck=torch.load(a.resume,map_location="cpu",weights_only=True)
        if ck["config"] != c or ck["data_sha256"] != data_hashes:
            raise ValueError("Resume requires matching config and dataset hashes")
        model.load_state_dict(ck["model_state_dict"],strict=True)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        restore_rng(ck["rng"],generator)
        start=ck["epoch"]
    (a.output/"config.json").write_text(json.dumps(c,indent=2)+"\n")
    log=a.output/"metrics.jsonl"
    for epoch in range(start,c["epochs"]):
        model.train(); total=0.; count=0
        for batch in loader:
            inputs, target_mean, target_std = prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            mu, log_std=model(inputs)
            loss=criterion(mu,log_std,target_mean,target_std)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss")
            loss.backward(); optimizer.step()
            total+=loss.item()*len(mu); count+=len(mu)
        row={"epoch":epoch+1,"loss":c.get("loss","kl"),"train_loss":total/count}
        if validation:
            model.eval(); val_total=0.; val_count=0
            with torch.no_grad():
                for batch in validation:
                    inputs, target_mean, target_std = prepare_batch(batch, device)
                    mu, ls=model(inputs)
                    loss=criterion(mu,ls,target_mean,target_std)
                    if not torch.isfinite(loss): raise FloatingPointError("Nonfinite validation loss")
                    val_total+=loss.item()*len(mu); val_count+=len(mu)
            row["validation_loss"]=val_total/val_count
        atomic_save({"model_state_dict":model.state_dict(),"config":c,
                     "optimizer_state_dict":optimizer.state_dict(),"epoch":epoch+1,
                     "rng":rng_state(generator),"data_sha256":data_hashes}, a.output/"latest.pt")
        with log.open("a") as f:
            f.write(json.dumps(row)+"\n"); f.flush()
        print(json.dumps(row),flush=True)

if __name__ == "__main__":
    main()
