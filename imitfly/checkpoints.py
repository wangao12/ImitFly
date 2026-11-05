"""Strict loading plus durable, recoverable training snapshots."""
from pathlib import Path
import os
import random
import shutil
import numpy as np
import torch
from .models import make_policy

def load_policy(path, device="cpu", model_name="imitfly"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})
    name = config.get("model", model_name)
    sequence_length = config.get("sequence_length", 5)
    model = make_policy(name, sequence_length).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, {**config, "model": name, "sequence_length": sequence_length}

def rng_state(generator):
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy_name": n[0],
            "numpy_keys": torch.from_numpy(n[1].astype(np.int64)),
            "numpy_pos": n[2], "numpy_gauss": n[3], "numpy_cached": n[4],
            "torch": torch.get_rng_state(), "generator": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state((state["numpy_name"], state["numpy_keys"].numpy().astype(np.uint32),
                         state["numpy_pos"],state["numpy_gauss"],state["numpy_cached"]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["generator"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

def atomic_save(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    previous = path.with_suffix(path.suffix + ".previous")
    if path.exists():
        backup_tmp = previous.with_suffix(previous.suffix + ".tmp")
        shutil.copyfile(path, backup_tmp)
        with backup_tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(backup_tmp, previous)
    with temporary.open("wb") as f:
        torch.save(checkpoint, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
