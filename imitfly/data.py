"""Strict episode-aligned NPZ dataset and migration of the original text logs."""
from bisect import bisect_right
from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from .contracts import SENSOR_DIMS
from .preprocessing import image_tensor


from .schema import validate_episode


class EpisodeDataset(Dataset):
    """Windows never cross episode boundaries; split by file, not by frame."""
    def __init__(self, paths, sequence_length=5):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = sequence_length
        self.episodes, self.ends = [], []
        for path in paths:
            with np.load(path, allow_pickle=False) as f:
                arrays = {k: f[k] for k in f.files}
            n = validate_episode(arrays)
            if n < sequence_length:
                raise ValueError(f"{path} has fewer frames than the requested window")
            self.episodes.append(arrays)
            self.ends.append((self.ends[-1] if self.ends else 0) + n-sequence_length+1)
        if not self.ends:
            raise ValueError("No episode files selected")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ep = bisect_right(self.ends, index)
        start = index - (self.ends[ep-1] if ep else 0)
        stop = start + self.sequence_length
        d = self.episodes[ep]
        images = [torch.stack([image_tensor(a) for a in d[k][start:stop]])
                  for k in ("left_eye", "right_eye")]
        sensors = [torch.as_tensor(d[k][start:stop], dtype=torch.float32) for k in SENSOR_DIMS]
        return (*images, *sensors,
                torch.as_tensor(d["expert_mean"][stop-1], dtype=torch.float32),
                torch.as_tensor(d["expert_std"][stop-1], dtype=torch.float32))


def _vector(text, dim, label):
    parts = [x for x in re.split(r"[,\s]+", text.strip()) if x]
    try:
        a = np.asarray([float(x) for x in parts], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"Invalid numbers in {label}") from exc
    if a.shape != (dim,) or not np.isfinite(a).all():
        raise ValueError(f"Invalid {label}: expected {dim} finite values")
    return a


def _image_map(directory):
    result = {}
    for p in Path(directory).glob("*.png"):
        match = re.search(r"(\d+)(?:_[A-Za-z]+)?$", p.stem)
        if not match:
            raise ValueError(f"Cannot identify frame number: {p.name}")
        key = int(match[1])
        if key in result:
            raise ValueError(f"Duplicate image frame {key}")
        result[key] = p
    return result


def convert_legacy_episode(folder, destination):
    folder, destination = Path(folder), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    raw = (folder / "inputs_log.txt").read_text()
    parts = re.split(r"Record\s+(\d+)\s*:\s*Policy Input\s*:", raw)
    inputs = {}
    for i in range(1, len(parts), 2):
        idx, block = int(parts[i]), parts[i+1]
        if idx in inputs:
            raise ValueError(f"Duplicate input record {idx}")
        row = {}
        for name, dim in SENSOR_DIMS.items():
            match = re.search(r"walker/" + name + r"\s*:\s*\[([^\]]*)\]", block)
            if not match:
                raise ValueError(f"Missing {name} at record {idx}")
            row[name] = _vector(match[1], dim, name)
        inputs[idx] = row
    raw = (folder / "outputs_log.txt").read_text()
    outputs = {}
    for idx, label, values in re.findall(r"Record\s+(\d+)\s*:\s*(Mean|Stddev)\s*:\s*\[([^\]]*)\]", raw):
        row = outputs.setdefault(int(idx), {})
        if label in row:
            raise ValueError(f"Duplicate output {idx}/{label}")
        row[label] = _vector(values, 12, label)
    left, right = _image_map(folder/"left_eye"), _image_map(folder/"right_eye")
    ids = sorted(inputs)
    if not ids or not (set(ids) == set(outputs) == set(left) == set(right)):
        raise ValueError("Input/output/left/right record IDs differ; refusing silent truncation")
    if ids != list(range(ids[0], ids[-1]+1)):
        raise ValueError("Non-contiguous records; refusing to create temporal windows across gaps")
    arrays = {k: np.stack([inputs[i][k] for i in ids]) for k in SENSOR_DIMS}
    for key, mapping in (("left_eye", left), ("right_eye", right)):
        images=[]
        for i in ids:
            with Image.open(mapping[i]) as image:
                image = image.convert("L").resize((32,32), Image.Resampling.BILINEAR)
                images.append(np.asarray(image, dtype=np.uint8))
        arrays[key] = np.stack(images)
    try:
        arrays["expert_mean"] = np.stack([outputs[i]["Mean"] for i in ids])
        arrays["expert_std"] = np.stack([outputs[i]["Stddev"] for i in ids])
    except KeyError as exc:
        raise ValueError("Each output record needs Mean and Stddev") from exc
    arrays["record_id"] = np.asarray(ids, dtype=np.int64)
    validate_episode(arrays)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as f:
        np.savez_compressed(f, **arrays)
    return len(ids)


class WalkingEpisodeDataset(Dataset):
    """Sensor-only temporal windows for walking distribution distillation."""
    def __init__(self, paths, sequence_length=5):
        from .schema import validate_walking_episode
        self.sequence_length = sequence_length
        self.episodes, self.ends = [], []
        for path in paths:
            with np.load(path, allow_pickle=False) as f:
                d = {k:f[k] for k in f.files}
            n = validate_walking_episode(d)
            if n < sequence_length:
                raise ValueError(f"{path} is shorter than the sequence length")
            self.episodes.append(d)
            self.ends.append((self.ends[-1] if self.ends else 0)+n-sequence_length+1)
        if not self.ends:
            raise ValueError("No walking episodes selected")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        from .contracts import WALKING_SENSOR_DIMS
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ep = bisect_right(self.ends, index)
        start = index-(self.ends[ep-1] if ep else 0)
        stop = start+self.sequence_length
        d = self.episodes[ep]
        sensors = {k:torch.as_tensor(d[k][start:stop],dtype=torch.float32)
                   for k in WALKING_SENSOR_DIMS}
        return (sensors, torch.as_tensor(d["expert_mean"][stop-1],dtype=torch.float32),
                torch.as_tensor(d["expert_std"][stop-1],dtype=torch.float32))
