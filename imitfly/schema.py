"""Array schema shared by teacher and student without importing ML runtimes."""
import numpy as np
from .contracts import SENSOR_DIMS

def validate_episode(data):
    required = {"left_eye", "right_eye", "expert_mean", "expert_std", *SENSOR_DIMS}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Missing episode arrays: {sorted(missing)}")
    n = len(data["expert_mean"])
    if n == 0:
        raise ValueError("Empty episode")
    for k in required:
        a = np.asarray(data[k])
        shape = (n, 32, 32) if k.endswith("eye") else (n, SENSOR_DIMS.get(k, 12))
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"Invalid {k}: {a.shape}; expected {shape} and finite values")
        if k.endswith("eye") and a.dtype != np.uint8:
            raise ValueError("Eye arrays must be uint8 [N,32,32]")
    if (data["expert_std"] <= 0).any():
        raise ValueError("Expert std must be positive")
    return n



def validate_walking_episode(data):
    from .contracts import WALKING_SENSOR_DIMS
    n = len(data["expert_mean"])
    if n == 0:
        raise ValueError("Empty walking episode")
    for name, dim in {**WALKING_SENSOR_DIMS,"expert_mean":59,"expert_std":59}.items():
        a = np.asarray(data[name])
        if a.shape != (n,dim) or not np.isfinite(a).all():
            raise ValueError(f"Invalid walking array {name}: expected {(n,dim)}")
    if (data["expert_std"] <= 0).any():
        raise ValueError("Expert std must be positive")
    return n
