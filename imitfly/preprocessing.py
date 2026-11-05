"""Shared training image transform; raw renderer conversion matches legacy inference."""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from .contracts import SENSOR_DIMS

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)), transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

def image_tensor(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return IMAGE_TRANSFORM(image.convert("L"))

def observation_tensors(observation):
    def eye(name):
        a = np.squeeze(np.asarray(observation[f"walker/{name}"]))
        if np.issubdtype(a.dtype, np.floating):
            if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
                raise ValueError("Floating eye observations must be in [0,1]")
            a = (a * 255).astype(np.uint8)
        if a.ndim == 3:
            import cv2  # Only simulation requires OpenCV.
            a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        return image_tensor(a)
    sensors = []
    for name, dim in SENSOR_DIMS.items():
        a = np.asarray(observation[f"walker/{name}"], dtype=np.float32).reshape(-1)
        if a.shape != (dim,) or not np.isfinite(a).all():
            raise ValueError(f"Invalid {name}: expected {dim} finite values")
        sensors.append(torch.from_numpy(a.copy()))
    return (eye("left_eye"), eye("right_eye"), *sensors)
