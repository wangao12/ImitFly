"""Historical feedforward comparator; not a controlled single-factor v51 ablation."""
import torch
from torch import nn


class FeedForwardPolicy(nn.Module):

    def __init__(self):
        super().__init__()
        self.eye_conv = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.sensor_fc = nn.Sequential(nn.Linear(3 + 3 + 25 + 25 + 3 + 3 + 2, 128), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(32 * 16 * 16 * 2 + 128, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 12 * 2))

    def forward(self, x):
        left_eye, right_eye, *sensors = x
        left_features = self.eye_conv(left_eye[:, -1, :, :, :])
        left_features = left_features.view(left_eye.size(0), -1)
        right_features = self.eye_conv(right_eye[:, -1, :, :, :])
        right_features = right_features.view(right_eye.size(0), -1)
        sensor_features = [s[:, -1, :] for s in sensors]
        sensor_features = torch.cat(sensor_features, dim=1)
        sensor_features = self.sensor_fc(sensor_features)
        combined = torch.cat([left_features, right_features, sensor_features], dim=1)
        output = self.fc(combined)
        mean = output[:, :12]
        log_std = output[:, 12:]
        log_std = torch.clamp(log_std, min=-20, max=2)
        return (mean, log_std)
