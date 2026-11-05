"""ImitFly: SCVE, recurrent sensory encoding, HMAM, and Gaussian action heads."""
import torch
from torch import nn
from .scve import SpikingConvolutionalVisionEncoder
from .hmam import hierarchical_multimodal_attention
from imitfly.contracts import SENSOR_DIMS


class ImitFlyPolicy(nn.Module):
    """Fruit-fly flight policy corresponding to the model_meansnn51 family.

    Input order: left eye, right eye, accelerometer, gyro, joint position,
    joint velocity, velocimeter, world z-axis, task input. Each input has
    leading [batch, sequence] dimensions. Returns mean and log-standard
    deviation of a diagonal Gaussian over 12 canonical actions.
    """
    def __init__(self, sequence_length=5, use_spiking=True, use_attention=True, use_memory=True):
        super().__init__()
        self.seq_len = sequence_length
        self.use_attention = use_attention
        self.use_memory = use_memory
        self.eye_conv = SpikingConvolutionalVisionEncoder(spiking=use_spiking)
        self.img_lstm = (nn.LSTM(32 * 16 * 16 * 2, 64, batch_first=True)
                         if use_memory else nn.Linear(32 * 16 * 16 * 2, 64))
        self.sensor_lstms = nn.ModuleList(
            [nn.LSTM(dim, 64, batch_first=True) if use_memory else nn.Linear(dim, 64)
             for dim in SENSOR_DIMS.values()]
        )
        self.sensor_attention = nn.Linear(64, 1)
        self.fusion_attention = nn.Sequential(
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 4),
        )
        for layer in self.fusion_attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.mu_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 12))
        self.log_std_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 12))

    def forward(self, inputs, training=False):
        left_eye, right_eye, *sensors = inputs
        batch_size = left_eye.shape[0]
        binocular = torch.cat([self.eye_conv(left_eye), self.eye_conv(right_eye)], dim=2)
        if self.use_memory:
            visual_sequence, _ = self.img_lstm(binocular)
            visual = visual_sequence[:, -1, :]
        else:
            visual = self.img_lstm(binocular[:, -1, :])
        sensor_features = []
        for sensor, lstm in zip(sensors, self.sensor_lstms):
            sensor = sensor.view(batch_size, self.seq_len, -1)
            if self.use_memory:
                sequence, _ = lstm(sensor)
                sensor_features.append(sequence[:, -1, :])
            else:
                sensor_features.append(lstm(sensor[:, -1, :]))
        stacked = torch.stack(sensor_features, dim=1)
        if self.use_attention:
            fused = hierarchical_multimodal_attention(
                visual, stacked, self.sensor_attention, self.fusion_attention,
            )
        else:
            fused = torch.cat([visual, stacked.mean(dim=1)], dim=1)
        return self.mu_head(fused), self.log_std_head(fused).clamp(-20, 2)
