"""ImitFly sensor-recurrent attention model for 59-action FlyBody walking."""
import torch
from torch import nn
from imitfly.contracts import WALKING_SENSOR_DIMS as SENSOR_GROUPS

class WalkingPolicy(nn.Module):

    def __init__(self, sensor_dims=SENSOR_GROUPS, hidden_size=256):
        super().__init__()
        self.sensor_lstms = nn.ModuleDict({name: nn.LSTM(input_size=dim, hidden_size=hidden_size, batch_first=True) for name, dim in sensor_dims.items()})
        self.attention = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
        self.mu_head = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 59))
        self.log_std_head = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 59))

    def forward(self, x):
        lstm_outputs = []
        for name, lstm in self.sensor_lstms.items():
            sensor_input = x[name]
            out, (h_n, _) = lstm(sensor_input)
            lstm_outputs.append(h_n.squeeze(0))
        combined = torch.stack(lstm_outputs, dim=1)
        attn_weights = torch.softmax(self.attention(combined), dim=1)
        context = torch.sum(combined * attn_weights, dim=1)
        mu = self.mu_head(context)
        log_std = self.log_std_head(context)
        return (mu, log_std)
