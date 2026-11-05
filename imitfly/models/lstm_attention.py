"""Historical CNN/LSTM/attention comparator. Fusion softmax has one output."""
import torch
from torch import nn


class LSTMAttentionPolicy(nn.Module):

    def __init__(self, sequence_length):
        super().__init__()
        self.seq_len = sequence_length
        self.eye_conv = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.img_lstm = nn.LSTM(32 * 16 * 16 * 2, 64, batch_first=True)
        self.sensor_lstms = nn.ModuleList([nn.LSTM(3, 64, batch_first=True), nn.LSTM(3, 64, batch_first=True), nn.LSTM(25, 64, batch_first=True), nn.LSTM(25, 64, batch_first=True), nn.LSTM(3, 64, batch_first=True), nn.LSTM(3, 64, batch_first=True), nn.LSTM(2, 64, batch_first=True)])
        self.sensor_attention = nn.Linear(64, 1)
        self.fusion_attention = nn.Linear(128, 1)
        self.mu_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 12))
        self.log_std_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 12))

    def forward(self, x):
        left_eye, right_eye, *sensors = x
        batch_size, seq_len = (left_eye.shape[0], left_eye.shape[1])
        left_features = self.eye_conv(left_eye.view(batch_size * seq_len, 1, 32, 32))
        left_features = left_features.view(batch_size, seq_len, -1)
        right_features = self.eye_conv(right_eye.view(batch_size * seq_len, 1, 32, 32))
        right_features = right_features.view(batch_size, seq_len, -1)
        combined_img = torch.cat([left_features, right_features], dim=2)
        img_out, _ = self.img_lstm(combined_img)
        img_feat = img_out[:, -1, :]
        sensor_feats = []
        for i, lstm in enumerate(self.sensor_lstms):
            sensor_data = sensors[i].view(batch_size, self.seq_len, -1)
            out, _ = lstm(sensor_data)
            sensor_feats.append(out[:, -1, :])
        sensor_stack = torch.stack(sensor_feats, dim=1)
        attn_weights = torch.softmax(self.sensor_attention(sensor_stack), dim=1)
        weighted_sensors = torch.sum(sensor_stack * attn_weights, dim=1)
        combined = torch.cat([img_feat, weighted_sensors], dim=1)
        fusion_weights = torch.softmax(self.fusion_attention(combined), dim=1)
        final_feat = combined * fusion_weights
        mu = self.mu_head(final_feat)
        log_std = self.log_std_head(final_feat)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return (mu, log_std)
