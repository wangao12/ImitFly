"""Spiking Convolutional Vision Encoder (SCVE) used by ImitFly v51."""
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class MembraneLIF(nn.Module):
    """Return the LIF membrane representation used by the original v51 model."""
    def __init__(self, threshold=1.0, decay=0.9):
        super().__init__()
        self.neuron = snn.Leaky(
            beta=decay, threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(),
            reset_mechanism="subtract", output=False,
        )

    def forward(self, inputs, mem=None):
        if mem is None:
            self.neuron.reset_mem()
        _, mem = self.neuron(inputs, mem)
        return mem


class SpikingConvolutionalVisionEncoder(nn.Sequential):
    """Shared binocular encoder: [B,T,1,32,32] -> [B,T,8192]."""
    def __init__(self, spiking=True):
        # Sequential indices preserve the historical eye_conv state-dict keys.
        super().__init__(
            nn.Conv2d(1, 32, 3, padding=1),
            MembraneLIF(threshold=1.0, decay=0.9) if spiking else nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, images):
        batch_size, sequence_length = images.shape[:2]
        features = super().forward(images.view(batch_size * sequence_length, 1, 32, 32))
        return features.view(batch_size, sequence_length, -1)
