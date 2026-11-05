"""Hierarchical Multimodal Attention Mechanism (HMAM)."""
import torch


def hierarchical_multimodal_attention(visual, sensors, sensor_attention, fusion_attention):
    """Sensor weighting followed by attention over four 32-feature groups.

    Args:
        visual: [B,64] recurrent visual features.
        sensors: [B,7,64] recurrent sensor features.
        sensor_attention: learned 64 -> 1 scoring layer.
        fusion_attention: learned 128 -> 4 group scoring network.
    Returns:
        Fused representation of shape [B,128].
    """
    sensor_weights = torch.softmax(sensor_attention(sensors), dim=1)
    sensor_context = torch.sum(sensors * sensor_weights, dim=1)
    combined = torch.cat([visual, sensor_context], dim=1)
    group_weights = torch.softmax(fusion_attention(combined), dim=1)
    groups = [combined[:, i * 32:(i + 1) * 32] * group_weights[:, i:i + 1]
              for i in range(4)]
    return torch.cat(groups, dim=1)
