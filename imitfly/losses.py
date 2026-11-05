"""Diagonal Gaussian distillation objective, student || expert."""
import torch
import math

def gaussian_kl_loss(mean, log_std, target_mean, target_std):
    if not torch.isfinite(target_std).all() or (target_std <= 0).any():
        raise ValueError("Expert standard deviations must be finite and positive")
    var_pred = torch.exp(2 * log_std)
    var_target = target_std ** 2
    return (0.5 * ((var_pred + (mean-target_mean)**2) / var_target
                  + torch.log(var_target) - torch.log(var_pred) - 1)).mean()


def gaussian_nll_loss(mean, log_std, target_mean, target_std):
    """Expected student NLL under the expert Gaussian, for loss ablation."""
    if not torch.isfinite(target_std).all() or (target_std <= 0).any():
        raise ValueError("Expert standard deviations must be finite and positive")
    return (log_std + 0.5 * ((target_mean - mean) ** 2 + target_std ** 2)
            / torch.exp(2 * log_std) + 0.5 * math.log(2 * math.pi)).mean()
