"""Forward pass wrapper – no business logic, only torch ops."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def run_inference(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model forward pass.

    Args:
        model: eval-mode classifier.
        tensor: batched image tensor of shape (1, 1, H, W).
        device: target device.

    Returns:
        ``(logits, probs)`` both of shape ``(num_classes,)`` on CPU.
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)          # (1, num_classes)
    logits = logits.squeeze(0).cpu()    # (num_classes,)
    probs = F.softmax(logits, dim=0)    # (num_classes,)
    return logits, probs
