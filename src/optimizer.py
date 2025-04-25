"""
This class contains helper functions which will help get the optimizer
"""

from typing import Any, Dict
import torch

def get_optimizer(
    model: torch.nn.Module, config: Dict[str, Any]
) -> torch.optim.Optimizer:
    optimizer_type = config.get("optimizer_type", "sgd")
    learning_rate = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)

    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return optimizer