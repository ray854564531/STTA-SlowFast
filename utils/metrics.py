"""Accuracy metrics for action recognition."""
import torch


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(targets.view(1, -1).expand_as(pred.t()))
        return (correct[:k].reshape(-1).float().sum() / targets.size(0)).item()
