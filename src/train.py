import torch
from typing import List


def compute_accuracy(preds: List[torch.Tensor], targets: List[torch.Tensor]) -> float:
    """
    Compute accuracy which equals part of right answers.

    :param preds: Predictions of a model.
    :param targets: Right answers.
    :return: Accuracy as float.
    """
    result = (targets == preds).float().mean()
    return result
