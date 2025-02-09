from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import wandb
from typing import List, Tuple

from configs.hparams import config


def compute_accuracy(preds: List[torch.Tensor], targets: List[torch.Tensor]) -> float:
    """
    Compute accuracy which equals part of right answers.

    :param preds: Predictions of a model.
    :param targets: Right answers.
    :return: Accuracy as float.
    """
    result = (targets == preds).float().mean()
    return result


def config_train_process(
    train_dataset: Dataset, test_dataset: Dataset, device: torch.device
) -> Tuple[DataLoader, DataLoader, torch.device, nn.Module, nn.Module, optim.Optimizer]:
    """
    Config several objects for training process: data loaders, model, loss function and optimizer.

    :param train_dataset: Dataset for train.
    :param test_dataset: Dataset for test.
    :param device: Device: "cpu", "cuda".
    :return:
    """
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=config["batch_size"]
    )

    model = resnet18(
        pretrained=False,
        num_classes=config["num_classes"],
        zero_init_residual=config["zero_init_residual"],
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    return train_loader, test_loader, device, model, criterion, optimizer


def train(train_dataset: Dataset, test_dataset: Dataset) -> None:
    """
    Main function for model training.

    :param train_dataset: Dataset for train.
    :param test_dataset: Dataset for test.
    :return: None.
    """
    device = torch.device("cuda")
    train_loader, test_loader, device, model, criterion, optimizer = (
        config_train_process(train_dataset, test_dataset, device)
    )
    wandb.watch(model)
