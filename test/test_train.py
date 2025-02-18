import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from unittest.mock import Mock

from src.prepare_data import download_data, prepare_data
from src import train


@pytest.fixture
def prepare_dataset():
    download_data()
    return prepare_data()


@pytest.fixture
def prepare_one_epoch_train():
    images = Mock()
    images.to = Mock()

    labels = Mock()
    labels.to = Mock()

    # Create mock analogs
    mock_model = Mock(return_value=labels)
    mock_param_1 = nn.Parameter(torch.randn(5, 5))
    mock_param_2 = nn.Parameter(torch.randn(10))
    mock_parameters = [mock_param_1, mock_param_2]
    mock_model.parameters = Mock(return_value=mock_parameters)

    mock_criterion = Mock()
    mock_criterion.return_value = Mock()
    mock_criterion.return_value.backward = Mock()

    mock_optimizer = optim.SGD(mock_model.parameters(), lr=0.01)
    mock_optimizer.zero_grad = Mock(side_effect=mock_optimizer.zero_grad)
    mock_optimizer.step = Mock()
    mock_optimizer.zero_grad = Mock()

    return images, labels, mock_model, mock_criterion, mock_optimizer


@pytest.mark.parametrize(["device_name"], [["cpu"], ["cuda"]])
def test_train_one_epoch(device_name, prepare_one_epoch_train):
    device = torch.device(device_name)
    images, labels, model, criterion, optimizer = prepare_one_epoch_train
    loss = train.train_one_epoch(images, labels, model, criterion, optimizer, device)

    # Check if data was transferred to the device
    images.to.assert_called_with(device), f"Images weren't transferred to {device_name}"
    labels.to.assert_called_with(device), f"Labels weren't transferred to {device_name}"

    # Change mocks for next tests
    images = images.to(device)
    labels = labels.to(device)

    # Check model calls
    model.assert_called_once(), "Model wasn't called once"
    model.assert_called_with(images), "Model got wrong input"

    # Check loss backward call
    (
        criterion.assert_called_with(model.return_value, labels),
        "Criterion got wrong args, must be model_output, labels",
    )
    (
        criterion.return_value.backward.assert_called_once(),
        "loss.backward() wasn't called",
    )

    # Check optimizer calls
    optimizer.step.assert_called_once(), "optimizer.step() wasn't called"
    optimizer.zero_grad.assert_called_once(), "optimizer.zero_grad() wasn't called"

    # Check the output of train function
    assert loss == criterion.return_value, "Wrong output of train function"
