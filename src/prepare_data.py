from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Tuple


def download_data() -> None:
    """
    Download data from torchvision library and save it to data/CIFAR10 folder.

    :return: None.
    """
    CIFAR10("../data/CIFAR10/train", download=True)
    CIFAR10("../data/CIFAR10/test", download=True)
    return


def prepare_data() -> Tuple[Dataset, Dataset]:
    """
    Initialize CIFAR10 test and train data with normalized tensors.

    :return: Tuple with train and test data as torchvision.datasets.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    train_dataset = CIFAR10(
        root="CIFAR10/train",
        train=True,
        transform=transform,
        download=False,
    )

    test_dataset = CIFAR10(
        root="CIFAR10/test",
        train=False,
        transform=transform,
        download=False,
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    download_data()
