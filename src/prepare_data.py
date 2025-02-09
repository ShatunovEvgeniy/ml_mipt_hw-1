from torchvision.datasets import CIFAR10


def download_data() -> None:
    """
    Download data from torchvision library and save it to data/CIFAR10 folder.

    :return: None.
    """
    CIFAR10("../data/CIFAR10/train", download=True)
    CIFAR10("../data/CIFAR10/test", download=True)
    return


if __name__ == "__main__":
    download_data()
