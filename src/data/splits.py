import numpy as np
from torch.utils.data import Dataset, Subset, random_split
from pathlib import Path


def split_dataset(
    dataset: Dataset, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> tuple[Subset, Subset, Subset]:
    total = len(dataset)

    # compute split sizes
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    return random_split(dataset, [train_size, val_size, test_size])


def save_split(
    train: Subset, val: Subset, test: Subset, path: str | Path = "data/splits.npz"
) -> None:
    # save indices for reproducibility
    np.savez(
        path,
        train_idx=train.indices,
        val_idx=val.indices,
        test_idx=test.indices,
    )


def load_split(
    dataset: Dataset, path: str | Path = "data/splits.npz"
) -> tuple[Subset, Subset, Subset]:
    data = np.load(path)

    # reconstruct subsets from saved indices
    train = Subset(dataset, data["train_idx"])
    val = Subset(dataset, data["val_idx"])
    test = Subset(dataset, data["test_idx"])

    return train, val, test
