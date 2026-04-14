import numpy as np
from torch.utils.data import Subset, random_split


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15):
    total = len(dataset)

    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    return random_split(dataset, [train_size, val_size, test_size])


def save_split(train, val, test, path="data/splits.npz"):
    np.savez(
        path,
        train_idx=train.indices,
        val_idx=val.indices,
        test_idx=test.indices
    )


def load_split(dataset, path="../data/splits.npz"):
    data = np.load(path)

    train = Subset(dataset, data["train_idx"])
    val   = Subset(dataset, data["val_idx"])
    test  = Subset(dataset, data["test_idx"])

    return train, val, test
