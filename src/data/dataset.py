from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class Custom3DDataset(Dataset):
    def __init__(self, path, transform=None):
        self.root_dir = Path(path)
        self.sample_paths = [p for p in self.root_dir.iterdir() if p.is_dir()]
        self.transform = transform
        
    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        path = self.sample_paths[idx]

        # Image
        img = cv2.imread(str(path / "rgb.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = img.transpose(2, 0, 1)  

        # PC, Mask, BBox
        pc = np.load(path / "pc.npy")        # (3,H,W)
        mask = np.load(path / "mask.npy")    # (N,H,W)
        bbox = np.load(path / "bbox3d.npy")  # (N,8,3)

        sample = {
            "image": rgb,
            "point_cloud": pc,
            "mask": mask,
            "bbox3d": bbox
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class ObjectDataset(Dataset):
    def __init__(self, objects):
        self.data = objects

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj = self.data[idx]

        return {
            "points": torch.tensor(obj["points"], dtype=torch.float32),
            "center": torch.tensor(obj["center"], dtype=torch.float32),
            "size": torch.tensor(obj["size"], dtype=torch.float32),
            "yaw": torch.tensor(obj["yaw"], dtype=torch.float32),
        }