import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetBBox(nn.Module):
    def __init__(self):
        super().__init__()

        # Point-wise feature extraction
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Global feature → prediction
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)   # 3 center + 3 size + 1 yaw
        )

    def forward(self, x):
        """
        x: (B, N, 3)
        """

        # Apply MLP per point
        x = self.mlp1(x)   # (B, N, 256)

        # Symmetric function (key idea of PointNet)
        x = torch.max(x, dim=1)[0]   # (B, 256)

        # Final prediction
        x = self.fc(x)   # (B, 7)

        center = x[:, 0:3]
        size = x[:, 3:6]
        yaw = x[:, 6]

        return center, size, yaw
    

    def bbox_loss(self, pred, target):
        pred_center, pred_size, pred_yaw = pred
        gt_center, gt_size, gt_yaw = target

        loss_center = F.mse_loss(pred_center, gt_center)
        loss_size = F.mse_loss(pred_size, gt_size)
        loss_yaw = F.mse_loss(pred_yaw, gt_yaw)

        return loss_center + loss_size + loss_yaw