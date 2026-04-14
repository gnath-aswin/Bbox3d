import torch
import torch.nn as nn
import torch.nn.functional as F

class BBoxLoss(nn.Module):
    def __init__(self, w_center=1.0, w_size=1.0, w_yaw=1.0,  w_diou=1.0):
        super().__init__()
        self.w_center = w_center
        self.w_size = w_size
        self.w_yaw = w_yaw
        self.w_diou = w_diou

    def forward(self, pred, target):
        pred_center, pred_size, pred_yaw = pred
        gt_center, gt_size, gt_yaw = target

        loss_center = F.mse_loss(pred_center, gt_center)
        loss_size = F.mse_loss(torch.log(pred_size + 1e-6), torch.log(gt_size + 1e-6))

        # # Symmetry
        # loss_yaw = (
        #     F.mse_loss(torch.sin(pred_yaw), torch.sin(gt_yaw)) +
        #     F.mse_loss(torch.cos(pred_yaw), torch.cos(gt_yaw))
        # )

        loss_yaw = 1 - torch.cos(pred_yaw - gt_yaw)
        loss_yaw = loss_yaw.mean()  

        # DIoU loss
        diou = self.compute_diou_3d(pred_center, pred_size, gt_center, gt_size)
        loss_diou = 1 - diou.mean()

        total_loss = (
            self.w_center * loss_center +
            self.w_size * loss_size +
            self.w_yaw * loss_yaw +
            self.w_diou * loss_diou
        )

        return {
            "total": total_loss,
            "center": loss_center,
            "size": loss_size,
            "yaw": loss_yaw, 
            "diou": loss_diou
        }
    
    def compute_diou_3d(self, center_pred, size_pred, center_gt, size_gt):
        # Convert to corners
        min_p = center_pred - size_pred / 2
        max_p = center_pred + size_pred / 2

        min_g = center_gt - size_gt / 2
        max_g = center_gt + size_gt / 2

        # Intersection
        min_inter = torch.maximum(min_p, min_g)
        max_inter = torch.minimum(max_p, max_g)
        inter_dim = torch.clamp(max_inter - min_inter, min=0)
        inter_vol = inter_dim.prod(dim=-1)

        # Volumes
        vol_p = size_pred.prod(dim=-1)
        vol_g = size_gt.prod(dim=-1)

        union = vol_p + vol_g - inter_vol
        iou = inter_vol / (union + 1e-6)

        # Center distance
        center_dist = ((center_pred - center_gt) ** 2).sum(dim=-1)

        # Enclosing box diagonal
        min_c = torch.minimum(min_p, min_g)
        max_c = torch.maximum(max_p, max_g)
        diag = ((max_c - min_c) ** 2).sum(dim=-1)

        diou = iou - center_dist / (diag + 1e-6)

        return diou