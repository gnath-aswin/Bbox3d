import numpy as np

def get_aabb(center, size):
    min_corner = center - size / 2
    max_corner = center + size / 2
    return min_corner, max_corner


def compute_iou_3d(center_pred, size_pred, center_gt, size_gt):
    min_p, max_p = get_aabb(center_pred, size_pred)
    min_g, max_g = get_aabb(center_gt, size_gt)

    min_inter = np.maximum(min_p, min_g)
    max_inter = np.minimum(max_p, max_g)

    inter_dim = np.maximum(0, max_inter - min_inter)
    inter_vol = inter_dim[0] * inter_dim[1] * inter_dim[2]

    vol_p = np.prod(size_pred)
    vol_g = np.prod(size_gt)

    union_vol = vol_p + vol_g - inter_vol

    if union_vol == 0:
        return 0.0

    return inter_vol / union_vol