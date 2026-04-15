import numpy as np
import numpy.typing as npt
from typing import Any

Array = npt.NDArray[np.float64]


def param_to_box(center: Array, size: Array, yaw: float) -> Array:
    length, width, height = size

    # local box (centered at origin)
    x = length / 2
    y = width / 2
    z = height / 2

    corners = np.array(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ]
    )

    # rotation
    R = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    corners = corners @ R.T
    corners = corners + center

    return corners


def box_to_param(box: Array) -> tuple[Array, Array, float]:
    """
    box: (8,3)
    returns: center, size, yaw
    """

    # Center
    center = box.mean(axis=0)

    # PCA
    pts_xy = box[:, :2]  # only XY for yaw
    cov = np.cov(pts_xy.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]
    yaw = np.arctan2(main_axis[1], main_axis[0])

    # Align box to exact size
    R = np.array(
        [
            [np.cos(-yaw), -np.sin(-yaw), 0.0],
            [np.sin(-yaw), np.cos(-yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    aligned = (box - center) @ R.T
    min_vals = aligned.min(axis=0)
    max_vals = aligned.max(axis=0)
    size = max_vals - min_vals

    return center, size, yaw


def extract_objects(sample: dict[str | Any]) -> list[dict[str, Array | float]]:
    pc = sample["point_cloud"]  # (3, H, W)
    masks = sample["mask"]  # (N, H, W)
    boxes = sample["bbox3d"]  # (N, 8, 3)

    objects = []
    for i in range(len(masks)):
        mask = masks[i] > 0
        points = pc[:, mask].T  # (Ni, 3)

        if len(points) < 20:
            continue

        # Basic z filter
        points = points[points[:, 2] > 0]
        if len(points) == 0:
            continue

        z = points[:, 2]
        z_min, z_max = np.percentile(z, [1, 99])
        points = points[(z > z_min) & (z < z_max)]
        if len(points) == 0:
            continue

        # Convert box to params
        box = boxes[i]
        center, size, yaw = box_to_param(box)

        obj = {
            "points": points,  # (Ni,3)
            "center": center,
            "size": size,
            "yaw": yaw,
        }

        objects.append(obj)

    return objects


# Centering, Scale, Point count
def preprocess_object(
    obj: dict[str, Array | float], num_points: int = 5120
) -> dict[str, Array | float]:
    points = obj["points"]

    # center
    pc_center = points.mean(axis=0)
    points = points - pc_center

    # scale
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / (scale + 1e-6)

    # Resample
    N = len(points)
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.random.choice(N, num_points, replace=True)

    points = points[idx]

    center = (obj["center"] - pc_center) / scale
    size = obj["size"] / scale
    yaw = obj["yaw"]

    return {
        "points": points,
        "center": center,
        "size": size,
        "yaw": yaw,
        "pc_center": pc_center,
        "scale": scale,
    }


def denormalize_prediction(pred_center, pred_size, obj):
    pc_center = obj["pc_center"]
    scale = obj["scale"]

    center = pred_center * scale + pc_center
    size = pred_size * scale

    return center, size
