import numpy as np


def param_to_box(center, size, yaw):
    l, w, h = size

    # local box (centered at origin)
    x = l / 2
    y = w / 2
    z = h / 2

    corners = np.array([
        [-x, -y, -z],
        [ x, -y, -z],
        [ x,  y, -z],
        [-x,  y, -z],
        [-x, -y,  z],
        [ x, -y,  z],
        [ x,  y,  z],
        [-x,  y,  z],
    ])

    # rotation 
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    corners = corners @ R.T

    corners = corners + center

    return corners

def box_to_param(box):
    """
    box: (8,3)
    returns: center, size, yaw
    """

    # Center
    center = box.mean(axis=0)

    # PCA (orientation)
    pts_xy = box[:, :2]   # only XY for yaw

    cov = np.cov(pts_xy.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    # main axis = largest eigenvector
    main_axis = eigvecs[:, np.argmax(eigvals)]

    yaw = np.arctan2(main_axis[1], main_axis[0])

    # Size (project onto axes)
    # rotate box to align with axes
    R = np.array([
        [np.cos(-yaw), -np.sin(-yaw), 0],
        [np.sin(-yaw),  np.cos(-yaw), 0],
        [0, 0, 1]
    ])

    aligned = (box - center) @ R.T

    min_vals = aligned.min(axis=0)
    max_vals = aligned.max(axis=0)

    size = max_vals - min_vals

    return center, size, yaw


def extract_objects(sample):
    pc = sample["point_cloud"]      # (3, H, W)
    masks = sample["mask"]          # (N, H, W)
    boxes = sample["bbox3d"]        # (N, 8, 3)

    objects = []
    for i in range(len(masks)):
        mask = masks[i] > 0

        # Extract points
        points = pc[:, mask].T   # (Ni, 3)

        if len(points) < 20:   # skip noisy objects
            continue

        # Filter points
        points = points[points[:, 2] > 0]

        if len(points) == 0:
            continue

        z = points[:, 2]
        z_min, z_max = np.percentile(z, [1, 99])
        points = points[(z > z_min) & (z < z_max)]

        if len(points) == 0:
            continue

        # Convert box → params (PCA)
        box = boxes[i]
        center, size, yaw = box_to_param(box)

        # Store
        obj = {
            "points": points,     # (Ni,3)
            "center": center,
            "size": size,
            "yaw": yaw
        }

        objects.append(obj)

    return objects


def preprocess_object(obj, num_points=5120):
    points = obj["points"]

    # center
    pc_center = points.mean(axis=0)
    points = points - pc_center

    # scale
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / (scale + 1e-6)

    # sample
    N = len(points)
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.random.choice(N, num_points, replace=True)

    points = points[idx]

    # adjust labels
    center = (obj["center"] - pc_center) / scale
    size = obj["size"] / scale
    yaw = obj["yaw"]

    return {
        "points": points,
        "center": center,
        "size": size,
        "yaw": yaw
    }