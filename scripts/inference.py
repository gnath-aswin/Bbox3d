# *************** Trial script *************

import argparse
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm

from src.model import PointNetBBox
from src.data.dataset import Custom3DDataset
from src.data.splits import load_split
from src.data.preprocess import (
    extract_objects,
    preprocess_object,
    denormalize_prediction,
    param_to_box,
)
from src.visualize import Visualizer

# Open3D helpers 
def create_box_lines(corners, color):
    lines = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    return line_set

def create_legend():
    geometries = []

    # Create small points for legend
    points = np.array([
        [0, 0, 0],   # GT
        [0, 0.2, 0], # Prediction
    ])

    colors = np.array([
        [0, 1, 0],  # green
        [1, 0, 0],  # red
    ])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries.append(pcd)

    return geometries

def visualize_scene(points, gt_boxes, pred_boxes, show_pc=True):
    geometries = []

    # Point cloud
    if show_pc:
        z = points[:, 2]
        z_min, z_max = np.percentile(z, [1, 99])
        points = points[(z > z_min) & (z < z_max)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        colors = np.tile(np.array([[0.7, 0.85, 1.0]]), (len(pcd.points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)

    # GT boxes (green)
    for box in gt_boxes:
        geometries.append(create_box_lines(box, [0, 1, 0]))

    # Pred boxes (red)
    for box in pred_boxes:
        geometries.append(create_box_lines(box, [1, 0, 0]))

    # Axis
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(axis)

    geometries.extend(create_legend())

    o3d.visualization.draw_geometries(geometries)


# Inference 
def run_inference(model, dataset, device="cpu", max_vis=3):
    model.eval()
    vis_count = 0

    with torch.no_grad():
        for sample in tqdm(dataset):
            objects = extract_objects(sample)

            pred_boxes_scene = []
            gt_boxes_scene = []

            for obj in objects:
                processed = preprocess_object(obj)

                points = (
                    torch.tensor(processed["points"], dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )

                # Predict
                pred_center, pred_size, pred_yaw = model(points)

                pred_center = pred_center.cpu().numpy()[0]
                pred_size = pred_size.cpu().numpy()[0]
                pred_yaw = pred_yaw.cpu().numpy()[0]

                # Convert back
                center, size = denormalize_prediction(
                    pred_center, pred_size, processed
                )

                pred_box = param_to_box(center, size, pred_yaw)
                gt_box = param_to_box(obj["center"], obj["size"], obj["yaw"])

                pred_boxes_scene.append(pred_box)
                gt_boxes_scene.append(gt_box)

            # Visualization
            if vis_count < max_vis:
                # -------- 2D Visualization (image + projection) --------
                sample_vis = {
                    "image": sample["image"],
                    "point_cloud": sample["point_cloud"],
                    "mask": sample["mask"],
                    "bbox3d": sample["bbox3d"],
                }


                vis = Visualizer(sample_vis)

                vis.show_scene_predictions(
                    gt_boxes=gt_boxes_scene,
                    pred_boxes=pred_boxes_scene,
                    show_pc=False,   # optional
                    show=True,
                    save_path=None,  # or path if you want
                )

 
                pc = sample["point_cloud"].reshape(3, -1).T
    
                visualize_scene(
                    points=pc,
                    gt_boxes=gt_boxes_scene,
                    pred_boxes=pred_boxes_scene,
                )


                vis_count += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_vis", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    data = Custom3DDataset(args.data_path)
    print(data.sample_paths)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = PointNetBBox().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    run_inference(
        model,
        data,
        device=device,
        max_vis=args.num_vis,
    )


if __name__ == "__main__":
    main()