import torch
import argparse
from tqdm import tqdm

from src.model import PointNetBBox
from src.metrics.iou3d import compute_iou_3d
from src.loss import BBoxLoss
from src.visualize import Visualizer
from src.data.dataset import Custom3DDataset
from src.data.splits import load_split
from src.data.preprocess import (
    extract_objects,
    preprocess_object,
    denormalize_prediction,
    param_to_box,
)


class Tester:
    def __init__(self, model, dataset, device="cpu"):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.loss_fn = BBoxLoss()

    def test(self, visualize=False, max_vis=1, save_vis=False):
        self.model.eval()

        total_iou = 0
        total_loss = 0
        total_center_loss = 0
        total_size_loss = 0
        total_yaw_loss = 0

        count = 0
        vis_count = 0

        with torch.no_grad():
            for sample in tqdm(self.dataset):
                objects = extract_objects(sample)

                pred_boxes_scene = []
                gt_boxes_scene = []

                for obj in objects:
                    processed = preprocess_object(obj)
                    points = (
                        torch.tensor(processed["points"], dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    gt_center = (
                        torch.tensor(processed["center"], dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    gt_size = (
                        torch.tensor(processed["size"], dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    gt_yaw = (
                        torch.tensor(processed["yaw"], dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )

                    # Predict
                    pred_center, pred_size, pred_yaw = self.model(points)

                    # Loss
                    loss_dict = self.loss_fn(
                        (pred_center, pred_size, pred_yaw), (gt_center, gt_size, gt_yaw)
                    )

                    pred_center = pred_center.cpu().numpy()[0]
                    pred_size = pred_size.cpu().numpy()[0]
                    pred_yaw = pred_yaw.cpu().numpy()[0]

                    # Convert back to scene
                    center, size = denormalize_prediction(
                        pred_center, pred_size, processed
                    )

                    pred_box = param_to_box(center, size, pred_yaw)
                    gt_box = param_to_box(obj["center"], obj["size"], obj["yaw"])
                    # Collect
                    pred_boxes_scene.append(pred_box)
                    gt_boxes_scene.append(gt_box)

                    # Metrics
                    iou = compute_iou_3d(center, size, obj["center"], obj["size"])
                    total_iou += iou
                    count += 1

                    total_loss += loss_dict["total"].item()
                    total_center_loss += loss_dict["center"].item()
                    total_size_loss += loss_dict["size"].item()
                    total_yaw_loss += loss_dict["yaw"].item()

                    # Visualization
                if visualize and vis_count < max_vis:
                    sample_vis = {
                        "image": sample["image"],
                        "point_cloud": sample["point_cloud"],
                        "mask": sample["mask"],
                        "bbox3d": sample["bbox3d"],
                    }

                    vis = Visualizer(sample_vis)
                    if save_vis:
                        save_path = f"outputs/vis/scene_{vis_count}.png"
                    else:
                        save_path = None
                    vis.show_scene_predictions(
                        gt_boxes=gt_boxes_scene,
                        pred_boxes=pred_boxes_scene,
                        show=visualize,
                        save_path=save_path,
                    )

                    vis_count += 1

            # Final Results
            print("\n===== TEST RESULTS =====")
            print(f"Mean IoU           : {total_iou / count:.4f}")
            print(f"Total Loss         : {total_loss / count:.4f}")
            print(f"Center Loss        : {total_center_loss / count:.4f}")
            print(f"Size Loss          : {total_size_loss / count:.4f}")
            print(f"Yaw Loss           : {total_yaw_loss / count:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--show_vis", action="store_true")
    parser.add_argument("--save_vis", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = args.data_path
    dataset = Custom3DDataset(data_path)
    test_sample = load_split(dataset)[2]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PointNetBBox().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    tester = Tester(model, test_sample, device=device)

    tester.test(
        visualize=args.show_vis,  # turn on for debugging
        max_vis=2,
        save_vis=args.save_vis,
    )


if __name__ == "__main__":
    main()
