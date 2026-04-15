import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, Subset, random_split

from src.logging.mlflow_logger import MLflowLogger
from src.train.trainer import Trainer
from src.model import PointNetBBox
from src.loss import BBoxLoss
from src.data.dataset import Custom3DDataset, ObjectDataset
from src.data.preprocess import extract_objects, preprocess_object
from src.data.splits import get_or_create_split
from src.utils.config import load_best_params


def _parse_args():
    parser = argparse.ArgumentParser(description="3D Bounding Box Training")

    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--name", type=str, default="baseline_run", help="MLflow experiment name"
    )
    parser.add_argument(
        "--use_tuned", action="store_true", help="Use tuned hyperparameters"
    )
    parser.add_argument("--epochs", type=int, default=100)
    # Override params
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    return parser.parse_args()


# Dataset builder
def _build_dataset(scene_split):
    objects = []
    for sample in scene_split:
        for obj in extract_objects(sample):
            objects.append(preprocess_object(obj))
    return ObjectDataset(objects)


def main():
    args = _parse_args()

    # Data
    data_path = args.data_path
    scene_dataset = Custom3DDataset(data_path)
    train_scene, val_scene, _ = get_or_create_split(scene_dataset)
    train_dataset = _build_dataset(train_scene)
    val_dataset = _build_dataset(val_scene)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training parameter
    default_params = {
        "lr": 1e-3,
        "batch_size": 32,
        "w_center": 1.0,
        "w_size": 1.0,
        "w_yaw": 1.0,
    }
    if args.use_tuned:
        try:
            tuned = load_best_params(path="configs/best_params_diou.json")
            default_params.update(tuned)
        except Exception as e:
            print("Failed to load tuned params:", e)
    # Overrides
    if args.lr is not None:
        default_params["lr"] = args.lr
    if args.batch_size is not None:
        default_params["batch_size"] = args.batch_size
    params = default_params

    # Loader
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

    # Model
    model = PointNetBBox().to(device)
    loss_fn = BBoxLoss(
        w_center=params["w_center"], w_size=params["w_size"], w_yaw=params["w_yaw"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    # Log
    logger = MLflowLogger(args.name)
    logger.log_params(params)
    print("\n===== CONFIG =====")
    print(f"Run name: {args.name}")
    print(f"Data path: {args.data_path}")
    for k, v in params.items():
        print(f"{k}: {v}")

    # Train
    trainer = Trainer(
        model,
        optimizer,
        train_loader,
        loss_fn=loss_fn,
        logger=logger,
        device=device,
        val_loader=val_loader,
    )
    print("\n===== TRAINING =====")
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
