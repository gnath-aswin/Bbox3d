import json
import optuna
import mlflow
import argparse
import torch
from torch.utils.data import DataLoader

from src.model import PointNetBBox
from src.train.trainer import Trainer
from src.data.dataset import Custom3DDataset, ObjectDataset
from src.data.preprocess import extract_objects, preprocess_object
from src.data.splits import load_split
from src.loss import BBoxLoss
from src.utils.config import load_config, sample_from_config
from src.logging.mlflow_logger import MLflowLogger


# Dataset builder
def build_dataset(scene_split):
    objects = []
    for sample in scene_split:
        for obj in extract_objects(sample):
            objects.append(preprocess_object(obj))
    return ObjectDataset(objects)


# Datset builders
def build_loaders(train_dataset, val_dataset, batch_size, num_workers=2):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Build Trainer
def build_trainer(params, train_loader, val_loader, trial_number, trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PointNetBBox().to(device)

    loss_fn = BBoxLoss(
        w_center=params["w_center"],
        w_size=params["w_size"],
        w_yaw=params["w_yaw"],
        w_diou=params["w_diou"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    logger = MLflowLogger(f"trial_{trial_number}")

    trainer = Trainer(
        model,
        optimizer,
        train_loader,
        logger=logger,
        device=device,
        val_loader=val_loader,
        loss_fn=loss_fn,
        trial=trial,
    )

    return trainer


# Objective function (Optuna)
def objective(trial, config, train_dataset, val_dataset, experiment_id):
    with mlflow.start_run(nested=True, experiment_id=experiment_id):
        # Sample params
        params = sample_from_config(trial, config["search_space"])

        # Fill defaults
        params["lr"] = params.get("lr", config["training"]["lr"])
        params["batch_size"] = params.get(
            "batch_size", config["training"]["batch_size"]
        )

        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)

        # Build loaders
        train_loader, val_loader = build_loaders(
            train_dataset, val_dataset, params["batch_size"]
        )

        # Build trainer
        trainer = build_trainer(params, train_loader, val_loader, trial.number, trial)

        # Train
        try:
            trainer.train(epochs=config["training"]["epochs"])
        except optuna.exceptions.TrialPruned:
            mlflow.log_param("pruned", True)
            raise

        # Validate
        val_loss, val_iou = trainer.validate()

        # Objective
        w_loss = config["tune_objective"]["w_loss"]
        w_iou = config["tune_objective"]["w_iou"]
        val_score = w_loss * val_loss - w_iou * val_iou

        # Log
        mlflow.log_metrics(
            {
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_score": val_score,
            }
        )

        trial.set_user_attr("val_iou", val_iou)

        return val_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tune.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    experiment = mlflow.set_experiment("Tuning")
    experiment_id = experiment.experiment_id

    # Dataset
    scene_dataset = Custom3DDataset(config["data"]["path"])
    train_scene, val_scene, _ = load_split(scene_dataset)

    train_dataset = build_dataset(train_scene)
    val_dataset = build_dataset(val_scene)

    with mlflow.start_run(run_name="Tune test"):
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=3, interval_steps=1
            ),
        )
        study.optimize(
            lambda trial: objective(
                trial, config, train_dataset, val_dataset, experiment_id
            ),
            n_trials=config["tuning"]["n_trials"],
        )
        best_params = study.best_params

        # Log best params
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Save best params
        with open("configs/best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)

        print("Best params saved!")


if __name__ == "__main__":
    main()
