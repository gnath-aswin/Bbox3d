import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split

from src.logging.mlflow_logger import MLflowLogger
from src.train.trainer import Trainer
from src.model import PointNetBBox
from src.data.dataset import Custom3DDataset, ObjectDataset
from src.data.preprocess import extract_objects, preprocess_object
from src.data.splits import load_split



def main():
    # Data processing
    data_path = "/media/void/Crucial X8/dl_challenge/dl_challenge"
    scene_dataset = Custom3DDataset(data_path)

    train_scene, val_scene, _ = load_split(scene_dataset)

    # Extract objects
    train_objects = []
    for sample in train_scene:
        objects = extract_objects(sample)

        for obj in objects:
            processed = preprocess_object(obj)
            train_objects.append(processed)

    train_dataset = ObjectDataset(train_objects)

    val_objects = []
    for sample in val_scene:
        objects = extract_objects(sample)

        for obj in objects:
            processed = preprocess_object(obj)
            val_objects.append(processed)

    val_dataset = ObjectDataset(val_objects)

    # Train pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loader 
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(val_dataset, 
            batch_size=16, 
            shuffle=True)

    model = PointNetBBox().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logger = MLflowLogger("test_")
    logger.log_params({"lr": 1e-3, "batch_size": 32})

    trainer = Trainer(
        model,
        optimizer,
        train_loader,
        logger,
        device,
        val_loader=val_loader   
    )

    trainer.train(epochs=300)

if __name__ == "__main__":
    main()