import torch
import os
from src.loss import BBoxLoss
from src.metrics.iou3d import compute_iou_3d

class Trainer:
    def __init__(self, model, optimizer, train_loader, loss_fn=None, logger=None, device="cpu", val_loader=None):
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else BBoxLoss()
        self.optimizer = optimizer
        self.loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.device = device

        self.best_loss = float("inf")
        self.last_loss = None
        self.best_iou = 0.0

    def train_one_epoch(self):
        self.model.train()
    
        total_loss = 0
        total_center = 0
        total_size = 0
        total_yaw = 0

        for batch in self.loader:
            points = batch["points"].to(self.device)
            center = batch["center"].to(self.device)
            size = batch["size"].to(self.device)
            yaw = batch["yaw"].to(self.device)

            pred = self.model(points)
            loss_dict = self.loss_fn(pred, (center, size, yaw))
            loss = loss_dict["total"] 
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss_dict["total"].item()
            total_center += loss_dict["center"].item()
            total_size += loss_dict["size"].item()
            total_yaw += loss_dict["yaw"].item()    

        return {
            "total": total_loss / len(self.loader),
            "center": total_center / len(self.loader),
            "size": total_size / len(self.loader),
            "yaw": total_yaw / len(self.loader),
        }

    def train(self, epochs):
        if self.logger:
            self.logger.start()

            self.logger.log_params({
                "batch_size": self.loader.batch_size,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "model": self.model.__class__.__name__,
                })
        try:
            for epoch in range(epochs):
                train_losses = self.train_one_epoch()

                # Validation
                if self.val_loader is not None:
                    val_loss, val_iou = self.validate()
                else:
                    val_loss, val_iou = None, None
                
                if val_loss is not None and val_loss < self.best_loss:
                    self.best_loss = val_loss

                if val_loss is not None:
                    print(
                        f"Epoch {epoch}: "
                        f"Train={train_losses['total']:.4f} "
                        f"(C={train_losses['center']:.3f}, "
                        f"S={train_losses['size']:.3f}, "
                        f"Y={train_losses['yaw']:.3f}) "
                        f"| Val={val_loss:.4f}, IoU={val_iou:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch}: "
                        f"Train={train_losses['total']:.4f}"
                    )

                # Log metrics
                if self.logger:
                    metrics = {
                        "train_loss": train_losses["total"],
                        "train_center": train_losses["center"],
                        "train_size": train_losses["size"],
                        "train_yaw": train_losses["yaw"],
                    }
                    if val_loss is not None:
                        metrics["val_loss"] = val_loss
                        metrics["val_iou"] = val_iou

                    self.logger.log_metrics(metrics, step=epoch)

                # Save best model
                if val_loss is not None:
                    if val_iou > self.best_iou:
                        self.best_iou = val_iou
                        path = self.save_checkpoint("outputs/models/best_model.pth", epoch, val_loss)

                        if self.logger:
                            self.logger.log_artifact(path)

        finally:
            if self.logger:
                self.logger.end()

        return self.best_loss
    
    def validate(self):
        self.model.eval()

        total_loss = 0
        total_iou = 0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                points = batch["points"].to(self.device)
                center = batch["center"].to(self.device)
                size = batch["size"].to(self.device)
                yaw = batch["yaw"].to(self.device)

                pred = self.model(points)
                loss_dict = self.loss_fn(pred, (center, size, yaw))
                loss = loss_dict["total"]

                total_loss += loss.item()

                # IoU 
                pred_center, pred_size, _ = pred

                for i in range(points.shape[0]):
                    iou = compute_iou_3d(
                        pred_center[i].cpu().numpy(),
                        pred_size[i].cpu().numpy(),
                        center[i].cpu().numpy(),
                        size[i].cpu().numpy()
                    )
                    total_iou += iou
                    count += 1

        avg_loss = total_loss / len(self.val_loader)
        avg_iou = total_iou / count if count > 0 else 0.0

        return avg_loss, avg_iou

    def save_checkpoint(self, path, epoch, loss):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss
        }, path)

        return path