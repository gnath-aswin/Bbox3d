import os
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self, sample):
        self.img = sample["image"].transpose(1, 2, 0)
        self.pc = sample["point_cloud"]
        self.mask = sample["mask"]
        self.boxes = sample["bbox3d"]

        self.H, self.W, _ = self.img.shape

        self.points = self._prepare_points()

        # Projection
        self.u_all, self.v_all = self._project(self.points)
        self.z_all = self.points[:, 2]

        # Normalization
        self._compute_normalization()

    # Preprocessing
    def _prepare_points(self):
        points = self.pc.reshape(3, -1).T

        # Remove invalid depth
        points = points[points[:, 2] > 0]

        # Remove depth outliers
        z = points[:, 2]
        z_min, z_max = np.percentile(z, [1, 99])
        points = points[(z > z_min) & (z < z_max)]

        return points

    # Projection: World frame(not sure) to image frame
    def _project(self, points):
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2] + 1e-6

        u = X / Z
        v = Y / Z

        return u, v

    # Normalization
    def _compute_normalization(self):
        self.u_min, self.u_max = np.percentile(self.u_all, [1, 99])
        self.v_min, self.v_max = np.percentile(self.v_all, [1, 99])
        self.z_min, self.z_max = np.percentile(self.z_all, [1, 99])

    def _normalize_uv(self, u, v):
        u = (u - self.u_min) / (self.u_max - self.u_min) * self.W
        v = (v - self.v_min) / (self.v_max - self.v_min) * self.H
        return u, v

    def _normalize_z(self, z):
        z = (z - self.z_min) / (self.z_max - self.z_min + 1e-6)
        return np.clip(z, 0, 1)

    def _draw_box(self, box, ax, color="r"):
        box = np.asarray(box)
        if box.shape != (8, 3):
            raise ValueError(f"Box must be (8,3), got {box.shape}")

        u, v = self._project(box)
        u, v = self._normalize_uv(u, v)

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        for i, j in edges:
            ax.plot([u[i], u[j]], [v[i], v[j]], color=color, linewidth=2)

    # Visualization - Scene level
    def show(
        self,
        show_pc=True,
        show_boxes=True,
        show_mask=False,
        mask_idx=0,
        box_color="r",
        ax=None,
    ):
        if ax is None:
            plt.figure(figsize=(8, 6))
            ax = plt.gca()

        ax.imshow(self.img)

        # Mask
        if show_mask:
            m = self.mask[mask_idx]
            ax.imshow(m, alpha=0.5, cmap="jet")

        # Point cloud
        if show_pc:
            u, v = self._normalize_uv(self.u_all, self.v_all)
            z_norm = self._normalize_z(self.z_all)

            sc = ax.scatter(u, v, c=z_norm, cmap="jet", alpha=0.3)

            plt.colorbar(sc, ax=ax, label="Depth (Z)")

        # Bounding boxes
        if show_boxes:
            for box in self.boxes:
               self._draw_box(box)
        ax.set_title("Visualization")
        ax.axis("off")

    # Single Object Visualization
    def show_object(self, mask_idx=None, box_idx=None, obj_points=None, ax=None):
        if ax is None:
            plt.figure(figsize=(6, 6))
            ax = plt.gca()

        ax.imshow(self.img)

        # Mask
        if mask_idx is not None:
            mask = self.mask[mask_idx] > 0
            ax.imshow(mask, alpha=0.5, cmap="jet")

            # if no object points provided,extract from mask
            if obj_points is None:
                obj_points = self.pc[:, mask].T

        # Object points (preferred)
        if obj_points is not None and len(obj_points) > 0:
            points = obj_points

            # minimal filtering (already in object extraction)
            points = points[points[:, 2] > 0]

            if len(points) > 0:
                u, v = self._project(points)
                u, v = self._normalize_uv(u, v)

                ax.scatter(u, v, s=2, c="cyan", alpha=0.7)

        # Bounding box
        if box_idx is not None:
            box = self.boxes[box_idx]
            self._draw_box(box)

        # Title
        title = "Object View"
        if mask_idx is not None:
            title += f" | mask {mask_idx}"
        if box_idx is not None:
            title += f" | box {box_idx}"

        ax.set_title(title)
        ax.axis("off")

    # Predicted vs GT bbox
    def show_scene_predictions(
        self,
        gt_boxes=None,
        pred_boxes=None,
        show_pc=False,
        show_mask=False,
        show=False,
        save_path=None,
        ax=None,
    ):
        """
        Visualize full scene with GT and predicted boxes
        """

        if ax is None:
            plt.figure(figsize=(8, 6))
            ax = plt.gca()

        ax.imshow(self.img)

        # Optional mask
        if show_mask:
            ax.imshow(self.mask[0], alpha=0.3, cmap="jet")

        # Point cloud
        if show_pc:
            u, v = self._normalize_uv(self.u_all, self.v_all)
            z_norm = self._normalize_z(self.z_all)

            sc = ax.scatter(u, v, c=z_norm, cmap="jet", alpha=0.2)
            plt.colorbar(sc, ax=ax, label="Depth")

        # GT
        if gt_boxes is not None:
            for box in gt_boxes:
                self._draw_box(box, ax, color="g")

        # Prediction
        if pred_boxes is not None:
            for box in pred_boxes:
                self._draw_box(box, ax, color="r")

        # Legend
        ax.plot([], [], color="g", label="GT")
        ax.plot([], [], color="r", label="Prediction")
        ax.legend()

        ax.set_title("Scene: GT vs Prediction")
        ax.axis("off")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        if show:
            plt.show()

        plt.close()
