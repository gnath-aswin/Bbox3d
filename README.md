# 3D Bounding Box Prediction

This project implements an end-to-end deep learning pipeline for **3D bounding box prediction** from point cloud data.

Current focus:
- Object-level point cloud processing
- PointNet-based regression model
- Training + validation pipeline
- Hyperparameter tuning integration (Optuna + MLflow)

---

## Training

```bash
uv run train\
  --data_path "/path/to/dataset" \
  --lr 1e-4 \
  --batch_size 64 \
  --epochs 200
```

```bash
uv run train\
  --data_path "/path/to/dataset" \
  --use_tuned \
  --lr 5e-4 \
  --batch_size 64 \
  --epochs 150 \
  --name tuned_run_v2
```


## Hyperparameter Tuning

Hyperparameter tuning is performed using **Optuna** with experiment tracking via **MLflow**.


```bash
uv run tune --config configs/tune.yaml
```

## Model Evaluation

Evaluate a trained model on the test split:

```bash
uv run test \
  --data_path "/path/to/dataset" \
  --checkpoint outputs/best_model.pth
```
With showing and saving visualization

```bash
uv run test \
  --data_path "/path/to/dataset" \
  --checkpoint outputs/best_model.pth \
  --show_vis \
  --save_vis
```

```markdown
### Example Predictions

![Image](outputs/vis/scene_1.png)
```