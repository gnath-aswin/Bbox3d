import mlflow
import matplotlib.pyplot as plt

run_id = "87cbf0afe757450f81110575e8c7346d"
client = mlflow.tracking.MlflowClient()

def get_metric(metric_name):
    history = client.get_metric_history(run_id, metric_name)
    steps = [m.step for m in history]
    values = [m.value for m in history]
    return steps, values

# Load metrics
train_steps, train_loss = get_metric("train_loss")
val_steps, val_loss = get_metric("val_loss")

# Plot
fig, ax = plt.subplots()

ax.plot(train_steps, train_loss, label="train_loss")
ax.plot(val_steps, val_loss, label="val_loss")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Curve")
ax.legend()

# Save
plt.savefig("loss_curve.png", dpi=150)
plt.show()