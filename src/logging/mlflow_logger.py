import mlflow
from .base_logger import BaseLogger

class MLflowLogger(BaseLogger):
    def __init__(self, experiment_name="default"):
        mlflow.set_experiment(experiment_name)

    def start(self):
        if mlflow.active_run() is None:
            self.run = mlflow.start_run()
       
    def end(self):
        mlflow.end_run()

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics, step):
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)

    def log_artifact(self, path):
        mlflow.log_artifact(path)