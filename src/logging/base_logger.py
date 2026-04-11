class BaseLogger:
    def log_params(self, params: dict): pass
    def log_metrics(self, metrics: dict, step: int): pass
    def log_artifact(self, path: str): pass