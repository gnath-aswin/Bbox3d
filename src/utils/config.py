import yaml
import json


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sample_from_config(trial, search_space):
    params = {}

    for name, cfg in search_space.items():
        if cfg["type"] == "float":
            low = float(cfg["low"])
            high = float(cfg["high"])
            params[name] = trial.suggest_float(name, low, high)

        elif cfg["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, cfg["choices"])

    return params


def load_best_params(path="best_params.json"):
    with open(path, "r") as f:
        return json.load(f)
