"""Includes functionality for loading config files."""

import os
import json

from functools import lru_cache


CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")


@lru_cache(maxsize=1)
def load_config():
    """Returns a dictionary loaded from the config.json file."""
    with open(os.path.join(CONFIG_DIR, "config.json")) as f:
        return json.load(f)


def load_model_config(name: str):
    """Returns a dictionary containing the model config."""
    model_config_path = os.path.join(CONFIG_DIR, "models", f"{name}.json")
    assert os.path.isfile(
        model_config_path
    ), f"Could not find config file for model={name} in config/models"
    with open(model_config_path) as f:
        return json.load(f)
