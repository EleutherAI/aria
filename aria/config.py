"""Includes functionality for loading config files."""

import os
import json

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../config/")


def load_config():
    with open(os.path.join(CONFIG_DIR, "config.json")) as f:
        return json.load(f)


def load_model_config(name: str):
    with open(os.path.join(CONFIG_DIR, f"{name}.json")) as f:
        return json.load(f)
