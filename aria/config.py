"""Includes functionality for loading config.json"""

import json

CONFIG_PATH = "./aria/config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    return config
