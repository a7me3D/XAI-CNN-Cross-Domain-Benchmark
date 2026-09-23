import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def setup(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def save_history(history, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
