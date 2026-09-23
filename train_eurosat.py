import argparse
import json
import os

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import mixed_precision

from model.cnn import build_eurosat_model, eurosat_callbacks
from model.loaders.eurosat import EuroSATMultiTFLoader
from model.runtime import save_history, setup


def parse_args():
    p = argparse.ArgumentParser(description="Train the EuroSAT multispectral CNN")
    p.add_argument("--data_dir", default="data/EuroSATallBands")
    p.add_argument("--tfrecord_dir", default="data/tfrecords/eurosat")
    p.add_argument("--model_out", default="models/eurosat.keras")
    p.add_argument("--history_out", default="models/eurosat_history.json")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    setup(args.seed)
    mixed_precision.set_global_policy("mixed_float16")
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    with open(os.path.join(args.data_dir, "label_map.json")) as f:
        label_map = json.load(f)

    loaders = {
        split: EuroSATMultiTFLoader(
            csv_file=os.path.join(args.data_dir, f"{split}.csv"), data_dir=args.data_dir,
            batch_size=args.batch_size, augment=(split == "train"), tfrecord_dir=args.tfrecord_dir)
        for split in ["train", "validation", "test"]
    }
    train_ds = loaders["train"].get_dataset(shuffle=True)
    val_ds = loaders["validation"].get_dataset(shuffle=False)

    labels = loaders["train"].labels
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)

    model = build_eurosat_model((64, 64, 13), len(label_map))
    model.summary()
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        class_weight=dict(enumerate(class_weights)),
        callbacks=eurosat_callbacks(args.model_out),
    )
    save_history(history, args.history_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
