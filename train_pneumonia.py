import argparse
import os
from pathlib import Path

from model.cnn import build_pneumonia_model, pneumonia_callbacks
from model.loaders.pneumonia import class_weights_from_first_batch, load_datasets, rebalance_splits
from model.runtime import save_history, setup


def parse_args():
    p = argparse.ArgumentParser(description="Train the chest X-ray pneumonia CNN")
    p.add_argument("--data_dir", default="data/chest_xray")
    p.add_argument("--rebalanced_dir", default="data/chest_xray_rebalanced")
    p.add_argument("--model_out", default="models/pneumonia.keras")
    p.add_argument("--history_out", default="models/pneumonia_history.json")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=244)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    setup(args.seed)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    if not Path(args.rebalanced_dir).exists():
        rebalance_splits(args.data_dir, args.rebalanced_dir)

    image_size = (args.image_size, args.image_size)
    raw_train_ds, val_ds, _ = load_datasets(args.rebalanced_dir, image_size, args.batch_size)
    train_ds, class_weights = class_weights_from_first_batch(raw_train_ds)
    print(f"Class weights: {class_weights}")

    model = build_pneumonia_model((*image_size, 1))
    model.summary()
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=pneumonia_callbacks(args.model_out),
        class_weight=class_weights,
    )
    save_history(history, args.history_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
