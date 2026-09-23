import argparse
import os

from model.cnn import build_leaf_model, leaf_callbacks
from model.loaders.leaf_disease import create_data_generators, create_dataframe_from_directory
from model.runtime import save_history, setup


def parse_args():
    p = argparse.ArgumentParser(description="Train the PlantVillage leaf-disease CNN")
    p.add_argument("--data_dir", default="data/PlantVillage")
    p.add_argument("--model_out", default="models/leaf.keras")
    p.add_argument("--history_out", default="models/leaf_history.json")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    setup(args.seed)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    train_df, val_df, test_df = create_dataframe_from_directory(args.data_dir)
    train_gen, val_gen, _ = create_data_generators(args.data_dir, train_df, val_df, test_df,
                                                   args.image_size, args.batch_size)

    model = build_leaf_model((args.image_size, args.image_size, 3), len(train_gen.class_indices))
    model.summary()
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_df) // args.batch_size,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=len(val_df) // args.batch_size,
        callbacks=leaf_callbacks(args.model_out),
    )
    save_history(history, args.history_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
