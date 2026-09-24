import argparse

import numpy as np
import tensorflow as tf

from model.loaders.leaf_disease import create_data_generators, create_dataframe_from_directory, load_background
from model.runtime import setup
from xai import leaf
from xai.runner import add_arguments, run_evaluation


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate XAI methods on the PlantVillage leaf-disease CNN")
    p.add_argument("--data_dir", default="data/PlantVillage")
    p.add_argument("--model", default="models/leaf.keras")
    p.add_argument("--out", default="results/leaf_metrics.csv")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=2, help="Evaluation images per class")
    p.add_argument("--background_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    add_arguments(p, "leaf")
    return p.parse_args()


def main():
    args = parse_args()
    setup(args.seed)

    train_df, val_df, test_df = create_dataframe_from_directory(args.data_dir)
    _, val_gen, _ = create_data_generators(args.data_dir, train_df, val_df, test_df, args.image_size)

    model = tf.keras.models.load_model(args.model)
    background = load_background(args.data_dir, train_df, args.background_size, args.image_size)
    images = leaf.get_balanced_sample(val_gen, num_samples=args.num_samples)
    print(f"Background: {background.shape} | Evaluation images: {len(images)}")

    evaluator = leaf.XAIEvaluator(model, img_shape=(args.image_size, args.image_size, 3), fractions=10,
                                  robustness_n=10, sensitivity_iters=5, selectivity_patches=10)
    leaf.register_all(evaluator, background)
    maps = None
    if args.save_maps:
        class_names = sorted(val_gen.class_indices, key=val_gen.class_indices.get)
        maps = dict(class_names=class_names, to_display=lambda img: np.clip(img, 0, 1), out_dir=args.maps_dir)
    run_evaluation(evaluator, images, args.metrics, args.methods, args.out, maps=maps)


if __name__ == "__main__":
    main()
