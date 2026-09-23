import argparse
from pathlib import Path

import tensorflow as tf

from model.loaders.pneumonia import CLASSES, load_datasets, process_grayscale, rebalance_splits
from model.runtime import setup
from xai import pneumonia
from xai.runner import add_arguments, run_evaluation
from xai.visualization import prepare_rgb_image


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate XAI methods on the chest X-ray pneumonia CNN")
    p.add_argument("--data_dir", default="data/chest_xray")
    p.add_argument("--rebalanced_dir", default="data/chest_xray_rebalanced")
    p.add_argument("--model", default="models/pneumonia.keras")
    p.add_argument("--out", default="results/pneumonia_metrics.csv")
    p.add_argument("--image_size", type=int, default=244)
    p.add_argument("--num_samples", type=int, default=10, help="Evaluation images per class")
    p.add_argument("--background_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    add_arguments(p, "pneumonia")
    return p.parse_args()


def main():
    args = parse_args()
    setup(args.seed)

    if not Path(args.rebalanced_dir).exists():
        rebalance_splits(args.data_dir, args.rebalanced_dir)
    raw_train_ds, _, test_ds = load_datasets(args.rebalanced_dir, (args.image_size, args.image_size))
    train_ds = raw_train_ds.map(process_grayscale)

    model = tf.keras.models.load_model(args.model)
    background = pneumonia.get_background_samples(train_ds, args.background_size)
    images = pneumonia.get_balanced_sample(test_ds, num_samples=args.num_samples)
    print(f"Background: {background.shape} | Evaluation images: {len(images)}")

    evaluator = pneumonia.XAIEvaluator(model, fractions=10, robustness_n=10,
                                       sensitivity_iters=5, selectivity_patches=10)
    pneumonia.register_all(evaluator, background)
    maps = dict(class_names=CLASSES, to_display=prepare_rgb_image, out_dir=args.maps_dir) if args.save_maps else None
    run_evaluation(evaluator, images, args.metrics, args.methods, args.out, binary=True, maps=maps)


if __name__ == "__main__":
    main()
