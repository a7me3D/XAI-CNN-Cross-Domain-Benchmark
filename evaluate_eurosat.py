import argparse
import json
import os
import warnings

import numpy as np

import tensorflow as tf
from tensorflow.keras import mixed_precision

from model.cnn import Float32L2
from model.loaders.eurosat import EuroSATMultiTFLoader
from model.runtime import setup
from xai import eurosat
from xai.common import get_background_samples
from xai.runner import add_arguments, run_evaluation
from xai.visualization import band_stretch_ranges, fixed_normalize

warnings.filterwarnings("ignore", message="TIFFReadDirectory: Warning, Unknown field with tag 42112")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate XAI methods on the EuroSAT multispectral CNN")
    p.add_argument("--data_dir", default="data/EuroSATallBands")
    p.add_argument("--tfrecord_dir", default="data/tfrecords/eurosat")
    p.add_argument("--model", default="models/eurosat.keras")
    p.add_argument("--out", default="results/eurosat_metrics.csv")
    p.add_argument("--num_samples", type=int, default=2, help="Evaluation images per class")
    p.add_argument("--threshold", type=float, default=0.7, help="Minimum confidence of evaluation images")
    p.add_argument("--background_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    add_arguments(p, "eurosat")
    return p.parse_args()


def main():
    args = parse_args()
    setup(args.seed)
    mixed_precision.set_global_policy("mixed_float16")

    loaders = {
        split: EuroSATMultiTFLoader(
            csv_file=os.path.join(args.data_dir, f"{split}.csv"), data_dir=args.data_dir,
            augment=(split == "train"), tfrecord_dir=args.tfrecord_dir)
        for split in ["train", "validation", "test"]
    }
    train_ds = loaders["train"].get_dataset(shuffle=True)
    test_ds = loaders["test"].get_dataset(shuffle=False)

    model = tf.keras.models.load_model(args.model, custom_objects={"Float32L2": Float32L2})
    background = get_background_samples(train_ds, args.background_size)
    images = eurosat.get_balanced_sample(test_ds, model, num_samples=args.num_samples, threshold=args.threshold)
    print(f"Background: {background.shape} | Evaluation images: {len(images)}")

    evaluator = eurosat.XAIEvaluator(model, fractions=10, robustness_n=10,
                                     sensitivity_iters=5, selectivity_patches=10)
    eurosat.register_all(evaluator, background)
    maps = None
    if args.save_maps:
        with open(os.path.join(args.data_dir, "label_map.json")) as f:
            label_map = json.load(f)
        lo, hi = band_stretch_ranges(loaders["validation"].get_dataset(shuffle=False))
        maps = dict(class_names=sorted(label_map, key=label_map.get), out_dir=args.maps_dir,
                    to_display=lambda img: fixed_normalize(np.asarray(img)[..., [3, 2, 1]], lo, hi))
    run_evaluation(evaluator, images, args.metrics, args.methods, args.out, maps=maps)


if __name__ == "__main__":
    main()
