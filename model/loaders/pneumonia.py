import collections
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight

CLASSES = ["NORMAL", "PNEUMONIA"]


def rebalance_splits(data_dir, output_dir, seed=42):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    all_files, all_labels = [], []
    for split in ["train", "val", "test"]:
        for class_dir in CLASSES:
            files = list((data_dir / split / class_dir).glob("*.jpeg"))
            all_files.extend(files)
            all_labels.extend([0 if class_dir == "NORMAL" else 1] * len(files))
    all_files = np.array(all_files)
    all_labels = np.array(all_labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val_idx, test_idx = next(sss.split(all_files, all_labels))
    train_val_files, test_files = all_files[train_val_idx], all_files[test_idx]
    train_val_labels, test_labels = all_labels[train_val_idx], all_labels[test_idx]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_idx, val_idx = next(sss.split(train_val_files, train_val_labels))
    train_files, val_files = train_val_files[train_idx], train_val_files[val_idx]
    train_labels, val_labels = train_val_labels[train_idx], train_val_labels[val_idx]

    for split in ["train", "val", "test"]:
        for class_dir in CLASSES:
            (output_dir / split / class_dir).mkdir(parents=True, exist_ok=True)

    for split, files, labels in [("train", train_files, train_labels),
                                 ("val", val_files, val_labels),
                                 ("test", test_files, test_labels)]:
        for file_path, label in zip(files, labels):
            shutil.copy(file_path, output_dir / split / CLASSES[label] / file_path.name)
    return output_dir


def process_grayscale(image, label):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def load_datasets(rebalanced_dir, image_size=(244, 244), batch_size=32):
    rebalanced_dir = Path(rebalanced_dir)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        rebalanced_dir / "train", image_size=image_size, batch_size=batch_size,
        label_mode="binary", shuffle=True)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        rebalanced_dir / "val", image_size=image_size, batch_size=batch_size, label_mode="binary")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        rebalanced_dir / "test", image_size=image_size, batch_size=batch_size,
        label_mode="binary", shuffle=False)
    return train_ds, val_ds.map(process_grayscale), test_ds.map(process_grayscale)


def class_weights_from_first_batch(raw_train_ds):
    first_label_batch = next(iter(raw_train_ds))[1]
    train_ds = raw_train_ds.map(process_grayscale)
    label_counts = collections.Counter()
    for _ in train_ds:
        label_counts.update(first_label_batch.numpy().flatten().astype(int))
    train_labels = np.array([1] * label_counts[1] + [0] * label_counts[0])
    weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
    return train_ds, {i: weight for i, weight in enumerate(weights)}


def preprocess_input(img_path, target_size=(244, 244)):
    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    image = tf.image.rgb_to_grayscale(img)
    image = tf.image.convert_image_dtype(image, tf.float32) * 255
    return np.expand_dims(image, axis=0)
