import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

try:
    import rasterio
except ImportError:
    rasterio = None


class EuroSATMultiTFLoader:
    def __init__(self, csv_file, data_dir, batch_size=64, img_size=(64, 64), num_bands=13,
                 class_map=None, augment=False, tfrecord_dir="."):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_bands = num_bands
        self.augment = augment
        self.num_classes = len(self.df["ClassName"].unique())
        self.csv_file = csv_file
        self.tfrecord_dir = tfrecord_dir

        if class_map is None:
            self.class_map = {name: idx for idx, name in enumerate(sorted(self.df["ClassName"].unique()))}
        else:
            self.class_map = class_map

        self.filepaths = [os.path.join(data_dir, fname) for fname in self.df["Filename"]]
        self.labels = [self.class_map[class_name] for class_name in self.df["ClassName"]]

        self.global_means, self.global_stds = self._compute_global_stats()
        self._create_tfrecords()

    def get_image_path(self, index):
        return self.filepaths[index]

    def _compute_global_stats(self, sample_size=500):
        sample_paths = np.random.choice(self.filepaths, size=min(sample_size, len(self.filepaths)), replace=False)
        all_pixels = []
        for path in tqdm(sample_paths, desc="Global stats"):
            with rasterio.open(path) as src:
                arr = src.read()
                all_pixels.append(arr.reshape(arr.shape[0], -1))
        full_data = np.concatenate(all_pixels, axis=1)
        means = np.mean(full_data, axis=1).astype(np.float32)
        stds = np.std(full_data, axis=1).astype(np.float32)
        return means, stds

    def _create_tfrecords(self):
        split_name = os.path.basename(self.csv_file).split(".")[0]
        os.makedirs(self.tfrecord_dir, exist_ok=True)
        self.tfrecord_path = os.path.join(self.tfrecord_dir, f"{split_name}.tfrecord")
        if os.path.exists(self.tfrecord_path):
            return

        with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
            for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"TFRecord {split_name}"):
                file_path = os.path.join(self.data_dir, row["Filename"])
                label_idx = self.class_map[row["ClassName"]]
                with rasterio.open(file_path) as src:
                    img = src.read().astype(np.float32)
                img = (img - self.global_means[:, None, None]) / (self.global_stds[:, None, None] + 1e-6)
                feature = {
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                    "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_idx])),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

    def _parse_tfrecord(self, example):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "shape": tf.io.FixedLenFeature([3], tf.int64),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature_description)
        img = tf.io.decode_raw(example["image"], tf.float32)
        img = tf.reshape(img, example["shape"])
        img = tf.transpose(img, [1, 2, 0])
        img = tf.image.resize(img, self.img_size)
        if self.augment:
            img = self._augment_image(img)
        label = tf.one_hot(example["label"], depth=self.num_classes)
        return img, label

    def _augment_image(self, img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.01)
        return img + noise

    def get_dataset(self, shuffle=True):
        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.filepaths), reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size)
        return dataset.prefetch(tf.data.AUTOTUNE)

    def print_dataset_info(self, dataset):
        print(f"Samples: {len(self.filepaths)} | Classes: {self.num_classes} | Batch size: {self.batch_size}")
        for images, labels in dataset.take(1):
            print(f"Image batch: {images.shape} | Label batch: {labels.shape}")


def preprocess_input(img_path, global_means, global_stds):
    with rasterio.open(img_path) as src:
        img = src.read().astype(np.float32)
    img = np.transpose(img, (1, 2, 0))
    img = (img - global_means[None, None, :]) / (global_stds[None, None, :] + 1e-6)
    return np.expand_dims(img, axis=0)
