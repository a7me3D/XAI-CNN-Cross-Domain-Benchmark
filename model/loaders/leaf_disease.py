import os

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_dataframe_from_directory(directory, val_split=0.1, test_split=0.1):
    data = []
    classes = os.listdir(directory)
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        class_dir = os.path.join(directory, cls)
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                data.append([os.path.join(cls, file_name), class_to_id[cls], cls])

    df = pd.DataFrame(data, columns=["File", "DiseaseID", "Disease Type"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    total_len = len(df)
    val_len = int(val_split * total_len)
    test_len = int(test_split * total_len)
    train_len = total_len - val_len - test_len

    return df.iloc[:train_len], df.iloc[train_len:train_len + val_len], df.iloc[train_len + val_len:]


def create_image_data_generator(is_training):
    if is_training:
        return ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=(0.8, 1.2),
            fill_mode="nearest",
        )
    return ImageDataGenerator(rescale=1. / 255)


def create_data_generators(data_dir, train_df, val_df, test_df, image_size=64, batch_size=64):
    classes = sorted(train_df["Disease Type"].unique())

    def flow(df, is_training):
        return create_image_data_generator(is_training).flow_from_dataframe(
            directory=data_dir, dataframe=df, x_col="File", y_col="Disease Type",
            target_size=(image_size, image_size), batch_size=batch_size,
            classes=classes, class_mode="categorical", shuffle=is_training)

    return flow(train_df, True), flow(val_df, False), flow(test_df, False)


def preprocess_input(img_path, target_size=(64, 64)):
    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def load_background(data_dir, train_df, n=100, image_size=64):
    images = np.array([preprocess_input(os.path.join(data_dir, f), (image_size, image_size))
                       for f in train_df["File"][:n]])
    return images.reshape(-1, image_size, image_size, 3)
