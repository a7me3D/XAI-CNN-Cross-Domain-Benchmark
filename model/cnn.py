import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D,
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


@tf.keras.utils.register_keras_serializable(package="xai_benchmark")
class Float32L2(tf.keras.regularizers.Regularizer):
    def __init__(self, l2=1e-5):
        self.l2 = l2

    def __call__(self, x):
        return self.l2 * tf.reduce_sum(tf.square(tf.cast(x, tf.float32)))

    def get_config(self):
        return {"l2": self.l2}


def build_eurosat_model(input_shape, num_classes):
    model = Sequential([
        tf.keras.layers.Input(shape=input_shape),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(),
        GlobalAveragePooling2D(),
        Dense(64, activation="relu", kernel_regularizer=l2(1e-3), activity_regularizer=Float32L2(1e-5)),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=1e-3),
        metrics=["accuracy"],
    )
    return model


def eurosat_callbacks(checkpoint_path):
    return [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_delta=0.001,
                          mode="min", min_lr=1e-7, verbose=1),
        EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=15, verbose=1,
                      mode="max", restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", mode="max",
                        save_best_only=True, verbose=1),
    ]


def _valid_padding_cnn(input_shape, output_units, output_activation):
    return Sequential([
        tf.keras.layers.Input(shape=input_shape),
        Conv2D(32, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(256, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(),
        GlobalAveragePooling2D(),
        Dense(64, activation="relu", kernel_regularizer=l2(1e-3)),
        Dropout(0.5),
        Dense(output_units, activation=output_activation),
    ])


def build_pneumonia_model(input_shape):
    model = _valid_padding_cnn(input_shape, 1, "sigmoid")
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"])
    return model


def pneumonia_callbacks(checkpoint_path):
    return [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", mode="max",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, verbose=1, min_lr=1e-8),
    ]


def build_leaf_model(input_shape, num_classes):
    model = _valid_padding_cnn(input_shape, num_classes, "softmax")
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"])
    return model


def leaf_callbacks(checkpoint_path):
    return [
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=1, min_lr=1e-8, mode="min"),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", mode="max",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ]
