import numpy as np
import pandas as pd
import tensorflow as tf


def replace2linear(model):
    new_model = tf.keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())
    for layer in new_model.layers:
        if isinstance(layer, tf.keras.layers.Activation):
            layer.activation = tf.keras.activations.linear
    return new_model


def get_background_samples(dataset, background_size):
    samples = []
    for batch_imgs, _ in dataset:
        for img in batch_imgs:
            samples.append(img.numpy())
            if len(samples) >= background_size:
                break
        if len(samples) >= background_size:
            break
    return np.stack(samples, axis=0)


def occlusion_sensitivity(model, img, target_class_idx, patch_size=3, stride=4):
    heatmap = np.zeros(img.shape[1:-1])
    predictions = model(img, training=False)[0].numpy()
    for i in range(0, img.shape[1] - patch_size + 1, stride):
        for j in range(0, img.shape[2] - patch_size + 1, stride):
            occluded_img = np.array(img).copy()
            occluded_img[:, i:i + patch_size, j:j + patch_size, :] = 0
            occluded = model(occluded_img, training=False)[0].numpy()
            heatmap[i:i + patch_size, j:j + patch_size] += predictions[target_class_idx] - occluded[target_class_idx]
    return heatmap


def _as_batch(x):
    t = tf.cast(tf.convert_to_tensor(x), tf.float32)
    while len(t.shape) > 4:
        t = tf.squeeze(t, axis=1)
    if len(t.shape) == 3:
        t = tf.expand_dims(t, 0)
    return t


def completeness(model, expl, img, baseline, method_name, binary=False):
    p_in = model(_as_batch(img), training=False).numpy().squeeze()
    p_b = model(_as_batch(baseline), training=False).numpy().squeeze()
    attribution_sum = float(np.sum(expl))
    if binary:
        p_in, p_b = float(p_in), float(p_b)
        target = p_in > 0.5
        if method_name.startswith("LRP"):
            reference = p_in if target else 1.0 - p_in
        else:
            reference = p_in - p_b if target else (1.0 - p_in) - (1.0 - p_b)
    else:
        target = int(np.argmax(p_in))
        if method_name.startswith("LRP"):
            reference = float(p_in[target])
        else:
            reference = float(p_in[target] - p_b[target])
    if abs(reference) < 1e-9:
        return abs(attribution_sum)
    return abs(attribution_sum - reference) / (abs(reference) + 1e-9)


def completeness_table(model, explainers, cache, binary=False):
    records = []
    for name, items in cache.items():
        if not explainers[name][1]:
            continue
        scores = [completeness(model, expl, img, baseline, name, binary) for img, baseline, expl in items]
        records.append({"method": name, "completeness_mean": np.mean(scores),
                        "completeness_std": np.std(scores)})
    return pd.DataFrame(records, columns=["method", "completeness_mean", "completeness_std"])


def summarize(rows_by_method):
    records = []
    for name, rows in rows_by_method.items():
        df = pd.DataFrame(rows)
        summary = {"method": name}
        for c in df.columns:
            summary[f"{c}_mean"] = df[c].mean()
            summary[f"{c}_std"] = df[c].std()
        records.append(summary)
    return pd.DataFrame(records)
