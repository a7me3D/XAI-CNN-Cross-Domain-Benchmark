from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def normalize_grads(grads, symmetric=True, eps=1e-10):
    mn, mx = grads.min(), grads.max()
    if symmetric:
        return grads / (max(abs(mn), abs(mx)) + eps)
    return grads / (mx + eps)


def apply_colormap(grads, normalization=True):
    if grads.ndim == 3:
        grads = np.mean(grads, axis=-1)
    symmetric = grads.min() < 0
    grads_n = normalize_grads(grads, symmetric=symmetric) if normalization else grads
    if symmetric:
        grads_n = (grads_n + 1) / 2
    return mpl.colormaps["RdBu_r" if symmetric else "Reds"](grads_n)


def overlay_gradients(img, grads, alpha=0.6, normalization=True):
    grads_rgb = apply_colormap(grads, normalization)[..., :3]
    img_f = np.asarray(img).astype(np.float32)
    if img_f.max() > 1.0:
        img_f /= 255.0
    blended = img_f * (1 - alpha) + grads_rgb * alpha
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


def prepare_rgb_image(img, bands_to_show=None, percentiles=(1, 99), eps=1e-9):
    if isinstance(img, tf.Tensor):
        img = img.numpy()
    if bands_to_show is not None:
        rgb = img[..., list(bands_to_show)].astype(np.float32)
    elif img.shape[-1] == 1:
        rgb = np.repeat(img.astype(np.float32), 3, axis=-1)
    else:
        rgb = img.astype(np.float32)
    out = np.zeros_like(rgb)
    for i in range(3):
        band = rgb[..., i]
        v_min, v_max = np.percentile(band, percentiles)
        out[..., i] = np.clip((band - v_min) / (v_max - v_min + eps), 0, 1)
    return (out * 255).round().astype(np.uint8)


def band_stretch_ranges(dataset, bands=(3, 2, 1), max_samples=500):
    values = {b: [] for b in bands}
    count = 0
    for batch_imgs, _ in dataset:
        batch = batch_imgs.numpy()
        for b in bands:
            values[b].append(batch[..., b].ravel())
        count += batch.shape[0]
        if count > max_samples:
            break
    lo = {b: np.percentile(np.concatenate(v), 2) for b, v in values.items()}
    hi = {b: np.percentile(np.concatenate(v), 98) for b, v in values.items()}
    return lo, hi


def fixed_normalize(img_rgb, lo_map, hi_map, bands=(3, 2, 1), gamma=1.0):
    out = np.zeros_like(img_rgb, dtype=np.float32)
    for i, band_idx in enumerate(bands):
        lo, hi = lo_map[band_idx], hi_map[band_idx]
        out[..., i] = (np.clip(img_rgb[..., i], lo, hi) - lo) / (hi - lo + 1e-6)
    if gamma != 1.0:
        out = out ** (1.0 / gamma)
    return np.clip(out, 0, 1)


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["accuracy"], label="Train")
    ax1.plot(history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy")
    ax1.legend()
    ax2.plot(history["loss"], label="Train")
    ax2.plot(history["val_loss"], label="Validation")
    ax2.set_title("Loss")
    ax2.legend()
    plt.tight_layout()
    plt.show()


def plot_cam_layers(model, img_array, score, layer_indices, display_img, model_modifier):
    from tf_keras_vis.gradcam import Gradcam
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

    methods = [(Gradcam, "Grad-CAM"), (GradcamPlusPlus, "Grad-CAM++")]
    fig, axes = plt.subplots(len(layer_indices), len(methods),
                             figsize=(4 * len(methods), 4 * len(layer_indices)), squeeze=False)
    for row, layer in enumerate(layer_indices):
        for col, (cam_cls, name) in enumerate(methods):
            cam = cam_cls(model, clone=True, model_modifier=model_modifier)
            heatmap = cam(score, img_array, penultimate_layer=layer)[0]
            axes[row, col].imshow(overlay_gradients(display_img, heatmap))
            axes[row, col].set_title(f"{name} — layer {layer}", fontsize=10, fontweight="bold")
            axes[row, col].axis("off")
    plt.tight_layout()
    plt.show()


def attribution_grid(display_img, attributions, title, n_cols=7):
    n_rows = -(-(len(attributions) + 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
    axes = axes.ravel()
    fig.suptitle(title, fontsize=13, fontweight="bold")
    axes[0].imshow(display_img)
    axes[0].set_title("Input", fontsize=10)
    for ax, (name, attr) in zip(axes[1:], attributions.items()):
        attr = np.squeeze(attr).astype(np.float32)
        if attr.ndim == 3:
            attr = np.mean(np.abs(attr), axis=-1)
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-9)
        ax.imshow(display_img)
        ax.imshow(attr, cmap="jet", alpha=0.5)
        ax.set_title(name, fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_attribution_maps(evaluator, img, target, display_img, methods, title, n_cols=7):
    evaluator.target = target
    batch = tf.convert_to_tensor(img[None, ...] if img.ndim == 3 else img)
    baseline = evaluator._get_baseline(batch)
    attributions = {name: evaluator.explainers[name][0](batch, baseline=baseline) for name in methods}
    attribution_grid(display_img, attributions, title, n_cols)
    plt.show()


def save_attribution_maps(model, cache, class_names, to_display, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = list(cache)
    for i in range(len(cache[methods[0]])):
        img = cache[methods[0]][i][0]
        probs = np.asarray(model(img, training=False)).reshape(-1)
        if probs.size == 1:
            pred, conf = int(probs[0] > 0.5), max(probs[0], 1 - probs[0])
        else:
            pred, conf = int(np.argmax(probs)), probs.max()
        title = f"Image {i:03d} — predicted {class_names[pred]} ({conf:.2f})"
        fig = attribution_grid(to_display(np.asarray(img)[0]), {m: cache[m][i][2] for m in methods}, title)
        fig.savefig(out_dir / f"image_{i:03d}_{class_names[pred]}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Attribution maps saved to {out_dir}")
