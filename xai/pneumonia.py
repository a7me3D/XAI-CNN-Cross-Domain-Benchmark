import logging

import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

from .common import replace2linear, summarize
from .lrp import build_folded_model_params, compute_relevance

INPUT_SCALE = 255.0
LAYER_INDICES = [0, 4, 7, 11]
LRP_RULES = {
    "basic": {"epsilon": 1e-4},
    "epsilon": {"epsilon": 1e-4},
    "gamma": {"epsilon": 1e-4, "gamma": 0.01},
    "alphabeta": {"epsilon": 1e-4, "alpha": 1.0, "beta": 0.0},
}
METRICS = ["faithfulness", "robustness", "effective_complexity", "sensitivity_n", "selectivity"]


def _as_batch(x):
    x = tf.cast(tf.convert_to_tensor(x), tf.float32)
    while len(x.shape) > 4:
        x = tf.squeeze(x, axis=1)
    if len(x.shape) == 3:
        x = tf.expand_dims(x, 0)
    return x


def compute_integrated_gradients(baseline, image, model, m_steps=200):
    image = _as_batch(image)
    baseline = _as_batch(baseline)
    target_cls = bool(float(model(image, training=False).numpy().squeeze()) > 0.5)
    gradients = []
    for alpha in tf.linspace(0.0, 1.0, m_steps + 1):
        interpolated = baseline + alpha * (image - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            p = model(interpolated, training=False)
            score = p[:, 0] if target_cls else (1.0 - p[:, 0])
        gradients.append(tape.gradient(score, interpolated))
    grads = tf.stack(gradients, axis=0)
    grads = (grads[:-1] + grads[1:]) / 2.0
    ig = (image - baseline) * tf.reduce_mean(grads, axis=0)
    return tf.squeeze(ig).numpy()


def occlusion_sensitivity(model, img, patch_size=3, stride=4):
    original_prob = model(img, training=False).numpy()[0, 0]
    heatmap = np.zeros(img.shape[1:-1])
    for i in range(0, img.shape[1] - patch_size + 1, stride):
        for j in range(0, img.shape[2] - patch_size + 1, stride):
            occluded_img = np.array(img).copy()
            occluded_img[:, i:i + patch_size, j:j + patch_size, :] = 0
            heatmap[i:i + patch_size, j:j + patch_size] += (
                original_prob - model(occluded_img, training=False).numpy()[0, 0])
    return heatmap


def get_background_samples(dataset, background_size):
    background_samples = []
    class_counts = [0, 0]
    target_per_class = max(1, background_size // 2)
    for batch_imgs, batch_labels in dataset:
        batch_imgs = batch_imgs.numpy()
        batch_labels = batch_labels.numpy()
        for img, label in zip(batch_imgs, batch_labels):
            if img.ndim != 3 or img.shape[-1] != 1:
                continue
            label_idx = int(label.item()) if label.size == 1 else int(label)
            add_sample = False
            if len(background_samples) < background_size:
                if class_counts[0] + class_counts[1] == 0 or min(class_counts) < 1:
                    add_sample = True
                elif class_counts[label_idx] < target_per_class:
                    add_sample = True
            if add_sample:
                background_samples.append(img)
                class_counts[label_idx] += 1
                if len(background_samples) >= background_size and min(class_counts) >= 1:
                    break
        if len(background_samples) >= background_size and min(class_counts) >= 1:
            break
    return np.stack(background_samples, axis=0)[:background_size]


def get_balanced_sample(dataset, num_samples=10):
    samples_needed = {0: num_samples, 1: num_samples}
    images = []
    for x_batch, y_batch in dataset:
        x_batch_np = x_batch.numpy()
        y_batch_np = y_batch.numpy().flatten()
        for i in range(len(x_batch_np)):
            if samples_needed[0] <= 0 and samples_needed[1] <= 0:
                break
            class_idx = 1 if y_batch_np[i] > .7 else 0
            if samples_needed[class_idx] > 0:
                images.append(x_batch_np[i])
                samples_needed[class_idx] -= 1
        if samples_needed[0] <= 0 and samples_needed[1] <= 0:
            break
    for cls in [0, 1]:
        if samples_needed[cls] > 0:
            logging.warning(f"Class {cls}: Collected {num_samples - samples_needed[cls]}/{num_samples} samples")
    return np.array(images)


class XAIEvaluator:
    def __init__(self, model, fractions=10, robustness_n=10, sensitivity_iters=5,
                 selectivity_patches=10, tol=1e-2, max_features=None):
        self.model = model
        self.predict = lambda x: model(x).numpy().squeeze()
        self.fractions = np.linspace(0, 1, fractions)
        self.robustness_n = robustness_n
        self.sensitivity_iters = sensitivity_iters
        self.selectivity_patches = selectivity_patches
        self.tol = tol
        self.max_features = max_features
        self.explainers = {}
        self.current_explainer = None
        self.target = None

    def register_explainer(self, name, explainer_fn, is_conservation=False, segments_fn=None):
        self.explainers[name] = (explainer_fn, is_conservation, segments_fn)

    def _normalize(self, expl):
        expl_abs = np.abs(expl)
        return (expl_abs - expl_abs.min()) / (expl_abs.max() - expl_abs.min() + 1e-9)

    def _get_baseline(self, img):
        img_np = img.numpy() if hasattr(img, "numpy") else np.array(img)
        while img_np.ndim > 4:
            img_np = np.squeeze(img_np, axis=1)
        if img_np.ndim == 3:
            img_np = np.expand_dims(img_np, 0)
        prediction_class = round(float(self.model(tf.cast(img_np, tf.float32), training=False).numpy().squeeze()))
        mean_val = np.mean(img_np)
        if prediction_class == 1:
            baseline_intensity = max(0, mean_val - 0.1 * np.std(img_np))
        else:
            baseline_intensity = min(1, mean_val + 0.1 * np.std(img_np))
        lung_window = np.percentile(img_np, 95) - np.percentile(img_np, 5)
        sigma = 15 + (0.3 - lung_window) * 30 if lung_window < 0.3 else 15
        baseline = gaussian_filter(img_np, sigma=sigma) * 0.7 + baseline_intensity * 0.3
        return np.clip(baseline, 0, 1)

    def generate_explanations(self, images):
        cache = {name: [] for name in self.explainers}
        for i, img_arr in enumerate(images):
            img = tf.convert_to_tensor(np.expand_dims(img_arr, axis=0))
            self.target = int(round(float(self.predict(img))))
            baseline = self._get_baseline(img)
            print(f"Explaining image {i + 1}/{len(images)}")
            for name, (fn, _, _) in self.explainers.items():
                expl = fn(img, baseline=baseline)
                if expl.ndim == 2:
                    expl = np.repeat(expl[..., None], img.shape[-1], -1)
                cache[name].append((img, baseline, expl))
        return cache

    def faithfulness(self, expl, img, baseline, segments=None):
        p0 = self.predict(img)
        direction = 1 if p0 >= 0.5 else -1
        scores = []
        if segments is None:
            idxs = np.argsort(expl.flatten())[::-1]
            for frac in self.fractions:
                k = int(frac * len(idxs))
                mask = np.zeros_like(expl)
                mask.flat[idxs[:k]] = 1
                mod = np.where(mask, img, baseline)
                scores.append(direction * (p0 - self.predict(tf.convert_to_tensor(mod))))
        else:
            flat_n = expl.reshape(-1, expl.shape[-1])
            flat_segs = segments.flatten()
            conts = [(s, flat_n[flat_segs == s].sum()) for s in np.unique(flat_segs)]
            order = [s for s, _ in sorted(conts, key=lambda x: x[1], reverse=True)]
            base = baseline.copy()
            img_np = img.numpy()
            for frac in self.fractions:
                nseg = int(frac * len(order))
                mod = base.copy()
                mask2d = np.isin(segments, order[:nseg])
                for c in range(expl.shape[-1]):
                    mod[0, mask2d, c] = img_np[0, mask2d, c]
                scores.append(direction * (p0 - self.predict(tf.convert_to_tensor(mod))))
        return np.trapz(scores, self.fractions)

    def robustness(self, expl, img, baseline, noise_scale=0.1):
        orig = self._normalize(expl)
        if orig.ndim == 3:
            orig = np.max(orig, axis=-1)
        win = min(7, min(orig.shape[:2]))
        win = win if win % 2 == 1 else win - 1
        sims = []
        for _ in range(self.robustness_n):
            pert = tf.clip_by_value(img + np.random.normal(scale=noise_scale, size=img.shape), 0, 1)
            nn = self._normalize(self.current_explainer[0](pert, baseline=baseline))
            if nn.ndim == 3:
                nn = np.max(nn, axis=-1)
            sims.append(ssim(orig, nn, win_size=win, data_range=1.0))
        return np.mean(sims)

    def effective_complexity(self, expl, img, baseline):
        p0 = self.predict(img)
        flat = np.abs(expl).ravel()
        idxs = np.argsort(flat)[::-1]
        mf = min(self.max_features or flat.size, flat.size)
        tol_abs = max(abs(p0 * self.tol), 1e-6)
        base = baseline.copy().ravel()
        img_f = img.numpy().ravel()

        def p_of(k):
            m = base.copy()
            m[idxs[:k]] = img_f[idxs[:k]]
            return self.predict(tf.convert_to_tensor(m.reshape(img.shape)))

        lo, hi, ans = 1, mf, mf
        while lo <= hi:
            mid = (lo + hi) // 2
            if abs(p0 - p_of(mid)) <= tol_abs:
                ans, hi = mid, mid - 1
            else:
                lo = mid + 1
        return ans

    def sensitivity_n(self, expl, img, baseline, sigma=0.05):
        arr = expl.squeeze() if expl.ndim == 3 and expl.shape[-1] == 1 else np.mean(expl, axis=-1)
        diffs = []
        for _ in range(self.sensitivity_iters):
            pt = tf.convert_to_tensor(np.clip(img.numpy() + np.random.normal(0, sigma, img.shape), 0, 1))
            ne = self._normalize(self.current_explainer[0](pt, baseline=baseline))
            ne = ne.squeeze() if ne.ndim == 3 and ne.shape[-1] == 1 else np.mean(ne, axis=-1)
            diffs.append(np.mean(np.abs(arr - ne)))
        return np.mean(diffs)

    def selectivity(self, expl, img, baseline, patch_size=16):
        p0 = self.predict(img)
        direction = 1 if p0 >= 0.5 else -1
        idxs = np.argsort(np.abs(expl).flatten())[::-1]
        area = patch_size * patch_size
        drops = []
        for k in range(1, self.selectivity_patches + 1):
            sel = idxs[:k * area]
            m = baseline.copy().flatten()
            m[sel] = img.numpy().flatten()[sel]
            p = self.predict(tf.convert_to_tensor(m.reshape(img.shape)))
            drops.append(direction * (p0 - p))
        return np.mean(drops)

    def evaluate_metrics(self, cache, metrics=METRICS):
        rows_by_method = {}
        for name, items in cache.items():
            fn, is_cons, seg_fn = self.explainers[name]
            self.current_explainer = (fn, is_cons, seg_fn)
            rows = []
            for index, (img, baseline, expl) in enumerate(items):
                print(f"Metrics for {name} {index + 1}/{len(items)}")
                norm_expl = self._normalize(expl)
                row = {}
                for metric in metrics:
                    if metric == "faithfulness":
                        segments = seg_fn(img) if seg_fn else None
                        row[metric] = self.faithfulness(norm_expl, img, baseline, segments)
                    else:
                        row[metric] = getattr(self, metric)(norm_expl, img, baseline)
                rows.append(row)
            rows_by_method[name] = rows
        return summarize(rows_by_method)


def register_all(evaluator, X_subset, layer_index=LAYER_INDICES[-1], lrp_rules=LRP_RULES):
    import shap
    from lime import lime_image
    from skimage.segmentation import slic
    from tf_keras_vis.saliency import Saliency
    from tf_keras_vis.gradcam import Gradcam
    from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
    from tf_keras_vis.scorecam import Scorecam
    from tf_keras_vis.utils.scores import BinaryScore

    model = evaluator.model
    score = BinaryScore(True)

    sal = Saliency(model, clone=True, model_modifier=replace2linear)
    evaluator.register_explainer("Saliency", lambda img, baseline=None: np.squeeze(sal(score, img)), False)
    evaluator.register_explainer("SmoothGrad", lambda img, baseline=None: np.squeeze(
        sal(score, img, smooth_samples=400, smooth_noise=0.4)), False)

    gradcam = Gradcam(model, clone=True, model_modifier=replace2linear)
    gradcam_pp = GradcamPlusPlus(model, clone=True, model_modifier=replace2linear)
    scorecam = Scorecam(model, clone=True, model_modifier=replace2linear)
    evaluator.register_explainer("GradCAM", lambda img, baseline=None: np.squeeze(
        gradcam(score, img, penultimate_layer=layer_index)), False)
    evaluator.register_explainer("GradCAM++", lambda img, baseline=None: np.squeeze(
        gradcam_pp(score, img, penultimate_layer=layer_index)), False)
    evaluator.register_explainer("ScoreCAM", lambda img, baseline=None: np.squeeze(
        scorecam(score, img, penultimate_layer=layer_index)), False)

    def ig_fn(img, baseline=None):
        return compute_integrated_gradients(evaluator._get_baseline(img), img, model, m_steps=200)
    evaluator.register_explainer("IntegratedGradients", ig_fn, True)

    shap_expl = shap.GradientExplainer(model, X_subset)
    evaluator.register_explainer("SHAP", lambda img, baseline=None: np.squeeze(
        shap_expl.shap_values(np.array(img), ranked_outputs=1)[0]), True)

    def to_rgb(img):
        arr = np.array(img)
        arr = arr[0] if arr.ndim == 4 else arr
        return np.stack([np.squeeze(arr)] * 3, axis=-1)

    def seg_fn(img, **kw):
        return slic(to_rgb(img), n_segments=50, compactness=0.1, sigma=1)

    def lime_fn(img, baseline=None):
        lime_expl = lime_image.LimeImageExplainer(verbose=False)
        expln = lime_expl.explain_instance(np.array(img)[0], lambda x: model.predict(x, verbose=0), top_labels=1,
                                           segmentation_fn=seg_fn, num_samples=1500, hide_color=0)
        expln.image = to_rgb(img)
        _, lime_mask = expln.get_image_and_mask(label=expln.top_labels[0], positive_only=True, hide_rest=False)
        return lime_mask
    evaluator.register_explainer("LIME", lime_fn, False, seg_fn)

    evaluator.register_explainer("Occlusion", lambda img, baseline=None: occlusion_sensitivity(
        model, img, patch_size=20, stride=5), False)

    folded_params, skip_bn = build_folded_model_params(model, input_scale=INPUT_SCALE)

    def make_lrp(rule, params):
        def wrapper(img, baseline=None):
            img_np = img.numpy() if hasattr(img, "numpy") else np.array(img)
            target_cls = bool(model(tf.cast(img, tf.float32), training=False).numpy().squeeze() > 0.5)
            return compute_relevance(model, img_np, target_cls, rule=rule, folded_params=folded_params,
                                     skip_bn=skip_bn, input_scale=INPUT_SCALE, **params)[0]
        return wrapper

    for rule, params in lrp_rules.items():
        evaluator.register_explainer(f"LRP-{rule.capitalize()}", make_lrp(rule, params), True)
    return evaluator
