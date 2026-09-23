import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

from .common import occlusion_sensitivity, replace2linear, summarize
from .lrp import compute_relevance

LAYER_INDICES = [0, 4, 7, 11]
LRP_RULES = {
    "basic": {"epsilon": 1e-4},
    "epsilon": {"epsilon": 1e-4},
    "gamma": {"epsilon": 1e-4, "gamma": 0.25},
    "alphabeta": {"epsilon": 1e-4, "alpha": 1.0, "beta": 0.0},
}
METRICS = ["faithfulness", "robustness", "effective_complexity", "sensitivity_n", "selectivity"]


def compute_integrated_gradients(baseline, image, model, target_class, m_steps=50):
    image = tf.cast(image, tf.float32)
    grads = []
    for alpha in np.linspace(0.0, 1.0, m_steps + 1):
        x = tf.convert_to_tensor(baseline + alpha * (image - baseline), dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            target = model(x, training=False)[:, target_class]
        grads.append(tape.gradient(target, x).numpy())
    grads = np.mean(np.array(grads), axis=0)
    return np.squeeze((image - baseline) * grads)


def backprop_conv2d(activation, relevance, layer, rule="epsilon", epsilon=1e-4, gamma=0.25,
                    alpha=2.0, beta=1.0):
    activation = tf.cast(activation, tf.float32)
    relevance = tf.cast(relevance, tf.float32)
    W = tf.cast(layer.kernel, tf.float32)
    b = tf.reshape(tf.cast(layer.bias, tf.float32), [1, 1, 1, -1]) if layer.use_bias else 0.0
    strides = layer.strides
    padding = layer.padding.upper()

    if rule == "alphabeta":
        Wp = tf.maximum(W, 0.)
        Wn = tf.minimum(W, 0.)
        zp = tf.nn.conv2d(activation, Wp, strides=strides, padding=padding)
        zn = tf.nn.conv2d(activation, Wn, strides=strides, padding=padding)
        if layer.use_bias:
            zp += tf.maximum(b, 0.)
            zn += tf.minimum(b, 0.)
        zp_safe = tf.where(tf.abs(zp) < epsilon, tf.ones_like(zp) * epsilon, zp)
        zn_safe = tf.where(tf.abs(zn) < epsilon, -tf.ones_like(zn) * epsilon, zn)
        Rp = tf.nn.conv2d_transpose(relevance / zp_safe, Wp, output_shape=tf.shape(activation),
                                    strides=strides, padding=padding)
        Rn = tf.nn.conv2d_transpose(relevance / zn_safe, Wn, output_shape=tf.shape(activation),
                                    strides=strides, padding=padding)
        if bool(tf.reduce_any(activation < 0).numpy()):
            return activation * (alpha * Rp - beta * Rn)
        return activation * (alpha * Rp)

    if rule == "gamma":
        W = W + gamma * tf.maximum(W, 0.)
    z = tf.nn.conv2d(activation, W, strides=strides, padding=padding)
    if layer.use_bias:
        z += b
    if rule in ("epsilon", "basic"):
        z = z + epsilon * tf.where(z >= 0, tf.ones_like(z), -tf.ones_like(z))
    R = tf.nn.conv2d_transpose(relevance / z, W, output_shape=tf.shape(activation),
                               strides=strides, padding=padding)
    return activation * R


def backprop_batchnorm(activation, relevance, layer, rule="basic", epsilon=1e-4):
    activation = tf.cast(activation, tf.float32)
    relevance = tf.cast(relevance, tf.float32)
    scale = tf.cast(layer.gamma, tf.float32) / tf.sqrt(tf.cast(layer.moving_variance, tf.float32) + layer.epsilon)
    scale = tf.reshape(scale, [1, 1, 1, -1] if len(relevance.shape) == 4 else [1, -1])
    z = activation * scale
    z_safe = tf.where(tf.abs(z) < epsilon, tf.ones_like(z) * epsilon, z)
    return activation * (relevance / z_safe) * scale


def get_balanced_sample(generator, num_samples=20):
    samples_needed = {idx: num_samples for idx in generator.class_indices.values()}
    images = []
    for x_batch, y_batch in generator:
        for i in range(len(x_batch)):
            if not any(samples_needed.values()):
                break
            class_idx = int(np.argmax(y_batch[i]))
            if samples_needed.get(class_idx, 0) > 0:
                images.append(np.expand_dims(x_batch[i], axis=0))
                samples_needed[class_idx] -= 1
        if not any(samples_needed.values()):
            break
    return images


class XAIEvaluator:
    def __init__(self, model, img_shape=(64, 64, 3), baseline_method="zeros", blur_sigma=3,
                 fractions=10, robustness_n=10, sensitivity_iters=5, selectivity_patches=10,
                 tol=1e-2, max_features=None, batch_size=256):
        self.model = model
        self.img_shape = img_shape
        self.blur_sigma = blur_sigma
        self.baseline_method = baseline_method
        self.predict = lambda x: model(x).numpy()[0]
        self.explainers = {}
        self.current_explainer = None
        self.target = 0
        self.fractions = np.linspace(0, 1, fractions)
        self.robustness_n = robustness_n
        self.sensitivity_iters = sensitivity_iters
        self.selectivity_patches = selectivity_patches
        self.tol = tol
        self.max_features = max_features
        self.batch_size = batch_size

    def register_explainer(self, name, explainer_fn, is_conservation=False, segments_fn=None):
        self.explainers[name] = (explainer_fn, is_conservation, segments_fn)

    def _normalize(self, expl):
        expl_abs = np.abs(expl)
        return (expl_abs - expl_abs.min()) / (expl_abs.max() - expl_abs.min() + 1e-9)

    def _get_baseline(self, img):
        arr = img.numpy()
        if self.baseline_method == "blur":
            return gaussian_filter(arr, sigma=self.blur_sigma)
        if self.baseline_method == "zeros":
            return np.zeros_like(arr)
        raise ValueError(f"Unknown baseline method: {self.baseline_method}")

    def faithfulness(self, expl, img, baseline, segments=None):
        p0 = self.predict(img)[self.target]
        expl_n = self._normalize(expl)
        img_np = img.numpy()
        scores = []
        if segments is None:
            idxs = np.argsort(expl_n.flatten())[::-1]
            for frac in self.fractions:
                k = int(frac * len(idxs))
                mod = baseline.copy()
                h, w, c = np.unravel_index(idxs[:k], expl_n.shape)
                mod[0, h, w, c] = img_np[0, h, w, c]
                scores.append(p0 - self.predict(tf.convert_to_tensor(mod))[self.target])
        else:
            H, W, C = expl_n.shape
            flat_n = expl_n.reshape(-1, C)
            flat_segs = segments.flatten()
            conts = [(s, flat_n[flat_segs == s].sum()) for s in np.unique(flat_segs)]
            order = [s for s, _ in sorted(conts, key=lambda x: x[1], reverse=True)]
            baseline_np = baseline.numpy() if isinstance(baseline, tf.Tensor) else baseline.copy()
            for frac in self.fractions:
                nseg = int(frac * len(order))
                mod = baseline_np.copy()
                indices = np.where(np.isin(segments, order[:nseg]))
                rows, cols = indices[0], indices[1]
                for c in range(C):
                    mod[0, rows, cols, c] = img_np[0, rows, cols, c]
                scores.append(p0 - self.predict(tf.convert_to_tensor(mod))[self.target])
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
        p0 = self.predict(img)[self.target]
        idxs = np.argsort(np.abs(expl).flatten())[::-1]
        mf = min(self.max_features or idxs.size, idxs.size)
        tol_abs = max(abs(p0 * self.tol), 1e-6)
        img_f = img.numpy().flatten()
        base_f = baseline.flatten()
        rank = np.empty(idxs.size, dtype=np.int64)
        rank[idxs] = np.arange(idxs.size)
        for start in range(1, mf + 1, self.batch_size):
            ks = np.arange(start, min(start + self.batch_size, mf + 1))
            restored = rank[None, :] < ks[:, None]
            batch = np.where(restored, img_f[None, :], base_f[None, :])
            preds = self.model(tf.convert_to_tensor(batch.reshape((len(ks),) + tuple(img.shape[1:]))))
            hits = np.where(np.abs(p0 - preds.numpy()[:, self.target]) <= tol_abs)[0]
            if hits.size:
                return int(ks[hits[0]])
        return mf

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
        p0 = self.predict(img)[self.target]
        idxs = np.argsort(np.abs(expl).flatten())[::-1]
        area = patch_size * patch_size
        drops = []
        for k in range(1, self.selectivity_patches + 1):
            sel = idxs[:k * area]
            m = baseline.copy().flatten()
            m[sel] = np.array(img).flatten()[sel]
            p = self.predict(tf.convert_to_tensor(m.reshape(img.shape)))[self.target]
            drops.append(abs(p0 - p))
        return np.mean(drops)

    def generate_explanations(self, images):
        cache = {name: [] for name in self.explainers}
        for i, img_arr in enumerate(images):
            img = tf.convert_to_tensor(img_arr)
            self.target = int(np.argmax(self.predict(img)))
            baseline = self._get_baseline(img)
            print(f"Explaining image {i + 1}/{len(images)}")
            for name, (fn, _, _) in self.explainers.items():
                expl = fn(img, baseline=baseline)
                if expl.ndim == 2:
                    expl = np.repeat(expl[..., None], img.shape[-1], -1)
                cache[name].append((img, baseline, expl))
        return cache

    def evaluate_metrics(self, cache, metrics=METRICS):
        rows_by_method = {}
        for name, items in cache.items():
            fn, is_cons, seg_fn = self.explainers[name]
            self.current_explainer = (fn, is_cons, seg_fn)
            rows = []
            for index, (img, baseline, expl) in enumerate(items):
                print(f"Metrics for {name} {index + 1}/{len(items)}")
                self.target = int(np.argmax(self.predict(img)))
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
    from tf_keras_vis.utils.scores import CategoricalScore

    model = evaluator.model
    sal = Saliency(model, clone=True, model_modifier=replace2linear)
    evaluator.register_explainer("Saliency", lambda img, baseline=None: np.squeeze(
        sal(CategoricalScore(evaluator.target), img)), False)
    evaluator.register_explainer("SmoothGrad", lambda img, baseline=None: np.squeeze(
        sal(CategoricalScore(evaluator.target), img, smooth_samples=100, smooth_noise=0.4)), False)

    gradcam = Gradcam(model, clone=True, model_modifier=replace2linear)
    gradcam_pp = GradcamPlusPlus(model, clone=True, model_modifier=replace2linear)
    scorecam = Scorecam(model, clone=True, model_modifier=replace2linear)
    evaluator.register_explainer("GradCAM", lambda img, baseline=None: np.squeeze(
        gradcam(CategoricalScore(evaluator.target), img, penultimate_layer=layer_index)), False)
    evaluator.register_explainer("GradCAM++", lambda img, baseline=None: np.squeeze(
        gradcam_pp(CategoricalScore(evaluator.target), img, penultimate_layer=layer_index)), False)
    evaluator.register_explainer("ScoreCAM", lambda img, baseline=None: np.squeeze(
        scorecam(CategoricalScore(evaluator.target), img, penultimate_layer=layer_index)), False)

    def ig_fn(img, baseline=None):
        return compute_integrated_gradients(evaluator._get_baseline(img), img, model,
                                            target_class=evaluator.target, m_steps=200)
    evaluator.register_explainer("IntegratedGradients", ig_fn, True)

    shap_expl = shap.GradientExplainer(model, X_subset)
    evaluator.register_explainer("SHAP", lambda img, baseline=None: np.squeeze(
        shap_expl.shap_values(np.array(img), ranked_outputs=1)[0]), True)

    def seg_fn(img, **kw):
        return slic(img, n_segments=100, compactness=20, sigma=1)

    def lime_fn(img, baseline=None):
        explanation = lime_image.LimeImageExplainer(verbose=False).explain_instance(
            np.array(img)[0], lambda x: model.predict(x, verbose=0), segmentation_fn=seg_fn, num_samples=100)
        _, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=False)
        return mask
    evaluator.register_explainer("LIME", lime_fn, False, seg_fn)

    evaluator.register_explainer("Occlusion", lambda img, baseline=None: occlusion_sensitivity(
        model, img, evaluator.target, patch_size=6, stride=6), False)

    def make_lrp(rule, params):
        def wrapper(img, baseline=None):
            img_np = img.numpy() if hasattr(img, "numpy") else img
            return compute_relevance(model, img_np, evaluator.target, rule=rule, conv_fn=backprop_conv2d,
                                     batchnorm_fn=backprop_batchnorm, **params)[0]
        return wrapper

    for rule, params in lrp_rules.items():
        evaluator.register_explainer(f"LRP-{rule.capitalize()}", make_lrp(rule, params), True)
    return evaluator
