import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter

from .common import occlusion_sensitivity, replace2linear, summarize
from .lrp import build_folded_model_params, compute_relevance
from .visualization import prepare_rgb_image

LAYER_INDICES = [1, 5, 8, 12]
LRP_RULES = {
    "basic": {"epsilon": 1e-4},
    "epsilon": {"epsilon": 1e-4},
    "gamma": {"epsilon": 1e-4, "gamma": 0.01},
    "alphabeta": {"epsilon": 1e-4, "alpha": 1.0, "beta": 0.0},
}
METRICS = ["faithfulness", "robustness", "effective_complexity", "sensitivity_n", "selectivity"]


def compute_integrated_gradients(baseline, image, model, m_steps=200):
    image = tf.cast(tf.convert_to_tensor(image), tf.float32)
    baseline = tf.cast(tf.convert_to_tensor(baseline), tf.float32)
    target_class_idx = int(tf.argmax(model(image, training=False)[0]).numpy())
    gradients = []
    for alpha in tf.linspace(0.0, 1.0, m_steps + 1):
        interpolated = baseline + alpha * (image - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            score = model(interpolated, training=False)[:, target_class_idx]
        gradients.append(tape.gradient(score, interpolated))
    grads = tf.stack(gradients, axis=0)
    grads = (grads[:-1] + grads[1:]) / 2.0
    ig = (image - baseline) * tf.reduce_mean(grads, axis=0)
    return tf.squeeze(ig).numpy()


def get_balanced_sample(dataset, model, num_samples=2, threshold=0.7):
    samples = {i: [] for i in range(model.output_shape[-1])}
    for x_batch, _ in dataset:
        preds = model.predict(x_batch, verbose=0)
        for img, p in zip(x_batch, preds):
            cls = int(np.argmax(p))
            if p[cls] >= threshold and len(samples[cls]) < num_samples:
                samples[cls].append(img.numpy())
        if all(len(v) >= num_samples for v in samples.values()):
            break
    return [img for v in samples.values() for img in v]


class XAIEvaluator:
    def __init__(self, model, img_shape=(64, 64, 13), baseline_method="zeros", blur_sigma=3,
                 fractions=10, robustness_n=10, sensitivity_iters=5, selectivity_patches=10,
                 tol=1e-2, max_features=None):
        self.model = model
        self.img_shape = img_shape
        self.blur_sigma = blur_sigma
        self.baseline_method = baseline_method
        self.predict = lambda x: model(x).numpy()[0]
        self.explainers = {}
        self.current_explainer = None
        self.target = None
        self.fractions = np.linspace(0, 1, fractions)
        self.robustness_n = robustness_n
        self.sensitivity_iters = sensitivity_iters
        self.selectivity_patches = selectivity_patches
        self.tol = tol
        self.max_features = max_features

    @staticmethod
    def to_numpy(x):
        return x.numpy() if isinstance(x, tf.Tensor) else np.array(x)

    def register_explainer(self, name, explainer_fn, is_conservation=False, segments_fn=None):
        self.explainers[name] = (explainer_fn, is_conservation, segments_fn)

    def _normalize(self, expl):
        expl_abs = np.abs(expl)
        return (expl_abs - expl_abs.min()) / (expl_abs.max() - expl_abs.min() + 1e-9)

    def _get_baseline(self, img):
        arr = self.to_numpy(img)
        if self.baseline_method == "blur":
            return gaussian_filter(arr, sigma=self.blur_sigma)
        if self.baseline_method == "zeros":
            return np.zeros_like(arr)
        raise ValueError(f"Unknown baseline method: {self.baseline_method}")

    def faithfulness(self, expl, img, baseline, segments=None):
        p0 = self.predict(img)[self.target]
        expl_n = self._normalize(expl)
        img_np = self.to_numpy(img)
        scores = []
        if segments is None:
            idxs = np.argsort(expl_n.flatten())[::-1]
            for frac in self.fractions:
                k = int(frac * len(idxs))
                mod = self.to_numpy(baseline).copy()
                h, w, _ = np.unravel_index(idxs[:k], expl_n.shape)
                mod[0, h, w, :] = img_np[0, h, w, :]
                scores.append(p0 - self.predict(tf.convert_to_tensor(mod))[self.target])
        else:
            H, W, C = expl_n.shape
            flat_n = expl_n.reshape(-1, C)
            flat_segs = segments.flatten()
            conts = [(s, flat_n[flat_segs == s].sum()) for s in np.unique(flat_segs)]
            order = [s for s, _ in sorted(conts, key=lambda x: x[1], reverse=True)]
            baseline_np = self.to_numpy(baseline)
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
        orig = expl
        if orig.ndim == 3:
            orig = np.max(orig, axis=-1)
        img_np = self.to_numpy(img).astype(np.float32)
        noises = np.random.normal(0.0, noise_scale, (self.robustness_n,) + img_np.shape).astype(np.float32)
        perturbed_imgs = tf.convert_to_tensor(np.clip(img_np + noises, 0.0, 1.0), dtype=tf.float32)
        sims = []
        for i in range(self.robustness_n):
            nn = self._normalize(self.current_explainer[0](perturbed_imgs[i], baseline=baseline))
            if nn.ndim == 3:
                nn = np.max(nn, axis=-1)
            orig_tf = tf.convert_to_tensor(orig[..., None][None, ...], dtype=tf.float32)
            nn_tf = tf.convert_to_tensor(nn[..., None][None, ...], dtype=tf.float32)
            sims.append(tf.image.ssim(orig_tf, nn_tf, max_val=1.0).numpy()[0])
        return np.mean(sims)

    def effective_complexity(self, expl, img, baseline):
        p0 = self.predict(img)[self.target]
        tol_abs = max(abs(p0 * self.tol), 1e-6)
        flat = np.abs(expl).ravel()
        idxs = np.argsort(flat)[::-1]
        mf = min(self.max_features or flat.size, flat.size)
        img_np = self.to_numpy(img).astype(np.float32)
        img_f = img_np.ravel()
        base_f = self.to_numpy(baseline).astype(np.float32).ravel()
        modified = base_f.copy()

        def p_of(k):
            modified[:] = base_f
            modified[idxs[:k]] = img_f[idxs[:k]]
            inp = tf.convert_to_tensor(modified.reshape(img_np.shape), dtype=tf.float32)
            return self.predict(inp)[self.target]

        lo, hi, answer = 1, mf, mf
        while lo <= hi:
            mid = (lo + hi) // 2
            if abs(p0 - p_of(mid)) <= tol_abs:
                answer = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return answer

    def sensitivity_n(self, expl, img, baseline, sigma=0.05):
        arr = expl
        if arr.ndim == 3:
            arr = np.mean(arr, axis=-1)
        img_np = self.to_numpy(img).astype(np.float32)
        noise_batch = np.random.normal(0.0, sigma, (self.sensitivity_iters,) + img_np.shape).astype(np.float32)
        perturbed_imgs = np.clip(img_np + noise_batch, 0, 1)
        diffs = []
        for i in range(self.sensitivity_iters):
            pt = tf.convert_to_tensor(perturbed_imgs[i], dtype=tf.float32)
            try:
                ne = self.current_explainer[0](pt, baseline=baseline)
                if ne.ndim == 3:
                    ne = np.mean(ne, axis=-1)
                diffs.append(np.mean(np.abs(arr - ne)))
            except Exception as e:
                print(f"[sensitivity_n explainer error] {e}")
        return np.mean(diffs) if diffs else np.nan

    def selectivity(self, expl, img, baseline, patch_size=16):
        p0 = self.predict(img)[self.target]
        idxs = np.argsort(np.abs(expl).flatten())[::-1]
        area = patch_size * patch_size
        baseline_np = self.to_numpy(baseline).flatten()
        img_np = self.to_numpy(img).flatten()
        drops = []
        for k in range(1, self.selectivity_patches + 1):
            sel = idxs[:k * area]
            m = baseline_np.copy()
            m[sel] = img_np[sel]
            p = self.predict(tf.convert_to_tensor(m.reshape(self.to_numpy(img).shape)))[self.target]
            drops.append(abs(p0 - p))
        return np.mean(drops)

    def generate_explanations(self, images):
        cache = {name: [] for name in self.explainers}
        for i, img_arr in enumerate(images):
            img = tf.convert_to_tensor(img_arr)
            if img.ndim == 3:
                img = tf.expand_dims(img, axis=0)
            preds = self.predict(img)
            self.target = int(np.argmax(preds))
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

    sal = Saliency(evaluator.model, clone=True, model_modifier=replace2linear)
    evaluator.register_explainer("Saliency", lambda img, baseline=None: np.squeeze(
        sal(CategoricalScore(evaluator.target), img)), False)
    evaluator.register_explainer("SmoothGrad", lambda img, baseline=None: np.squeeze(
        sal(CategoricalScore(evaluator.target), img, smooth_samples=100, smooth_noise=0.4)), False)

    gradcam = Gradcam(evaluator.model, clone=True, model_modifier=replace2linear)
    evaluator.register_explainer("GradCAM", lambda img, baseline=None: np.squeeze(
        gradcam(CategoricalScore(evaluator.target), img, penultimate_layer=layer_index)), False)

    gradcam_pp = GradcamPlusPlus(evaluator.model, clone=True, model_modifier=replace2linear)
    scorecam = Scorecam(evaluator.model, clone=True, model_modifier=replace2linear)
    evaluator.register_explainer("GradCAM++", lambda img, baseline=None: np.squeeze(
        gradcam_pp(CategoricalScore(evaluator.target), img, penultimate_layer=layer_index)), False)
    evaluator.register_explainer("ScoreCAM", lambda img, baseline=None: np.squeeze(
        scorecam(CategoricalScore(evaluator.target), img, penultimate_layer=layer_index)), False)

    def ig_fn(img, baseline=None):
        return compute_integrated_gradients(evaluator._get_baseline(img), img, evaluator.model, m_steps=200)
    evaluator.register_explainer("IntegratedGradients", ig_fn, True)

    shap_expl = shap.GradientExplainer(evaluator.model, X_subset)
    evaluator.register_explainer("SHAP", lambda img, baseline=None: np.squeeze(
        shap_expl.shap_values(np.array(img), ranked_outputs=1)[0]), True)

    def seg_fn(img, **kw):
        return slic(prepare_rgb_image(img, bands_to_show=[4, 3, 2], eps=0),
                    n_segments=75, compactness=15, sigma=1.5, start_label=1)

    def lime_fn(img, baseline=None):
        lime_expl = lime_image.LimeImageExplainer(verbose=False)
        hide_color = np.mean(np.array(img)[0], axis=(0, 1))
        expln = lime_expl.explain_instance(np.array(img)[0], lambda x: evaluator.model.predict(x, verbose=0), top_labels=1,
                                           segmentation_fn=seg_fn, num_samples=200, hide_color=hide_color)
        _, lime_mask = expln.get_image_and_mask(expln.top_labels[0])
        return lime_mask
    evaluator.register_explainer("LIME", lime_fn, False, seg_fn)

    evaluator.register_explainer("Occlusion", lambda img, baseline=None: occlusion_sensitivity(
        evaluator.model, img, evaluator.target, patch_size=6, stride=6), False)

    folded_params, skip_bn = build_folded_model_params(evaluator.model)

    def make_lrp(rule, params):
        def wrapper(img, baseline=None):
            img_np = img.numpy() if hasattr(img, "numpy") else np.array(img)
            return compute_relevance(evaluator.model, img_np, evaluator.target, rule=rule,
                                     folded_params=folded_params, skip_bn=skip_bn, **params)[0]
        return wrapper

    for rule, params in lrp_rules.items():
        evaluator.register_explainer(f"LRP-{rule.capitalize()}", make_lrp(rule, params), True)
    return evaluator
