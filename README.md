# Post-hoc XAI Benchmark for CNN Image Classifiers

Supplementary code for the paper:

> *A Cross-Domain Review and Benchmarking Framework for Post-hoc Explainability in CNN-based Image Classification*

The project compares 10 post-hoc explanation methods, quantitatively (6 metrics) and qualitatively (attribution
maps), on near-identical CNN models trained on three different image domains. Keeping the model essentially fixed
isolates how explanation quality depends on the method and the data domain. Each domain reproduces the
configuration used to produce the results reported in the paper.

| Case study | Data | Task | Input |
| --- | --- | --- | --- |
| EuroSAT | Sentinel-2 multispectral patches | 10-class land use (softmax) | 64×64×13 |
| Pneumonia | Pediatric chest X-rays | NORMAL vs PNEUMONIA (sigmoid) | 244×244×1 |
| Leaf disease | PlantVillage leaf images | 15-class disease (softmax) | 64×64×3 |

## Models

All three domains use the same backbone, defined in `model/cnn.py`: four Conv2D blocks (32, 64, 128 and 256
filters, 3×3 kernels, each followed by BatchNorm and 2×2 max-pooling, with dropout 0.2 after the first and third
blocks), global average pooling, a 64-unit dense layer with L2 regularization and dropout 0.5, and a classification
head. Only the input shape, the output head and the training setup follow each dataset:

| | EuroSAT | Pneumonia | Leaf disease |
| --- | --- | --- | --- |
| Conv padding | same | valid | valid |
| Output | 10-way softmax | 1 sigmoid unit | 15-way softmax |
| Loss | categorical CE, label smoothing 0.2 | binary CE | categorical CE |
| Optimizer | Adam, lr 1e-5, weight decay 1e-3 | Adam, lr 1e-3 | Adam, lr 1e-3 |
| Extra | activity L2 (1e-5), mixed precision, class weights, augmentation | class weights | augmentation |

## Explanation methods

The same 10 methods are evaluated in every domain:

| Method | Family |
| --- | --- |
| Saliency | Gradient |
| SmoothGrad | Gradient |
| Integrated Gradients | Gradient |
| Grad-CAM | CAM |
| Grad-CAM++ | CAM |
| Score-CAM | CAM |
| SHAP (GradientExplainer) | Perturbation |
| LIME | Perturbation |
| Occlusion | Perturbation |
| LRP, with Basic, Epsilon, Gamma and Alpha-Beta rules | Relevance |

## Quantitative evaluation

| Metric | Description |
| --- | --- |
| Faithfulness | Area under the prediction-change curve as the most relevant features are restored onto a baseline |
| Robustness | Mean SSIM between the explanation of an image and of Gaussian-perturbed copies |
| Effective complexity | Minimum number of top-ranked features that reproduce the prediction within tolerance |
| Sensitivity-n | Mean absolute change of the explanation under small input noise |
| Selectivity | Mean prediction change as the most relevant patches are progressively revealed |
| Completeness | Deviation of the attribution sum from the model output (conservative methods: IG, SHAP, LRP) |

## Qualitative evaluation

Attribution maps of every method are overlaid on the input image for side-by-side visual comparison. The
evaluation scripts save one figure per evaluated image with `--save_maps`, built from the same explanations the
metrics are computed on. The notebooks additionally show Grad-CAM/Grad-CAM++ across convolutional layers and maps
for confidently classified examples.

## Repository structure

```
├── data/README.md              dataset download and folder layout
├── model/
│   ├── cnn.py                  per-domain CNN architectures and training callbacks
│   ├── runtime.py              seeding and GPU setup
│   └── loaders/                eurosat.py, pneumonia.py, leaf_disease.py
├── xai/
│   ├── eurosat.py              EuroSAT evaluator, explainers and sampling
│   ├── pneumonia.py            pneumonia evaluator, explainers and sampling
│   ├── leaf.py                 leaf-disease evaluator, explainers and sampling
│   ├── lrp.py                  layer-wise relevance propagation (with Conv+BatchNorm folding)
│   ├── common.py               shared helpers and the completeness metric
│   ├── runner.py               evaluation driver writing metrics and attribution maps to results/
│   └── visualization.py        attribution-map figures and plotting helpers
├── notebooks/                  one end-to-end notebook per case study
├── train_{eurosat,pneumonia,leaf}.py
├── evaluate_{eurosat,pneumonia,leaf}.py
├── models/                     created by the training scripts
└── results/                    created by the evaluation scripts
```

## Reproducing the results

Tested with Python 3.12. A GPU is recommended for training.

```bash
pip install -r requirements.txt
```

**1. Get the data** into `data/` as described in [`data/README.md`](data/README.md).

**2. Train** (writes `models/<domain>.keras` and its training history):

```bash
python train_eurosat.py
python train_pneumonia.py
python train_leaf.py
```

**3. Evaluate** (writes `results/<domain>_metrics.csv`, and with `--save_maps` the attribution maps to
`results/<domain>_maps/`):

```bash
python evaluate_eurosat.py --save_maps
python evaluate_pneumonia.py --save_maps
python evaluate_leaf.py --save_maps
```

All paths default to `data/`, `models/` and `results/`; run any script with `--help` to override them. The full
evaluation is slow (LIME, Score-CAM and effective complexity dominate), so it can be split across runs, e.g.
`--metrics faithfulness robustness` or `--methods LIME SHAP --out results/eurosat_lime_shap.csv`. Passing
`--metrics` with no values produces only the attribution maps.

**4. Explore** — each notebook in `notebooks/` walks through data loading, classification performance,
attribution maps and the metric table for its case study, using the trained model from step 2.

## Citation

If you use this code, please cite this repository. Citation details are in [`CITATION.cff`](CITATION.cff), also
available through GitHub's "Cite this repository" button.

## License

Code is released under the MIT License. The datasets keep their own licenses, listed in
[`data/README.md`](data/README.md).
