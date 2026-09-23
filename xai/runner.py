from pathlib import Path

from .common import completeness_table
from .visualization import save_attribution_maps

ALL_METRICS = ["faithfulness", "robustness", "effective_complexity", "sensitivity_n", "selectivity", "completeness"]


def add_arguments(parser, domain):
    parser.add_argument("--metrics", nargs="*", default=ALL_METRICS, choices=ALL_METRICS,
                        help="Metrics to compute; pass no values to skip the quantitative evaluation")
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of explainers to evaluate")
    parser.add_argument("--save_maps", action="store_true",
                        help="Save an attribution-map figure per evaluated image")
    parser.add_argument("--maps_dir", default=f"results/{domain}_maps")


def run_evaluation(evaluator, images, metrics, methods, out_path, binary=False, maps=None):
    if methods:
        unknown = set(methods) - set(evaluator.explainers)
        if unknown:
            raise ValueError(f"Unknown methods {sorted(unknown)}; available: {list(evaluator.explainers)}")
        evaluator.explainers = {k: v for k, v in evaluator.explainers.items() if k in methods}

    cache = evaluator.generate_explanations(images)
    if maps is not None:
        save_attribution_maps(evaluator.model, cache, **maps)
    if not metrics:
        return None

    results = evaluator.evaluate_metrics(cache, [m for m in metrics if m != "completeness"])
    if "completeness" in metrics:
        comp = completeness_table(evaluator.model, evaluator.explainers, cache, binary=binary)
        results = results.merge(comp, on="method", how="left")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(results.to_string(index=False))
    print(f"Results saved to {out_path}")
    return results
