"""Few-shot fine-tuning on target labels (PRD D3) and the learning curve (D5).

The source detector is fine-tuned on 10, 50 and 200 labelled images of a target country, over
several seeds. The recipe is fixed up front and never tuned on target data: the final weights
(last.pt) are scored on that country's held-out test split. Each run writes its own result
file, so an interrupted Colab session continues where it stopped.
"""

import json
import logging
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from roadcond.data.rdd import annotated_names, read_name_list, slug, write_name_list
from roadcond.detection.metrics import learning_curve
from roadcond.detection.train_yolo import (
    Env,
    evaluate_detector,
    last_weights,
    results_dir,
    save_json,
    split_images,
    subset_yaml,
    train_detector,
    tree_paths,
)

log = logging.getLogger(__name__)


def sample_budgets(
    names: Sequence[str], budgets: Sequence[int], seed: int, nested: bool = True
) -> dict[int, list[str]]:
    """Seeded image subsets for each label budget; nested means 10 within 50 within 200."""
    ordered = sorted(names)
    for n in budgets:
        if n > len(ordered):
            raise ValueError(f"Budget {n} is larger than the {len(ordered)} annotated pool images")
    if nested:
        shuffled = list(ordered)
        random.Random(seed).shuffle(shuffled)
        return {n: sorted(shuffled[:n]) for n in budgets}
    return {n: sorted(random.Random(f"{seed}-{n}").sample(ordered, n)) for n in budgets}


def fewshot_epochs(n_images: int, batch: int, epochs: int, min_iterations: int) -> int:
    """Epochs for a budget: the base value, raised so even tiny budgets get enough updates."""
    per_epoch = math.ceil(n_images / batch)
    return max(epochs, math.ceil(min_iterations / per_epoch))


def run_few_shot(
    env: Env,
    source_weights: str | Path,
    countries: Sequence[str] | None = None,
    budgets: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Fine-tune and score every country x budget x seed; returns the result dicts."""
    rdd = env.cfg["rdd2022"]
    few = rdd["few_shot"]
    countries = list(countries or rdd["target_countries"])
    budgets = [b for b in (budgets if budgets is not None else rdd["label_budgets"]) if b > 0]
    seeds = list(seeds if seeds is not None else few["seeds"])

    runs = []
    for country in countries:
        pool = read_name_list(env.splits_dir / f"rdd_{slug(country)}_pool.txt")
        annotated = annotated_names(env.work_dir, country, pool)
        val_images = split_images(env, country, "val")
        test_images = split_images(env, country, "test")
        for seed in seeds:
            subsets = sample_budgets(annotated, budgets, seed, few["nested"])
            for n in budgets:
                name = f"fewshot_{slug(country)}_n{n}_s{seed}"
                result_path = results_dir(env) / "fewshot" / f"{name}.json"
                if result_path.is_file() and not force:
                    runs.append(json.loads(result_path.read_text()))
                    continue
                names = subsets[n]
                write_name_list(
                    env.splits_dir / f"rdd_{slug(country)}_fewshot_n{n}_seed{seed}.txt", names
                )
                data = subset_yaml(
                    env, name, tree_paths(env, country, names), val_images, test_images
                )
                batch = min(few["batch"], n)
                epochs = fewshot_epochs(n, batch, few["epochs"], few["min_iterations"])
                log.info("%s: %d images, %d epochs", name, n, epochs)
                train_detector(
                    env,
                    name,
                    source_weights,
                    data,
                    epochs=epochs,
                    batch=batch,
                    seed=seed,
                    val=False,  # no model selection on target data: the final weights are scored
                    optimizer="AdamW",
                    lr0=few["lr0"],
                    freeze=few["freeze"],
                    force=force,
                )
                scores = evaluate_detector(env, last_weights(env, name), data, "test", name)
                result = {
                    "country": country,
                    "budget": n,
                    "seed": seed,
                    "epochs": epochs,
                    "batch": batch,
                    "images": names,
                    **scores,
                }
                save_json(result_path, result)
                runs.append(result)
    return runs


def load_runs(env: Env) -> list[dict[str, Any]]:
    """Every saved few-shot result, sorted by country, budget and seed."""
    folder = results_dir(env) / "fewshot"
    runs = (
        [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
        if folder.is_dir()
        else []
    )
    return sorted(runs, key=lambda r: (r["country"], r["budget"], r["seed"]))


def build_curve(env: Env, source_name: str = "source") -> dict[str, Any] | None:
    """Learning curve (mean ± std over seeds) from the saved source and few-shot results."""
    path = results_dir(env) / f"detection_{source_name}.json"
    if not path.is_file():
        return None
    source_run = json.loads(path.read_text())
    source_country = source_run["source_country"]
    runs = load_runs(env)
    countries = {r["country"] for r in runs}
    zero_shot = {c: source_run["countries"][c] for c in source_run["countries"] if c in countries}
    if not zero_shot:
        return None
    return learning_curve(runs, zero_shot, source_run["countries"][source_country])


def save_learning_curve_plot(
    curve: dict[str, Any], source_value: float | None, path: Path, metric: str = "map50"
) -> None:
    """mAP against label budget, one panel per target country, error bars = std over seeds."""
    from matplotlib.figure import Figure

    countries = list(curve)
    fig = Figure(figsize=(4.2 * len(countries), 3.6))
    axes = fig.subplots(1, len(countries), sharey=True, squeeze=False)[0]
    for ax, country in zip(axes, countries, strict=True):
        budgets = sorted(curve[country])
        means = [curve[country][b][metric]["mean"] for b in budgets]
        stds = [curve[country][b][metric]["std"] or 0.0 for b in budgets]
        ax.errorbar(range(len(budgets)), means, yerr=stds, marker="o", capsize=3, label="target")
        if source_value is not None:
            ax.axhline(source_value, color="gray", linestyle="--", label="source country")
        ax.set_xticks(range(len(budgets)), [str(b) for b in budgets])
        ax.set_xlabel("labelled target images")
        ax.set_title(country)
    axes[0].set_ylabel("mAP@0.5" if metric == "map50" else "mAP@0.5:0.95")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
