"""Training engine and CLI for the deep classifiers (PRD C3, C4, C8, C9).

AdamW, cosine annealing and early stopping on validation macro-F1. A checkpoint is written
every epoch (last.pt) and on every improvement (best.pt), so a timed-out Colab session
continues with --resume. After training, a temperature is fitted on validation logits. The
test split is scored only with --eval-test, once per final model.

    python -m roadcond.train --model small_cnn --root /path/to/RSCD
    python -m roadcond.train --model resnet50 --root /path/to/RSCD --eval-test --gradcam 4
"""

import argparse
import copy
import json
import logging
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from roadcond.calibration import (
    TemperatureScaler,
    expected_calibration_error,
    nll,
    reliability_bins,
    save_reliability_plot,
)
from roadcond.config import load_config, resolve_path
from roadcond.data.dataset import RoadDataset, build_transforms, make_loader
from roadcond.data.rscd import Record, load_splits
from roadcond.metrics import accuracy, classification_report, format_table, macro_f1, mean_std
from roadcond.models.pretrained import (
    build_pretrained,
    count_parameters,
    freeze_batchnorm_stats,
    set_backbone_trainable,
    split_parameters,
)
from roadcond.models.small_cnn import SmallCNN

log = logging.getLogger(__name__)

MODEL_NAMES = ("small_cnn", "resnet50", "efficientnet")


class EarlyStopper:
    """Tracks the best validation score; should_stop once patience epochs pass without gain."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.best_epoch = -1
        self.bad_epochs = 0

    def update(self, score: float, epoch: int) -> bool:
        """Record an epoch's score; True if it is a new best."""
        if score > self.best + self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = score, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "best_epoch": self.best_epoch, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best, self.best_epoch, self.bad_epochs = (
            state["best"],
            state["best_epoch"],
            state["bad_epochs"],
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def build_model(
    name: str, cfg: dict[str, Any], pretrained: bool = True, freeze_backbone: bool = False
) -> nn.Module:
    """Build one of MODEL_NAMES with the class count and settings from the config."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; use one of {MODEL_NAMES}")
    n_classes = len(cfg["rscd"]["classes"])
    mcfg = cfg["classifier"]["models"][name]
    if name == "small_cnn":
        return SmallCNN(n_classes, dropout=mcfg["dropout"])
    return build_pretrained(mcfg["timm_name"], n_classes, pretrained, freeze_backbone)


def make_optimizer(model: nn.Module, mcfg: dict[str, Any], weight_decay: float):
    """AdamW; pretrained models get a smaller learning rate on the backbone than the head."""
    if "backbone_lr" in mcfg:
        head, backbone = split_parameters(model)
        groups = [
            {"params": head, "lr": mcfg["lr"]},
            {"params": backbone, "lr": mcfg["backbone_lr"]},
        ]
    else:
        groups = [{"params": list(model.parameters()), "lr": mcfg["lr"]}]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    """Write via a temporary file, so a disconnect mid-write cannot corrupt the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def predict_logits(
    model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """(logits (N, C) float32, labels (N,)) on the CPU, in loader order."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=amp):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mcfg: dict[str, Any],
    ccfg: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    freeze_epochs: int = 0,
    criterion: nn.Module | None = None,
    resume: bool = False,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train with early stopping on validation macro-F1; returns history and the best epoch.

    The first freeze_epochs epochs train only the head (BatchNorm statistics stay fixed).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    criterion = criterion or nn.CrossEntropyLoss()
    epochs = mcfg["epochs"]
    amp = bool(ccfg["amp"]) and device.type == "cuda"
    model.to(device)
    optimizer = make_optimizer(model, mcfg, ccfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    grad_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    stopper = EarlyStopper(mcfg["patience"])
    history: list[dict[str, Any]] = []
    start = 0

    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        grad_scaler.load_state_dict(state["grad_scaler"])
        stopper.load_state_dict(state["stopper"])
        history, start = state["history"], state["epoch"] + 1
        log.info("Resumed from %s at epoch %d", last_path, start)

    for epoch in range(start, epochs):
        if stopper.should_stop:
            break
        started = time.time()
        frozen = epoch < freeze_epochs
        if freeze_epochs:
            set_backbone_trainable(model, not frozen)
        model.train()
        if frozen:
            freeze_batchnorm_stats(model)

        lr = optimizer.param_groups[0]["lr"]
        total, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = criterion(model(x), y)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            total += loss.item() * len(y)
            seen += len(y)
        scheduler.step()

        logits, labels = predict_logits(model, val_loader, device, amp)
        preds = logits.argmax(dim=1).numpy()
        entry = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "val_loss": float(F.cross_entropy(logits, labels)),
            "val_macro_f1": macro_f1(labels.numpy(), preds, logits.shape[1]),
            "val_accuracy": accuracy(labels.numpy(), preds),
            "lr": lr,
            "frozen_backbone": frozen,
            "seconds": time.time() - started,
        }
        history.append(entry)
        improved = stopper.update(entry["val_macro_f1"], epoch)
        log.info(
            "epoch %d/%d train %.4f val %.4f macro-F1 %.4f acc %.4f%s (%.0fs)",
            epoch + 1,
            epochs,
            entry["train_loss"],
            entry["val_loss"],
            entry["val_macro_f1"],
            entry["val_accuracy"],
            " *" if improved else "",
            entry["seconds"],
        )
        if improved:
            best = {"model": model.state_dict(), "epoch": epoch, "val_macro_f1": stopper.best}
            save_checkpoint({**best, **(meta or {})}, best_path)
        save_checkpoint(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "grad_scaler": grad_scaler.state_dict(),
                "stopper": stopper.state_dict(),
                "history": history,
                "epoch": epoch,
            },
            last_path,
        )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {
        "history": history,
        "best_epoch": stopper.best_epoch,
        "best_val_macro_f1": stopper.best,
        "stopped_early": stopper.should_stop,
        "best_path": best_path,
    }


def load_best(model: nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    """Load the best checkpoint's weights into model; returns the checkpoint metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}; train first")
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return {k: v for k, v in state.items() if k != "model"}


def save_loss_curves(history: Sequence[dict[str, Any]], path: Path) -> None:
    from matplotlib.figure import Figure

    epochs = [h["epoch"] + 1 for h in history]
    fig = Figure(figsize=(9, 3.5))
    loss_ax, f1_ax = fig.subplots(1, 2)
    loss_ax.plot(epochs, [h["train_loss"] for h in history], label="train")
    loss_ax.plot(epochs, [h["val_loss"] for h in history], label="validation")
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("loss")
    loss_ax.legend()
    f1_ax.plot(epochs, [h["val_macro_f1"] for h in history], color="tab:green")
    f1_ax.set_xlabel("epoch")
    f1_ax.set_ylabel("validation macro-F1")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)


def evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    classes: Sequence[str],
    scaler: TemperatureScaler,
    n_bins: int,
) -> dict[str, Any]:
    """Classification report plus calibration before and after temperature scaling."""
    report = classification_report(labels.numpy(), logits.argmax(dim=1).numpy(), classes)
    with torch.no_grad():
        scaled = scaler(logits)
    probs = {"before": F.softmax(logits, dim=1).numpy(), "after": F.softmax(scaled, dim=1).numpy()}
    bins = {k: reliability_bins(p, labels.numpy(), n_bins) for k, p in probs.items()}
    report["calibration"] = {
        "temperature": scaler.temperature,
        "ece_before": expected_calibration_error(probs["before"], labels.numpy(), n_bins),
        "ece_after": expected_calibration_error(probs["after"], labels.numpy(), n_bins),
        "nll_before": nll(logits, labels),
        "nll_after": nll(scaled, labels),
        "reliability_before": bins["before"],
        "reliability_after": bins["after"],
    }
    return report


def run_seed(
    cfg: dict[str, Any],
    model_name: str,
    seed: int,
    splits: dict[str, list[Record]],
    root: Path,
    out_dir: Path,
    device: torch.device,
    eval_test: bool = False,
    resume: bool = False,
    pretrained: bool = True,
    gradcam_n: int = 0,
) -> dict[str, Any]:
    """Train one model with one seed, calibrate it, and score validation (and test)."""
    classes = cfg["rscd"]["classes"]
    ccfg = cfg["classifier"]
    mcfg = ccfg["models"][model_name]
    size = tuple(cfg["rscd"]["image_size"])
    amp = bool(ccfg["amp"]) and device.type == "cuda"
    seed_everything(seed)

    eval_tf = build_transforms(False, size)
    train_tf = build_transforms(True, size, ccfg["augment_strength"])
    names = ("train", "val", "test") if eval_test else ("train", "val")
    datasets = {
        s: RoadDataset(splits[s], root, classes, train_tf if s == "train" else eval_tf)
        for s in names
    }
    loaders = {
        s: make_loader(
            ds, ccfg["batch_size"], shuffle=s == "train", num_workers=ccfg["num_workers"], seed=seed
        )
        for s, ds in datasets.items()
    }

    freeze_epochs = ccfg["freeze_epochs"] if "backbone_lr" in mcfg else 0
    model = build_model(model_name, cfg, pretrained, freeze_backbone=freeze_epochs > 0)
    run_dir = out_dir / "checkpoints" / f"{model_name}_seed{seed}"
    log.info(
        "%s seed %d: %d parameters, %d train images",
        model_name,
        seed,
        count_parameters(model),
        len(datasets["train"]),
    )
    fitted = fit(
        model,
        loaders["train"],
        loaders["val"],
        mcfg,
        ccfg,
        device,
        run_dir,
        freeze_epochs=freeze_epochs,
        resume=resume,
        meta={"model_name": model_name, "classes": list(classes)},
    )
    load_best(model, fitted["best_path"], device)
    save_loss_curves(fitted["history"], run_dir / "loss_curves.png")

    val_logits, val_labels = predict_logits(model, loaders["val"], device, amp)
    scaler = TemperatureScaler().fit(val_logits, val_labels)
    run: dict[str, Any] = {
        "seed": seed,
        "best_epoch": fitted["best_epoch"],
        "epochs_run": len(fitted["history"]),
        "stopped_early": fitted["stopped_early"],
        "temperature": scaler.temperature,
        "checkpoint": str(fitted["best_path"]),
        "val": evaluate_logits(val_logits, val_labels, classes, scaler, ccfg["ece_bins"]),
    }
    shown, shown_logits = "val", val_logits
    if eval_test:
        test_logits, test_labels = predict_logits(model, loaders["test"], device, amp)
        run["test"] = evaluate_logits(test_logits, test_labels, classes, scaler, ccfg["ece_bins"])
        shown, shown_logits = "test", test_logits

    cal = run[shown]["calibration"]
    save_reliability_plot(
        cal["reliability_before"],
        cal["reliability_after"],
        cal["ece_before"],
        cal["ece_after"],
        run_dir / f"reliability_{shown}.png",
        title=f"{model_name} seed {seed} ({shown})",
    )
    if gradcam_n > 0:
        with torch.no_grad():
            probs = F.softmax(scaler(shown_logits), dim=1).numpy()
        from roadcond.explain import save_gallery

        run["gradcam"] = save_gallery(
            model.to(device),
            datasets[shown],
            probs,
            run_dir / f"gradcam_{shown}.png",
            n_each=gradcam_n,
            seed=seed,
        )
    return run


def summary_tables(results: dict[str, Any]) -> str:
    """Headline table (macro-F1, accuracy, calibration) and per-class precision / recall."""
    split = "test" if results["eval_test"] else "val"
    headers = ["model", "val macro-F1"]
    if split == "test":
        headers.append("test macro-F1")
    headers += [f"{split} accuracy", f"{split} ECE before -> after", "temperature", "best epochs"]
    rows, per_class_rows = [], {c: [c] for c in results["classes"]}
    for name, runs in results["models"].items():
        row = [name, mean_std([r["val"]["macro_f1"] for r in runs])]
        if split == "test":
            row.append(mean_std([r["test"]["macro_f1"] for r in runs]))
        before = np.mean([r[split]["calibration"]["ece_before"] for r in runs])
        after = np.mean([r[split]["calibration"]["ece_after"] for r in runs])
        row += [
            mean_std([r[split]["accuracy"] for r in runs]),
            f"{before:.3f} -> {after:.3f}",
            f"{np.mean([r['temperature'] for r in runs]):.2f}",
            [r["best_epoch"] + 1 for r in runs],
        ]
        rows.append(row)
        for c in results["classes"]:
            p = np.mean([r[split]["per_class"][c]["precision"] for r in runs])
            rec = np.mean([r[split]["per_class"][c]["recall"] for r in runs])
            per_class_rows[c].append(f"{p:.3f} / {rec:.3f}")

    n = len(results["seeds"])
    return "\n\n".join(
        [
            f"Deep classifiers, mean ± std over {n} seeds ({split} split):",
            format_table(headers, rows),
            f"Per-class precision / recall on {split}, mean over seeds:",
            format_table(
                ["class", *[f"{m} P / R" for m in results["models"]]], list(per_class_rows.values())
            ),
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a surface classifier.")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--root", help="RSCD folder (default: rscd.root in the config)")
    parser.add_argument("--splits-dir", help="split files folder (default: paths.splits)")
    parser.add_argument("--out-dir", help="checkpoints and results folder (default: artifacts)")
    parser.add_argument("--seeds", type=int, nargs="+", help="default: project.seeds")
    parser.add_argument("--epochs", type=int, help="override the model's epochs")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, help="dataloader workers")
    parser.add_argument("--freeze-epochs", type=int, help="head-only epochs for pretrained models")
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="random init (no download)")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt")
    parser.add_argument(
        "--gradcam",
        type=int,
        default=0,
        metavar="N",
        help="draw N correct and N wrong Grad-CAM examples per seed",
    )
    parser.add_argument("--limit", type=int, help="debug: use at most N images per split")
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="score the final models on the test split (do this once)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = copy.deepcopy(load_config(args.config))
    ccfg, mcfg = cfg["classifier"], cfg["classifier"]["models"][args.model]
    if args.epochs is not None:
        mcfg["epochs"] = args.epochs
    for key, value in (
        ("batch_size", args.batch_size),
        ("num_workers", args.workers),
        ("freeze_epochs", args.freeze_epochs),
    ):
        if value is not None:
            ccfg[key] = value

    cwd = Path.cwd()

    def pick(arg: str | None, default: str | Path) -> Path:
        return resolve_path(arg, cwd) if arg else resolve_path(default)

    root = pick(args.root, cfg["rscd"]["root"])
    out_dir = pick(args.out_dir, cfg["paths"]["artifacts"])
    splits = load_splits(pick(args.splits_dir, cfg["paths"]["splits"]))
    if args.limit:
        splits = {s: random.Random(0).sample(r, min(args.limit, len(r))) for s, r in splits.items()}
    device = select_device(args.device)
    seeds = args.seeds or cfg["project"]["seeds"]
    log.info("Device %s, model %s, seeds %s", device, args.model, seeds)

    runs = [
        run_seed(
            cfg,
            args.model,
            seed,
            splits,
            root,
            out_dir,
            device,
            eval_test=args.eval_test,
            resume=args.resume,
            pretrained=not args.no_pretrained,
            gradcam_n=args.gradcam,
        )
        for seed in seeds
    ]
    results = {
        "classes": cfg["rscd"]["classes"],
        "seeds": seeds,
        "eval_test": args.eval_test,
        "config": {"classifier": {k: v for k, v in ccfg.items() if k != "models"}, **mcfg},
        "models": {args.model: runs},
    }
    results_path = out_dir / "results" / f"{args.model}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(summary_tables(results))
    print(f"\nSaved {results_path}")


if __name__ == "__main__":
    main()
