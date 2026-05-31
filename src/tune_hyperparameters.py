from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "Optuna is required for Bayesian/TPE hyperparameter tuning. "
        "Install it with: pip install optuna"
    ) from exc

import config as _project_config
from config import ExperimentConfig, KNOWN_CLASSES, SEED

print(f"Imported config module: {_project_config.__file__}")
try:
    from .datasets import (
        PadUfesClosedSetDataset,
        DomainAdaptationDataset,
        get_loss_weights_from_train_df,
        make_soft_weighted_random_sampler,
    )
    from .metrics import evaluate_closed_set
    from .models import build_model, build_dann_model
    from .preprocessing import (
        prepare_padufes_dataframe,
        stratified_train_val_test_split,
        load_standardized_splits,
        add_known_unknown_columns,
    )
    from .transforms import get_train_transform, get_eval_transform
    from .utils import get_device, seed_everything
except ImportError:
    from datasets import (
        PadUfesClosedSetDataset,
        DomainAdaptationDataset,
        get_loss_weights_from_train_df,
        make_soft_weighted_random_sampler,
    )
    from metrics import evaluate_closed_set
    from models import build_model, build_dann_model
    from preprocessing import (
        prepare_padufes_dataframe,
        stratified_train_val_test_split,
        load_standardized_splits,
        add_known_unknown_columns,
    )
    from transforms import get_train_transform, get_eval_transform
    from utils import get_device, seed_everything


def metric_value(metrics: dict, *names: str, default: float = float("nan")) -> float:
    for name in names:
        if name in metrics:
            return float(metrics[name])
    return default


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    losses = []

    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        out = model(
            batch["pixel_values"],
            batch.get("input_ids"),
            batch.get("attention_mask"),
        )
        logits = out[0] if isinstance(out, tuple) else out
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


def build_closed_set_objective(args, cfg: ExperimentConfig):
    seed_everything(SEED)
    device = get_device()

    df = prepare_padufes_dataframe(args.padufes_csv, args.padufes_image_dir)
    train_df, val_df, _ = stratified_train_val_test_split(df)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = replace(
            cfg,
            lr=trial.suggest_float("lr", args.lr_low, args.lr_high, log=True),
            weight_decay=trial.suggest_float("weight_decay", args.weight_decay_low, args.weight_decay_high, log=True),
            fusion_dim=trial.suggest_categorical("fusion_dim", args.fusion_dims),
            balance_beta=trial.suggest_float("balance_beta", 0.90, 0.999),
        )

        model_family = trial.suggest_categorical("model_family", args.model_families)
        text_col = trial.suggest_categorical("text_col", args.text_cols)
        batch_size = trial.suggest_categorical("batch_size", args.batch_sizes)

        if "cross_attention" in model_family:
            num_heads = trial.suggest_categorical("num_heads", args.num_heads_choices)
        else:
            num_heads = cfg.num_heads

        image_model_name = (
            trial_cfg.resnet_model_name
            if model_family.startswith("resnet50")
            else trial_cfg.image_model_name
        )

        train_ds = PadUfesClosedSetDataset(
            train_df,
            tokenizer,
            get_train_transform(),
            text_col,
            max_text_len=trial_cfg.max_text_len,
        )
        val_ds = PadUfesClosedSetDataset(
            val_df,
            tokenizer,
            get_eval_transform(),
            text_col,
            max_text_len=trial_cfg.max_text_len,
        )

        sampler = (
            make_soft_weighted_random_sampler(train_df, beta=trial_cfg.balance_beta)
            if args.use_sampler
            else None
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=trial_cfg.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=trial_cfg.num_workers,
        )

        model = build_model(
            model_family,
            image_model_name,
            trial_cfg.text_model_name,
            len(KNOWN_CLASSES),
            trial_cfg.fusion_dim,
            num_heads,
            getattr(trial_cfg, "freeze_backbones", False),
        ).to(device)

        loss_weights = (
            get_loss_weights_from_train_df(train_df, len(KNOWN_CLASSES), trial_cfg.balance_beta).to(device)
            if args.use_weighted_loss
            else None
        )
        criterion = nn.CrossEntropyLoss(weight=loss_weights)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trial_cfg.lr,
            weight_decay=trial_cfg.weight_decay,
        )

        best_score = -1.0
        for epoch in range(1, args.trial_epochs + 1):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics, _, _, _ = evaluate_closed_set(
                model,
                val_loader,
                criterion,
                device,
                len(KNOWN_CLASSES),
            )
            score = metric_value(val_metrics, "macro_f1", "f1_macro")
            best_score = max(best_score, score)

            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return float(best_score)

    return objective


def dann_lambda_schedule(step: int, total_steps: int, max_lambda: float = 1.0) -> float:
    p = step / max(total_steps, 1)
    return float(max_lambda * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))).item() - 1.0))


def train_dann_epoch_once(
    model,
    source_loader,
    target_loader,
    optimizer,
    class_criterion,
    domain_criterion,
    device,
    epoch: int,
    total_epochs: int,
    domain_loss_weight: float,
    max_lambda: float,
):
    model.train()
    n_steps = min(len(source_loader), len(target_loader))
    total_steps = n_steps * total_epochs

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    losses = []
    for local_step in range(n_steps):
        src = next(source_iter)
        tgt = next(target_iter)

        src = {k: v.to(device) if hasattr(v, "to") else v for k, v in src.items()}
        tgt = {k: v.to(device) if hasattr(v, "to") else v for k, v in tgt.items()}

        lambd = dann_lambda_schedule((epoch - 1) * n_steps + local_step, total_steps, max_lambda)

        optimizer.zero_grad(set_to_none=True)

        src_class_logits, src_domain_logits, _ = model(
            src["pixel_values"],
            src.get("input_ids"),
            src.get("attention_mask"),
            dann_lambda=lambd,
        )
        _, tgt_domain_logits, _ = model(
            tgt["pixel_values"],
            tgt.get("input_ids"),
            tgt.get("attention_mask"),
            dann_lambda=lambd,
        )

        class_loss = class_criterion(src_class_logits, src["label"])
        domain_logits = torch.cat([src_domain_logits, tgt_domain_logits], dim=0)
        domain_labels = torch.cat([src["domain"], tgt["domain"]], dim=0)
        domain_loss = domain_criterion(domain_logits, domain_labels)

        loss = class_loss + domain_loss_weight * domain_loss
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


def build_dann_objective(args, cfg: ExperimentConfig):
    seed_everything(SEED)
    device = get_device()

    if args.standardized_csv is None:
        raise ValueError("--standardized-csv is required for --study dann.")

    source_df = load_standardized_splits(
        args.standardized_csv,
        args.source_image_roots,
        args.source_dataset,
    )
    target_df = load_standardized_splits(
        args.standardized_csv,
        args.target_image_roots,
        args.target_dataset,
    )

    source_df = add_known_unknown_columns(source_df)
    target_df = add_known_unknown_columns(target_df)

    source_train_df = source_df[
        (source_df["split"].isin(["source_train", "train"]))
        & (~source_df["is_unknown"])
    ].copy()

    target_adapt_df = target_df[
        (target_df["split"] == "target_adapt")
        & (~target_df["is_unknown"])
    ].copy()

    target_val_known_df = target_df[
        (target_df["split"] == "target_val")
        & (~target_df["is_unknown"])
    ].copy()

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = replace(
            cfg,
            lr=trial.suggest_float("lr", args.lr_low, args.lr_high, log=True),
            weight_decay=trial.suggest_float("weight_decay", args.weight_decay_low, args.weight_decay_high, log=True),
            fusion_dim=trial.suggest_categorical("fusion_dim", args.fusion_dims),
            balance_beta=trial.suggest_float("balance_beta", 0.90, 0.999),
            domain_loss_weight=trial.suggest_float("domain_loss_weight", 0.1, 2.0),
            dann_lambda_max=trial.suggest_float("dann_lambda_max", 0.1, 1.5),
        )

        model_family = trial.suggest_categorical("model_family", args.model_families)
        text_col = trial.suggest_categorical("text_col", args.text_cols)
        batch_size = trial.suggest_categorical("batch_size", args.batch_sizes)

        if "cross_attention" in model_family:
            num_heads = trial.suggest_categorical("num_heads", args.num_heads_choices)
        else:
            num_heads = cfg.num_heads

        image_model_name = (
            trial_cfg.resnet_model_name
            if model_family.startswith("resnet50")
            else trial_cfg.image_model_name
        )

        source_ds = DomainAdaptationDataset(
            source_train_df,
            tokenizer,
            get_train_transform(),
            text_col,
            domain_label=0,
            max_text_len=trial_cfg.max_text_len,
        )
        target_ds = DomainAdaptationDataset(
            target_adapt_df,
            tokenizer,
            get_train_transform(),
            text_col,
            domain_label=1,
            max_text_len=trial_cfg.max_text_len,
        )
        val_ds = DomainAdaptationDataset(
            target_val_known_df,
            tokenizer,
            get_eval_transform(),
            text_col,
            domain_label=1,
            max_text_len=trial_cfg.max_text_len,
        )

        source_loader = DataLoader(
            source_ds,
            batch_size=batch_size,
            sampler=make_soft_weighted_random_sampler(source_train_df, beta=trial_cfg.balance_beta),
            num_workers=trial_cfg.num_workers,
        )
        target_loader = DataLoader(
            target_ds,
            batch_size=batch_size,
            sampler=make_soft_weighted_random_sampler(target_adapt_df, beta=trial_cfg.balance_beta),
            num_workers=trial_cfg.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=trial_cfg.num_workers,
        )

        model = build_dann_model(
            model_family,
            image_model_name,
            trial_cfg.text_model_name,
            len(KNOWN_CLASSES),
            trial_cfg.fusion_dim,
            num_heads,
            getattr(trial_cfg, "freeze_backbones", False),
        ).to(device)

        class_weights = (
            get_loss_weights_from_train_df(source_train_df, len(KNOWN_CLASSES), trial_cfg.balance_beta).to(device)
            if args.use_weighted_loss
            else None
        )
        class_criterion = nn.CrossEntropyLoss(weight=class_weights)
        domain_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trial_cfg.lr,
            weight_decay=trial_cfg.weight_decay,
        )

        best_score = -1.0
        for epoch in range(1, args.trial_epochs + 1):
            train_dann_epoch_once(
                model,
                source_loader,
                target_loader,
                optimizer,
                class_criterion,
                domain_criterion,
                device,
                epoch,
                args.trial_epochs,
                trial_cfg.domain_loss_weight,
                trial_cfg.dann_lambda_max,
            )
            val_metrics, _, _, _ = evaluate_closed_set(
                model,
                val_loader,
                class_criterion,
                device,
                len(KNOWN_CLASSES),
            )
            score = metric_value(val_metrics, "macro_f1", "f1_macro")
            best_score = max(best_score, score)

            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return float(best_score)

    return objective


def save_study_outputs(study, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "direction": study.direction.name,
    }
    with (output_dir / "best_hyperparameters.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            **trial.params,
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / "all_trials.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--study", choices=["closed_set", "dann"], default="closed_set")

    # Closed-set inputs
    parser.add_argument("--padufes-csv")
    parser.add_argument("--padufes-image-dir")

    # DANN inputs
    parser.add_argument("--standardized-csv")
    parser.add_argument("--source-dataset", default="PAD-UFES")
    parser.add_argument("--target-dataset")
    parser.add_argument("--source-image-roots", nargs="+")
    parser.add_argument("--target-image-roots", nargs="+")

    # Optuna outputs
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URI, e.g. sqlite:///study.db")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--trial-epochs", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--n-startup-trials", type=int, default=5)

    # Search spaces
    parser.add_argument(
        "--model-families",
        nargs="+",
        default=[
            "mobilevit_cross_attention",
            "mobilevit_gated",
            "mobilevit_concat",
            "resnet50_cross_attention",
            "resnet50_gated",
            "resnet50_concat",
        ],
    )
    parser.add_argument(
        "--text-cols",
        nargs="+",
        default=["text_full", "text_core", "text_missing_explicit"],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--fusion-dims", nargs="+", type=int, default=[128, 256, 384, 512])
    parser.add_argument("--num-heads-choices", nargs="+", type=int, default=[2, 4, 8])

    parser.add_argument("--lr-low", type=float, default=1e-6)
    parser.add_argument("--lr-high", type=float, default=3e-4)
    parser.add_argument("--weight-decay-low", type=float, default=1e-6)
    parser.add_argument("--weight-decay-high", type=float, default=1e-3)

    parser.add_argument("--use-sampler", action="store_true")
    parser.add_argument("--use-weighted-loss", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(SEED)

    cfg = ExperimentConfig()
    output_dir = Path(args.output_dir)

    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=args.n_startup_trials,
        multivariate=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(2, args.n_startup_trials // 2),
        n_warmup_steps=1,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    if args.study == "closed_set":
        if not args.padufes_csv or not args.padufes_image_dir:
            raise ValueError("--padufes-csv and --padufes-image-dir are required for --study closed_set.")
        objective = build_closed_set_objective(args, cfg)
    else:
        required = [
            args.standardized_csv,
            args.target_dataset,
            args.source_image_roots,
            args.target_image_roots,
        ]
        if any(x is None for x in required):
            raise ValueError(
                "--standardized-csv, --target-dataset, --source-image-roots, "
                "and --target-image-roots are required for --study dann."
            )
        objective = build_dann_objective(args, cfg)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    save_study_outputs(study, output_dir)

    print("Best value:", study.best_value)
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("Saved tuning outputs to:", output_dir)


if __name__ == "__main__":
    main()
