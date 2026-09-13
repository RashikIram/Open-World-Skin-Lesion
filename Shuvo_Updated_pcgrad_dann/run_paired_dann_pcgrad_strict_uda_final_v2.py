from __future__ import annotations

"""
PAIRED STANDARD DANN + ASYMMETRIC PCGRAD-DANN
================================================

Runs a strictly matched ablation in which EVERY final closed-set checkpoint is
adapted twice on each target domain:

    A) standard DANN
    B) the CORRECTED (v2) asymmetric PCGrad-DANN implementation from
       final_training_pipeline_reviewed.pcgrad_dann.dann_pcgrad_step

V2 CHANGE -- READ THIS BEFORE COMPARING TO THE v1 RESULTS
---------------------------------------------------------
v1 projected the domain gradient off the classification gradient over the
WHOLE flattened parameter vector. Because the domain loss does not touch the
classification head, that subtraction manufactured a spurious gradient on the
classifier head (and on the domain head), silently rescaling both task-specific
updates on every conflicting step. v2 restricts the projection to the SHARED
representation (image_encoder / text_encoder / fusion), which is the only place
the two objectives genuinely compete. Task heads now always receive exactly
their own gradient.

Per the review, standard DANN is NOT rerun: its 108 runs are frozen and reused.
Only the PCGrad branch is recomputed, so the paired comparison stays valid.

The two methods start from the exact same same-seed closed-set best.pt and use
the same seed, batches, optimizer, DANN hyperparameters, GRL schedule, source
classification loss, target exposure policy, and source_val checkpoint
selection. The ONLY optimization difference is how the class/domain gradients
are combined:

    standard: (L_cls + lambda_dom * L_dom).backward()
    pcgrad:   dann_pcgrad_step(L_cls, lambda_dom * L_dom, model)

The PCGrad implementation protects classification and only projects the
conflicting domain gradient, on shared parameters only. Its source file is
hashed and copied into the output provenance directory during preflight.

Matrix:
    54 final closed-set checkpoints x 2 targets x 2 methods = 216 runs.

No open-world thresholding is performed in this script. Both DANN variants are
selected ONLY by PAD source_val Macro-F1. Target known val/test metrics are
analysis outputs and never affect checkpoint selection.
"""

import argparse
import inspect
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(r"D:\Deep Learning")
PACKAGE = "final_training_pipeline_reviewed"
PACKAGE_DIR = ROOT / PACKAGE

STANDARDIZED_CSV = (
    ROOT / r"preprocessed_outputs\all_preprocessed_splits_standardized_text.csv"
)

SOURCE_IMAGE_ROOTS = [
    ROOT / "images",
]

CLOSED_RESULTS_ROOT = ROOT / "results_closedset_frozen_protocol"
CLOSED_CHECKPOINT_REGISTRY = CLOSED_RESULTS_ROOT / "dann_checkpoint_registry.csv"

OUTPUT_ROOT = ROOT / "results_dann_paired_strict_uda_v2"

# The v1 output tree. Standard DANN is FROZEN per the review and is never rerun;
# aggregation reuses its 108 standard runs from here when they are not present
# in the v2 tree. Only the PCGrad branch is recomputed.
FROZEN_STANDARD_ROOT = ROOT / "results_dann_paired_strict_uda"
RESOLVED_ROOTS_JSON = OUTPUT_ROOT / "resolved_target_image_roots.json"

SOURCE_DATASET = "PAD-UFES-20"
TARGET_DATASETS = ["ISIC 2019", "MCR-SL"]
METHODS = ["standard", "pcgrad"]
# SHA256 of the CORRECTED (v2) shared-parameter pcgrad_dann.py.
# v1 (whole-vector projection, superseded) was:
#   0714bc9275248f6ee64583ddc7e75deb7bdd2113823d8a9e3e899d9aa85d1923
EXPECTED_USER_PCGRAD_SHA256 = "0f7da335665949fbfe36872c635d006b6c23dd5d7adab2a6775b3cad3647b165"


# =============================================================================
# TARGET IMAGE-ROOT HINTS
# =============================================================================
#
# The script first tries these common locations. If they do not resolve the
# entire frozen target manifest, preflight automatically scans likely folders
# under D:\Deep Learning and adds directories containing representative target
# image files.
#
# If your local folder names are unusual, add the exact folders here. The
# correct root is the directory that directly contains image_file, not merely
# a parent several levels above it.

TARGET_IMAGE_ROOT_HINTS = {
    "ISIC 2019": [
        ROOT / "ISIC_2019_Training_Input",
        ROOT / "ISIC_2019_Test_Input",
        ROOT / "ISIC_2019_Training_Input" / "ISIC_2019_Training_Input",
        ROOT / "ISIC_2019_Test_Input" / "ISIC_2019_Test_Input",
        ROOT / "ISIC 2019" / "ISIC_2019_Training_Input",
        ROOT / "ISIC 2019" / "ISIC_2019_Test_Input",
        ROOT / "ISIC2019" / "ISIC_2019_Training_Input",
        ROOT / "ISIC2019" / "ISIC_2019_Test_Input",
    ],
    "MCR-SL": [
        ROOT / "MCR-SL_dataset",
        ROOT / "MCR-SL_dataset" / "images",
        ROOT / "MCR-SL_dataset" / "Images",
        ROOT / "MCR-SL_dataset" / "clinical",
        ROOT / "MCR-SL_dataset" / "Clinical",
        ROOT / "MCR-SL_dataset" / "dermoscopy",
        ROOT / "MCR-SL_dataset" / "Dermoscopy",
        ROOT / "MCR-SL_dataset" / "dermoscopic",
        ROOT / "MCR-SL_dataset" / "Dermoscopic",
    ],
}


# =============================================================================
# FINAL EXPERIMENT MATRIX
# =============================================================================

MODEL_FAMILIES = [
    "mobilevit_cross_attention",
    "mobilevit_gated",
    "mobilevit_concat",
    "resnet50_cross_attention",
    "resnet50_gated",
    "resnet50_concat",
]

TEXT_COLS = [
    "text_core",
    "text_full",
    "text_missing_explicit",
]

SEEDS = [42, 123, 2026]


# =============================================================================
# FROZEN SOURCE / DANN CONFIGURATION
# =============================================================================

BATCH_SIZE = 16
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 5.829384542994739e-04
FUSION_DIM = 128
NUM_HEADS = 4
DROPOUT = 0.1353970008207678
BALANCE_BETA = 0.9596031602585381

MAX_TEXT_LEN = 192
IMAGE_SIZE = 224
FREEZE_BACKBONES = False
NUM_WORKERS = 0

DANN_EPOCHS = 15
DANN_EARLY_STOP_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3

DOMAIN_LOSS_WEIGHT = 0.005
GRL_MAX_LAMBDA = 1.0

GPU_COOLDOWN_SECONDS = 10

EXPECTED_SOURCE_SPLITS = {
    "source_train": 1593,
    "source_val": 349,
    "source_test": 356,
}

EXPECTED_TARGET_SPLITS = {
    "ISIC 2019": {
        "target_adapt": 19864,
        "target_val": 5467,
        "target_test": 8238,
        "target_val_known": 4975,
        "target_val_unknown": 492,
        "target_test_known": 5996,
        "target_test_unknown": 2242,
    },
    "MCR-SL": {
        "target_adapt": 135,
        "target_val": 49,
        "target_test": 50,
        "target_val_known": 42,
        "target_val_unknown": 7,
        "target_test_known": 43,
        "target_test_unknown": 7,
    },
}


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def sanitize(x: str) -> str:
    return (
        str(x)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def child_env() -> dict:
    env = os.environ.copy()
    package_dir = str(PACKAGE_DIR)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = package_dir + (os.pathsep + current if current else "")
    return env


def run_dir(
    method: str,
    target_dataset: str,
    model_family: str,
    text_col: str,
    seed: int,
) -> Path:
    if method not in METHODS:
        raise ValueError(f"Unknown DANN method: {method}")
    return (
        OUTPUT_ROOT
        / "runs"
        / method
        / sanitize(target_dataset)
        / model_family
        / text_col
        / f"seed_{seed}"
    )


def run_complete(
    method: str,
    target_dataset: str,
    model_family: str,
    text_col: str,
    seed: int,
) -> bool:
    out = run_dir(method, target_dataset, model_family, text_col, seed)
    return (
        (out / "run_summary.json").exists()
        and (out / "best_dann.pt").exists()
    )


def cooldown(label: str) -> None:
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            subprocess.run(
                [
                    smi,
                    "--query-gpu=index,name,memory.used,memory.total,"
                    "utilization.gpu,temperature.gpu",
                    "--format=csv,noheader",
                ],
                cwd=str(ROOT),
                check=False,
            )
        except Exception as exc:
            log(f"nvidia-smi check skipped: {exc}")

    if GPU_COOLDOWN_SECONDS > 0:
        log(f"GPU cooldown {GPU_COOLDOWN_SECONDS}s after {label}")
        time.sleep(GPU_COOLDOWN_SECONDS)


def json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(x):
        try:
            import numpy as np
            if isinstance(x, np.integer):
                return int(x)
            if isinstance(x, np.floating):
                value = float(x)
                return None if math.isnan(value) else value
            if isinstance(x, np.ndarray):
                return x.tolist()
        except Exception:
            pass

        if isinstance(x, Path):
            return str(x)
        if isinstance(x, float) and math.isnan(x):
            return None
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        return x

    with path.open("w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=2)


# =============================================================================
# TARGET IMAGE-ROOT DISCOVERY / PREFLIGHT
# =============================================================================

IMAGE_EXTS = ("", ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _candidate_names(image_file: str) -> set[str]:
    name = str(image_file).strip()
    if not name:
        return set()

    out = {name}
    suffix = Path(name).suffix
    if not suffix:
        for ext in IMAGE_EXTS:
            if ext:
                out.add(name + ext)
    return out


def _representative_target_names(dataset_name: str) -> set[str]:
    import pandas as pd

    df = pd.read_csv(STANDARDIZED_CSV, low_memory=False)
    sub = df[df["dataset"].astype(str) == dataset_name].copy()
    if sub.empty:
        raise RuntimeError(
            f"No rows for dataset={dataset_name!r} in {STANDARDIZED_CSV}"
        )

    names: set[str] = set()

    for split_name in ["target_adapt", "target_val", "target_test"]:
        split_df = sub[sub["split"].astype(str) == split_name]
        if split_df.empty:
            continue

        # A few beginning/middle/end examples make discovery robust to
        # separate train/test image directories.
        indices = sorted(
            set(
                [
                    0,
                    len(split_df) // 3,
                    len(split_df) // 2,
                    max(len(split_df) - 1, 0),
                ]
            )
        )
        vals = split_df.iloc[indices]["image_file"].astype(str).tolist()
        for v in vals:
            names.update(_candidate_names(v))

    return names


def _scan_matching_directories(
    scan_base: Path,
    wanted_names: set[str],
    max_depth: int = 5,
) -> list[Path]:
    """
    One filesystem pass. Returns directories containing at least one
    representative image filename.
    """
    if not scan_base.exists():
        return []

    found: list[Path] = []
    base_depth = len(scan_base.parts)

    skip_names = {
        "results",
        "results_closedset_frozen_protocol",
        "results_dann_paired_strict_uda",
        "protocol_ablation",
        "lr_calibration_backbone",
        "tuning",
        "__pycache__",
        ".git",
    }

    for dirpath, dirnames, filenames in os.walk(scan_base):
        p = Path(dirpath)
        depth = len(p.parts) - base_depth

        # Prune known output/code folders and overly deep traversal.
        dirnames[:] = [
            d
            for d in dirnames
            if d not in skip_names
            and not d.startswith("results_")
            and not d.startswith(".")
        ]

        if depth >= max_depth:
            dirnames[:] = []

        if wanted_names.intersection(filenames):
            found.append(p)

    return found


def resolve_target_roots(dataset_name: str, force_scan: bool = False) -> list[Path]:
    """
    Resolve directories that collectively contain the entire target manifest.
    Hints are tried first; auto-discovery adds directories containing
    representative images.
    """
    import pandas as pd

    hints = [
        Path(p)
        for p in TARGET_IMAGE_ROOT_HINTS.get(dataset_name, [])
        if Path(p).exists()
    ]

    candidate_roots: list[Path] = []
    seen = set()

    def add_root(p: Path):
        rp = str(p.resolve()) if p.exists() else str(p)
        if p.exists() and rp not in seen:
            seen.add(rp)
            candidate_roots.append(p)

    for p in hints:
        add_root(p)

    # Determine if hints already appear sufficient using the raw manifest.
    df = pd.read_csv(STANDARDIZED_CSV, low_memory=False)
    sub = df[df["dataset"].astype(str) == dataset_name].copy()

    def count_resolved(roots: list[Path]) -> int:
        if not roots:
            return 0

        def exists_for(image_file: str) -> bool:
            for root in roots:
                for candidate_name in _candidate_names(image_file):
                    if (root / candidate_name).exists():
                        return True
            return False

        return int(sub["image_file"].astype(str).map(exists_for).sum())

    resolved_by_hints = count_resolved(candidate_roots)

    if force_scan or resolved_by_hints != len(sub):
        wanted = _representative_target_names(dataset_name)

        if dataset_name == "MCR-SL":
            preferred = ROOT / "MCR-SL_dataset"
            scan_bases = [preferred] if preferred.exists() else [ROOT]
            max_depth = 7
        else:
            scan_bases = [ROOT]
            max_depth = 4

        for base in scan_bases:
            for p in _scan_matching_directories(base, wanted, max_depth=max_depth):
                add_root(p)

    # Keep only roots that resolve at least one target row.
    useful: list[Path] = []
    for root in candidate_roots:
        hit = False
        for image_file in sub["image_file"].astype(str).iloc[
            :: max(len(sub) // 50, 1)
        ]:
            if any((root / n).exists() for n in _candidate_names(image_file)):
                hit = True
                break
        if hit:
            useful.append(root)

    # If coarse subsampling missed a train/test root, keep any discovered root
    # that contains a representative name.
    rep_names = _representative_target_names(dataset_name)
    for root in candidate_roots:
        if root in useful:
            continue
        try:
            files_here = set(os.listdir(root))
        except Exception:
            continue
        if files_here.intersection(rep_names):
            useful.append(root)

    return useful


def _call_load_standardized_splits(
    *,
    csv_path: str,
    image_roots: list[str],
    dataset_name: str,
    required_text_col: str,
):
    """
    Compatibility wrapper: use strict_images=True when the active reviewed
    preprocessing loader supports it.
    """
    from final_training_pipeline_reviewed.preprocessing import (
        load_standardized_splits,
    )

    sig = inspect.signature(load_standardized_splits)
    kwargs = {}

    if "required_text_col" in sig.parameters:
        kwargs["required_text_col"] = required_text_col
    if "strict_images" in sig.parameters:
        kwargs["strict_images"] = True

    return load_standardized_splits(
        csv_path,
        image_roots,
        dataset_name,
        **kwargs,
    )


def preflight(force_scan: bool = False) -> dict:
    """
    Validate:
      - standardized CSV
      - 54 closed checkpoint registry rows
      - every same-seed best.pt exists
      - target image roots resolve all V5 rows
      - exact V5 target split sizes and unknown counts
      - no unknowns in target_adapt
      - active build_model exposes the frozen dropout parameter
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not STANDARDIZED_CSV.exists():
        raise FileNotFoundError(f"Missing standardized CSV: {STANDARDIZED_CSV}")
    if not CLOSED_CHECKPOINT_REGISTRY.exists():
        raise FileNotFoundError(
            f"Missing closed checkpoint registry: {CLOSED_CHECKPOINT_REGISTRY}"
        )

    sys.path.insert(0, str(PACKAGE_DIR))
    sys.path.insert(0, str(ROOT))

    import pandas as pd

    from final_training_pipeline_reviewed.models import build_model
    from final_training_pipeline_reviewed.pcgrad_dann import dann_pcgrad_step
    from final_training_pipeline_reviewed.preprocessing import (
        add_known_unknown_columns,
    )

    build_sig = inspect.signature(build_model)
    if "dropout" not in build_sig.parameters:
        raise RuntimeError(
            "Active build_model() does not expose a dropout argument. "
            "The final closed-set runs used a dropout-aware reviewed package, "
            "so this indicates a package-version mismatch."
        )

    # Verify and snapshot the user's exact asymmetric PCGrad implementation.
    pcgrad_file = PACKAGE_DIR / "pcgrad_dann.py"
    if not pcgrad_file.exists():
        raise FileNotFoundError(
            f"Missing user's PCGrad implementation: {pcgrad_file}"
        )
    from final_training_pipeline_reviewed.pcgrad_dann import dann_pcgrad_step
    pcgrad_sig = inspect.signature(dann_pcgrad_step)
    expected_pcgrad_params = [
        "class_loss",
        "weighted_domain_loss",
        "model",
        "eps",
        "shared_prefixes",
    ]
    if list(pcgrad_sig.parameters) != expected_pcgrad_params:
        raise RuntimeError(
            "dann_pcgrad_step signature differs from the reviewed user version. "
            f"Observed: {pcgrad_sig}"
        )
    pcgrad_sha256 = hashlib.sha256(pcgrad_file.read_bytes()).hexdigest()
    if pcgrad_sha256 != EXPECTED_USER_PCGRAD_SHA256:
        raise RuntimeError(
            "Active pcgrad_dann.py is NOT byte-identical to the user's reviewed "
            "version supplied for this study. Refusing to run a different PCGrad "
            "implementation.\n"
            f"Expected SHA256: {EXPECTED_USER_PCGRAD_SHA256}\n"
            f"Observed SHA256: {pcgrad_sha256}\n"
            f"File: {pcgrad_file}"
        )
    provenance_dir = OUTPUT_ROOT / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pcgrad_file, provenance_dir / "pcgrad_dann_USER_VERSION.py")
    (provenance_dir / "pcgrad_dann_sha256.txt").write_text(
        pcgrad_sha256 + "\n", encoding="utf-8"
    )

    registry = pd.read_csv(CLOSED_CHECKPOINT_REGISTRY)

    expected_cols = {
        "model_family",
        "text_col",
        "seed",
        "best_checkpoint",
    }
    missing_cols = expected_cols - set(registry.columns)
    if missing_cols:
        raise RuntimeError(
            f"Closed checkpoint registry missing columns: {sorted(missing_cols)}"
        )

    expected_jobs = {
        (m, t, int(s))
        for m in MODEL_FAMILIES
        for t in TEXT_COLS
        for s in SEEDS
    }
    actual_jobs = {
        (str(r.model_family), str(r.text_col), int(r.seed))
        for r in registry.itertuples(index=False)
    }

    if len(registry) != 54 or actual_jobs != expected_jobs:
        missing = sorted(expected_jobs - actual_jobs)
        extra = sorted(actual_jobs - expected_jobs)
        raise RuntimeError(
            "Closed checkpoint registry is not the exact 54-run final matrix.\n"
            f"rows={len(registry)}, missing={missing}, extra={extra}"
        )

    missing_ckpts = [
        str(p)
        for p in registry["best_checkpoint"].astype(str)
        if not Path(p).exists()
    ]
    if missing_ckpts:
        raise FileNotFoundError(
            "Some same-seed closed best.pt checkpoints are missing. "
            f"Examples: {missing_ckpts[:5]}"
        )

    resolved = {}

    for target_dataset in TARGET_DATASETS:
        roots = resolve_target_roots(
            target_dataset,
            force_scan=force_scan,
        )
        if not roots:
            raise RuntimeError(
                f"Could not resolve image roots for {target_dataset}. "
                "Edit TARGET_IMAGE_ROOT_HINTS near the top of this script "
                "to include the directories that directly contain its images."
            )

        target_df = _call_load_standardized_splits(
            csv_path=str(STANDARDIZED_CSV),
            image_roots=[str(p) for p in roots],
            dataset_name=target_dataset,
            required_text_col="text_missing_explicit",
        )
        target_df = add_known_unknown_columns(target_df)

        expected = EXPECTED_TARGET_SPLITS[target_dataset]

        counts = {
            "target_adapt": int((target_df["split"] == "target_adapt").sum()),
            "target_val": int((target_df["split"] == "target_val").sum()),
            "target_test": int((target_df["split"] == "target_test").sum()),
        }

        val_df = target_df[target_df["split"] == "target_val"]
        test_df = target_df[target_df["split"] == "target_test"]
        adapt_df = target_df[target_df["split"] == "target_adapt"]

        counts.update(
            {
                "target_val_known": int((~val_df["is_unknown"]).sum()),
                "target_val_unknown": int(val_df["is_unknown"].sum()),
                "target_test_known": int((~test_df["is_unknown"]).sum()),
                "target_test_unknown": int(test_df["is_unknown"].sum()),
            }
        )

        if counts != expected:
            raise RuntimeError(
                f"{target_dataset} V5 target split mismatch after image "
                f"resolution.\nExpected: {expected}\nObserved: {counts}\n"
                f"Roots tried: {[str(p) for p in roots]}"
            )

        if bool(adapt_df["is_unknown"].any()):
            raise RuntimeError(
                f"{target_dataset}: target_adapt contains unknown rows. "
                "DANN must not start."
            )

        resolved[target_dataset] = [str(p) for p in roots]
        log(
            f"PREFLIGHT PASS {target_dataset}: "
            f"roots={resolved[target_dataset]} counts={counts}"
        )

    json_dump(resolved, RESOLVED_ROOTS_JSON)

    manifest = {
        "stage": "paired_standard_and_pcgrad_dann",
        "protocol": "strict_source_validation_selection_paired_ablation",
        "methods": METHODS,
        "source_dataset": SOURCE_DATASET,
        "target_datasets": TARGET_DATASETS,
        "model_families": MODEL_FAMILIES,
        "text_cols": TEXT_COLS,
        "seeds": SEEDS,
        "n_closed_checkpoints": 54,
        "n_runs_per_method": 108,
        "n_total_paired_runs": 216,
        "checkpoint_initialization": (
            "same-seed source_val-selected final closed-set best.pt"
        ),
        "dann_checkpoint_selection": "source_val Macro-F1 only",
        "target_val_used_for_dann_selection": False,
        "target_test_used_for_dann_selection": False,
        "target_labels_used_for_sampling": False,
        "target_labels_used_for_classification_loss": False,
        "target_adapt_protocol": "pre-frozen known-only V5 benchmark split",
        "source_sampler": "natural_shuffle",
        "target_sampler": "uniform_shuffle",
        "source_epoch_anchor": True,
        "cycle_target_if_exhausted": True,
        "source_classification_loss": (
            "effective-number weighted cross-entropy"
        ),
        "optimizer": "AdamW",
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "fusion_dim": FUSION_DIM,
        "num_heads": NUM_HEADS,
        "dropout": DROPOUT,
        "balance_beta": BALANCE_BETA,
        "max_text_len": MAX_TEXT_LEN,
        "image_size": IMAGE_SIZE,
        "augmentation": "strong_no_blur",
        "domain_loss_weight": DOMAIN_LOSS_WEIGHT,
        "grl_schedule": "2/(1+exp(-10p))-1",
        "grl_max_lambda": GRL_MAX_LAMBDA,
        "max_epochs": DANN_EPOCHS,
        "early_stop_patience": DANN_EARLY_STOP_PATIENCE,
        "scheduler": "ReduceLROnPlateau on source_val Macro-F1",
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "pcgrad_ablation": {
            "implementation": "final_training_pipeline_reviewed.pcgrad_dann.dann_pcgrad_step",
            "mode": "asymmetric",
            "protected_objective": "classification",
            "primary_index": 0,
            "pcgrad_source_sha256": pcgrad_sha256,
            "expected_user_pcgrad_sha256": EXPECTED_USER_PCGRAD_SHA256,
            "byte_identical_to_user_supplied_version": True,
            "difference_from_standard": "gradient combination only",
        },
        "paired_rng_protocol": "same seed and loader generators per standard/pcgrad pair",
        "openworld_in_this_stage": False,
        "resolved_target_image_roots": resolved,
    }
    json_dump(manifest, OUTPUT_ROOT / "protocol_manifest.json")

    log("PREFLIGHT COMPLETE — safe to launch paired Standard DANN + user PCGrad-DANN.")
    return resolved


def load_resolved_roots() -> dict:
    if not RESOLVED_ROOTS_JSON.exists():
        return preflight()

    with RESOLVED_ROOTS_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# DANN WORKER
# =============================================================================

def worker(
    method: str,
    target_dataset: str,
    model_family: str,
    text_col: str,
    seed: int,
    closed_checkpoint: str,
) -> None:
    sys.path.insert(0, str(PACKAGE_DIR))
    sys.path.insert(0, str(ROOT))

    import gc
    from dataclasses import replace

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
    from torch.autograd import Function
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from config import ExperimentConfig, KNOWN_CLASSES
    from final_training_pipeline_reviewed.datasets import (
        DomainAdaptationDataset,
        get_loss_weights_from_train_df,
    )
    from final_training_pipeline_reviewed.models import build_model
    # IMPORTANT: each worker runs in a fresh subprocess.  The PCGrad function
    # imported during preflight is therefore not present in the worker's
    # namespace. Import the user's reviewed implementation inside the worker.
    from final_training_pipeline_reviewed.pcgrad_dann import dann_pcgrad_step
    from final_training_pipeline_reviewed.preprocessing import (
        add_known_unknown_columns,
        select_split,
        validate_frozen_protocol,
    )
    from final_training_pipeline_reviewed.transforms import get_eval_transform
    from final_training_pipeline_reviewed.utils import (
        get_device,
        make_generator,
        seed_everything,
        seed_worker,
    )

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")

    if method == "pcgrad":
        pcgrad_sig = inspect.signature(dann_pcgrad_step)
        expected_pcgrad_params = [
            "class_loss",
            "weighted_domain_loss",
            "model",
            "eps",
            "shared_prefixes",
        ]
        if list(pcgrad_sig.parameters) != expected_pcgrad_params:
            raise RuntimeError(
                "Worker loaded an unexpected dann_pcgrad_step signature: "
                f"{pcgrad_sig}. Refusing to run a different PCGrad implementation."
            )

    out = run_dir(method, target_dataset, model_family, text_col, seed)
    out.mkdir(parents=True, exist_ok=True)

    resolved_roots = load_resolved_roots()
    target_roots = resolved_roots[target_dataset]

    cfg = ExperimentConfig(seed=seed)
    cfg = replace(
        cfg,
        max_text_len=MAX_TEXT_LEN,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fusion_dim=FUSION_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        balance_beta=BALANCE_BETA,
        source_sampler="none",
        use_weighted_loss=True,
        scheduler_factor=SCHEDULER_FACTOR,
        scheduler_patience=SCHEDULER_PATIENCE,
        num_workers=NUM_WORKERS,
        freeze_backbones=FREEZE_BACKBONES,
    )

    seed_everything(seed)
    device = get_device()

    # -------------------------------------------------------------------------
    # Load frozen source + target manifests.
    # -------------------------------------------------------------------------
    source_df = _call_load_standardized_splits(
        csv_path=str(STANDARDIZED_CSV),
        image_roots=[str(p) for p in SOURCE_IMAGE_ROOTS],
        dataset_name=SOURCE_DATASET,
        required_text_col=text_col,
    )
    validate_frozen_protocol(source_df)
    source_df = add_known_unknown_columns(source_df)

    target_df = _call_load_standardized_splits(
        csv_path=str(STANDARDIZED_CSV),
        image_roots=target_roots,
        dataset_name=target_dataset,
        required_text_col=text_col,
    )
    target_df = add_known_unknown_columns(target_df)

    source_train_df = select_split(
        source_df, "source_train"
    ).reset_index(drop=True)
    source_val_df = select_split(
        source_df, "source_val"
    ).reset_index(drop=True)
    source_test_df = select_split(
        source_df, "source_test"
    ).reset_index(drop=True)

    if source_train_df["is_unknown"].any():
        raise RuntimeError("Source train unexpectedly contains unknown rows.")

    # Use the entire PRE-FROZEN target_adapt split as-is. We do not use
    # target class labels to decide which adaptation rows enter training.
    # The V5 protocol itself guarantees that this split is known-only; the
    # is_unknown check below is an audit assertion, not a sampling/filter rule.
    target_adapt_df = target_df[
        target_df["split"] == "target_adapt"
    ].reset_index(drop=True)
    target_val_all_df = target_df[
        target_df["split"] == "target_val"
    ].reset_index(drop=True)
    target_test_all_df = target_df[
        target_df["split"] == "target_test"
    ].reset_index(drop=True)

    target_val_known_df = target_val_all_df[
        ~target_val_all_df["is_unknown"]
    ].reset_index(drop=True)
    target_test_known_df = target_test_all_df[
        ~target_test_all_df["is_unknown"]
    ].reset_index(drop=True)

    if target_adapt_df["is_unknown"].any():
        raise RuntimeError(
            "Unknown target sample reached target_adapt. Abort."
        )

    if len(source_train_df) != EXPECTED_SOURCE_SPLITS["source_train"]:
        raise RuntimeError("Source train size mismatch.")
    if len(source_val_df) != EXPECTED_SOURCE_SPLITS["source_val"]:
        raise RuntimeError("Source val size mismatch.")
    if len(source_test_df) != EXPECTED_SOURCE_SPLITS["source_test"]:
        raise RuntimeError("Source test size mismatch.")

    expected = EXPECTED_TARGET_SPLITS[target_dataset]
    observed = {
        "target_adapt": len(target_adapt_df),
        "target_val": len(target_val_all_df),
        "target_test": len(target_test_all_df),
        "target_val_known": len(target_val_known_df),
        "target_val_unknown": int(target_val_all_df["is_unknown"].sum()),
        "target_test_known": len(target_test_known_df),
        "target_test_unknown": int(target_test_all_df["is_unknown"].sum()),
    }
    if observed != expected:
        raise RuntimeError(
            f"{target_dataset} split mismatch. "
            f"Expected={expected}, observed={observed}"
        )

    # Save the exact rows used by this run.
    source_train_df.to_csv(out / "source_train.csv", index=False)
    source_val_df.to_csv(out / "source_val.csv", index=False)
    source_test_df.to_csv(out / "source_test.csv", index=False)
    target_adapt_df.to_csv(out / "target_adapt_known.csv", index=False)
    target_val_all_df.to_csv(out / "target_val_all.csv", index=False)
    target_test_all_df.to_csv(out / "target_test_all.csv", index=False)

    # -------------------------------------------------------------------------
    # Frozen strong-no-blur training augmentation.
    # -------------------------------------------------------------------------
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(
                cfg.image_size,
                scale=(0.65, 1.0),
                ratio=(0.8, 1.25),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomGrayscale(p=0.05),
            # GaussianBlur deliberately omitted.
            transforms.ToTensor(),
            transforms.Normalize(
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            transforms.RandomErasing(
                p=0.1,
                scale=(0.02, 0.1),
            ),
        ]
    )
    eval_transform = get_eval_transform(cfg.image_size)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    # -------------------------------------------------------------------------
    # Datasets/loaders.
    # Natural source shuffle + uniform target shuffle. NO samplers.
    # -------------------------------------------------------------------------
    source_train_ds = DomainAdaptationDataset(
        source_train_df,
        tokenizer,
        train_transform,
        text_col,
        domain_label=0,
        max_text_len=cfg.max_text_len,
    )
    target_adapt_ds = DomainAdaptationDataset(
        target_adapt_df,
        tokenizer,
        train_transform,
        text_col,
        domain_label=1,
        max_text_len=cfg.max_text_len,
    )

    source_val_ds = DomainAdaptationDataset(
        source_val_df,
        tokenizer,
        eval_transform,
        text_col,
        domain_label=0,
        max_text_len=cfg.max_text_len,
    )
    source_test_ds = DomainAdaptationDataset(
        source_test_df,
        tokenizer,
        eval_transform,
        text_col,
        domain_label=0,
        max_text_len=cfg.max_text_len,
    )
    target_val_known_ds = DomainAdaptationDataset(
        target_val_known_df,
        tokenizer,
        eval_transform,
        text_col,
        domain_label=1,
        max_text_len=cfg.max_text_len,
    )
    target_test_known_ds = DomainAdaptationDataset(
        target_test_known_df,
        tokenizer,
        eval_transform,
        text_col,
        domain_label=1,
        max_text_len=cfg.max_text_len,
    )

    common_loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )

    source_train_loader = DataLoader(
        source_train_ds,
        shuffle=True,
        sampler=None,
        generator=make_generator(seed),
        drop_last=False,
        **common_loader_kwargs,
    )

    # drop_last=True keeps each target training batch full-sized. The final
    # source batch is matched by cropping the target batch to the source size.
    target_adapt_loader = DataLoader(
        target_adapt_ds,
        shuffle=True,
        sampler=None,
        generator=make_generator(seed + 10000),
        drop_last=True,
        **common_loader_kwargs,
    )

    if len(target_adapt_loader) == 0:
        raise RuntimeError(
            "Target adapt loader has zero full batches. "
            "This should not happen with the frozen V5 targets and batch=16."
        )

    def eval_loader(ds):
        return DataLoader(
            ds,
            shuffle=False,
            drop_last=False,
            **common_loader_kwargs,
        )

    source_val_loader = eval_loader(source_val_ds)
    source_test_loader = eval_loader(source_test_ds)
    target_val_known_loader = eval_loader(target_val_known_ds)
    target_test_known_loader = eval_loader(target_test_known_ds)

    # -------------------------------------------------------------------------
    # Exact DANN architecture built from the SAME closed-set architecture.
    #
    # This avoids relying on an older build_dann_model implementation. We build
    # the proven dropout-aware closed architecture, load best.pt STRICTLY, then
    # add only the new domain head.
    # -------------------------------------------------------------------------
    build_sig = inspect.signature(build_model)
    if "dropout" not in build_sig.parameters:
        raise RuntimeError(
            "Active build_model() has no dropout argument. "
            "Package mismatch with the final closed-set experiment."
        )

    image_model_name = (
        cfg.resnet_model_name
        if model_family.startswith("resnet50")
        else cfg.image_model_name
    )

    closed_model = build_model(
        model_family,
        image_model_name,
        cfg.text_model_name,
        len(KNOWN_CLASSES),
        fusion_dim=cfg.fusion_dim,
        num_heads=cfg.num_heads,
        freeze_backbones=cfg.freeze_backbones,
        dropout=cfg.dropout,
    )

    closed_checkpoint_path = Path(closed_checkpoint)
    if not closed_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing same-seed closed checkpoint: {closed_checkpoint_path}"
        )

    closed_payload = torch.load(
        closed_checkpoint_path,
        map_location="cpu",
    )
    closed_state = closed_payload.get(
        "model_state_dict",
        closed_payload,
    )

    # Strict=True ensures the DANN run starts from exactly the final closed
    # architecture rather than silently ignoring incompatible source weights.
    closed_model.load_state_dict(closed_state, strict=True)

    class GradReverse(Function):
        @staticmethod
        def forward(ctx, x, lambd: float):
            ctx.lambd = float(lambd)
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambd * grad_output, None

    class ExactDANN(nn.Module):
        def __init__(self, source_model, fusion_dim: int, dropout: float):
            super().__init__()

            # Preserve root module names expected by the project's DANN model.
            self.image_encoder = source_model.image_encoder
            self.text_encoder = source_model.text_encoder
            self.fusion = source_model.fusion
            self.classifier = source_model.classifier

            self.domain_classifier = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, 2),
            )

        def extract_features(
            self,
            pixel_values,
            input_ids,
            attention_mask,
        ):
            image_tokens = self.image_encoder(pixel_values)
            text_tokens = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            return self.fusion(
                image_tokens,
                text_tokens,
                attention_mask,
            )

        def forward(
            self,
            pixel_values,
            input_ids,
            attention_mask,
            dann_lambda: float = 0.0,
        ):
            fused = self.extract_features(
                pixel_values,
                input_ids,
                attention_mask,
            )
            class_logits = self.classifier(fused)
            domain_logits = self.domain_classifier(
                GradReverse.apply(fused, float(dann_lambda))
            )
            return class_logits, domain_logits, fused

    model = ExactDANN(
        closed_model,
        cfg.fusion_dim,
        cfg.dropout,
    ).to(device)
    del closed_model

    # -------------------------------------------------------------------------
    # Losses / optimizer / scheduler.
    # -------------------------------------------------------------------------
    class_weights = get_loss_weights_from_train_df(
        source_train_df,
        len(KNOWN_CLASSES),
        cfg.balance_beta,
    ).to(device)

    class_criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )
    domain_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )

    # -------------------------------------------------------------------------
    # Evaluation helpers.
    # -------------------------------------------------------------------------
    labels = list(range(len(KNOWN_CLASSES)))

    @torch.no_grad()
    def collect_classifier_outputs(loader):
        model.eval()
        ys, preds, probs = [], [], []

        for batch in loader:
            batch = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in batch.items()
            }
            class_logits, _, _ = model(
                batch["pixel_values"],
                batch.get("input_ids"),
                batch.get("attention_mask"),
                dann_lambda=0.0,
            )
            p = F.softmax(class_logits, dim=1)
            yhat = torch.argmax(p, dim=1)

            ys.append(batch["label"].detach().cpu().numpy())
            preds.append(yhat.detach().cpu().numpy())
            probs.append(p.detach().cpu().numpy())

        return (
            np.concatenate(ys),
            np.concatenate(preds),
            np.concatenate(probs),
        )

    def metric_bundle(y_true, y_pred, y_prob):
        y_bin = label_binarize(
            y_true,
            classes=labels,
        )

        # Use the same one-vs-rest macro AUC concept as the closed-set stage.
        try:
            auc_macro = float(
                roc_auc_score(
                    y_bin,
                    y_prob,
                    average="macro",
                    multi_class="ovr",
                )
            )
        except Exception:
            aucs = []
            from sklearn.metrics import roc_auc_score as _binary_auc
            for i in labels:
                if len(np.unique(y_bin[:, i])) < 2:
                    continue
                aucs.append(
                    float(_binary_auc(y_bin[:, i], y_prob[:, i]))
                )
            auc_macro = (
                float(np.mean(aucs))
                if aucs
                else float("nan")
            )

        return {
            "accuracy": float(
                accuracy_score(y_true, y_pred)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true, y_pred)
            ),
            "macro_precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_auc_ovr": auc_macro,
        }

    def evaluate_split(
        name: str,
        split_df,
        loader,
        save_outputs: bool = True,
    ):
        y_true, y_pred, y_prob = collect_classifier_outputs(loader)
        metrics = metric_bundle(y_true, y_pred, y_prob)

        if save_outputs:
            split_dir = out / name
            split_dir.mkdir(parents=True, exist_ok=True)

            json_dump(metrics, split_dir / "metrics.json")
            pd.DataFrame([metrics]).to_csv(
                split_dir / "metrics.csv",
                index=False,
            )

            report = classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=KNOWN_CLASSES,
                zero_division=0,
                output_dict=True,
                digits=6,
            )
            pd.DataFrame(report).transpose().to_csv(
                split_dir / "classification_report.csv"
            )

            cm_count = confusion_matrix(
                y_true,
                y_pred,
                labels=labels,
            )
            cm_norm = confusion_matrix(
                y_true,
                y_pred,
                labels=labels,
                normalize="true",
            )
            cm_norm = np.nan_to_num(cm_norm, nan=0.0)

            pd.DataFrame(
                cm_count,
                index=KNOWN_CLASSES,
                columns=KNOWN_CLASSES,
            ).to_csv(
                split_dir / "confusion_matrix_counts.csv"
            )
            pd.DataFrame(
                cm_norm,
                index=KNOWN_CLASSES,
                columns=KNOWN_CLASSES,
            ).to_csv(
                split_dir / "confusion_matrix_normalized.csv"
            )

            pred_df = split_df.reset_index(drop=True).copy()
            pred_df["y_true"] = y_true
            pred_df["y_pred"] = y_pred
            pred_df["y_true_label"] = [
                KNOWN_CLASSES[int(i)] for i in y_true
            ]
            pred_df["y_pred_label"] = [
                KNOWN_CLASSES[int(i)] for i in y_pred
            ]
            for i, cls in enumerate(KNOWN_CLASSES):
                pred_df[f"prob_{cls}"] = y_prob[:, i]
            pred_df.to_csv(
                split_dir / "predictions.csv",
                index=False,
            )

        return metrics

    # -------------------------------------------------------------------------
    # Before-DANN direct-transfer baseline.
    # This is evaluation only; target metrics never select a checkpoint.
    # -------------------------------------------------------------------------
    before_target_val = evaluate_split(
        "before_dann_target_val_known",
        target_val_known_df,
        target_val_known_loader,
    )
    before_target_test = evaluate_split(
        "before_dann_target_test_known",
        target_test_known_df,
        target_test_known_loader,
    )
    before_source_val = evaluate_split(
        "before_dann_source_val",
        source_val_df,
        source_val_loader,
    )

    # -------------------------------------------------------------------------
    # Metadata before training.
    # -------------------------------------------------------------------------
    run_config = {
        "stage": "paired_dann_ablation",
        "method": method,
        "protocol_version": "strict_source_val_selection_paired_v1",
        "pcgrad": method == "pcgrad",
        "pcgrad_implementation": (
            "final_training_pipeline_reviewed.pcgrad_dann.dann_pcgrad_step"
            if method == "pcgrad" else None
        ),
        "pcgrad_mode": "asymmetric_primary_classification" if method == "pcgrad" else None,
        "openworld": False,
        "model_family": model_family,
        "text_col": text_col,
        "seed": seed,
        "source_dataset": SOURCE_DATASET,
        "target_dataset": target_dataset,
        "closed_checkpoint": str(closed_checkpoint_path),
        "closed_checkpoint_role": "same_seed_best_source_val",
        "selection_split": "source_val",
        "selection_metric": "macro_f1",
        "target_val_used_for_selection": False,
        "target_test_used_for_selection": False,
        "target_labels_used_for_sampling": False,
        "target_labels_used_for_classification_loss": False,
        "target_adapt_protocol": "pre_frozen_known_only_V5",
        "source_train_n": len(source_train_df),
        "source_val_n": len(source_val_df),
        "source_test_n": len(source_test_df),
        "target_adapt_known_n": len(target_adapt_df),
        "target_val_known_n": len(target_val_known_df),
        "target_val_unknown_reserved_n": int(
            target_val_all_df["is_unknown"].sum()
        ),
        "target_test_known_n": len(target_test_known_df),
        "target_test_unknown_reserved_n": int(
            target_test_all_df["is_unknown"].sum()
        ),
        "source_sampler": "none_natural_shuffle",
        "target_sampler": "none_uniform_shuffle",
        "source_epoch_anchor": True,
        "n_source_steps_per_epoch": len(source_train_loader),
        "n_target_batches": len(target_adapt_loader),
        "target_cycle_when_exhausted": True,
        "source_classification_loss": (
            "effective_number_weighted_cross_entropy"
        ),
        "balance_beta": cfg.balance_beta,
        "domain_loss": "cross_entropy",
        "domain_loss_weight": DOMAIN_LOSS_WEIGHT,
        "grl_schedule": "2/(1+exp(-10p))-1",
        "grl_max_lambda": GRL_MAX_LAMBDA,
        "optimizer": "AdamW",
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "fusion_dim": cfg.fusion_dim,
        "num_heads": cfg.num_heads,
        "dropout": cfg.dropout,
        "max_text_len": cfg.max_text_len,
        "image_size": cfg.image_size,
        "augmentation": "strong_no_blur",
        "max_epochs": DANN_EPOCHS,
        "early_stop_patience": DANN_EARLY_STOP_PATIENCE,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_monitor": "source_val_macro_f1",
        "scheduler_factor": cfg.scheduler_factor,
        "scheduler_patience": cfg.scheduler_patience,
        "before_target_val_known_macro_f1": (
            before_target_val["macro_f1"]
        ),
        "before_target_test_known_macro_f1": (
            before_target_test["macro_f1"]
        ),
        "before_source_val_macro_f1": (
            before_source_val["macro_f1"]
        ),
    }
    json_dump(run_config, out / "run_config.json")

    # -------------------------------------------------------------------------
    # Source-anchored DANN training.
    # -------------------------------------------------------------------------
    def grl_lambda(
        global_step: int,
        total_steps: int,
    ) -> float:
        p = global_step / max(total_steps - 1, 1)
        return float(
            GRL_MAX_LAMBDA
            * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
        )

    def crop_batch_to_n(batch: dict, n: int) -> dict:
        out_batch = {}
        for k, v in batch.items():
            if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] >= n:
                out_batch[k] = v[:n]
            else:
                out_batch[k] = v
        return out_batch

    best_source_val_f1 = -1.0
    best_epoch = 0
    patience_left = DANN_EARLY_STOP_PATIENCE
    history = []

    best_path = out / "best_dann.pt"
    last_path = out / "last_dann.pt"

    n_steps_per_epoch = len(source_train_loader)
    total_planned_steps = n_steps_per_epoch * DANN_EPOCHS

    global_step_counter = 0

    for epoch in range(1, DANN_EPOCHS + 1):
        model.train()

        target_iter = iter(target_adapt_loader)

        total_loss = 0.0
        total_class_loss = 0.0
        total_domain_loss = 0.0
        total_domain_correct = 0
        total_domain_n = 0
        total_pcgrad_conflicts = 0.0
        # v2 diagnostics: the review noted the v1 conflict *rate* alone had no
        # relationship to how much PCGrad helped (Pearson r ~ -0.047). Log the
        # geometry too, so the mechanism can be tested rather than assumed.
        total_pcgrad_cosine = 0.0
        total_pcgrad_proj_scale = 0.0
        lambda_values = []

        for src in tqdm(
            source_train_loader,
            desc=(
                f"DANN {sanitize(target_dataset)} "
                f"{model_family}/{text_col}/seed{seed}"
            ),
            leave=False,
        ):
            try:
                tgt = next(target_iter)
            except StopIteration:
                # Tiny MCR-SL is intentionally cycled so the source defines
                # epoch length. A new iterator reshuffles target samples.
                target_iter = iter(target_adapt_loader)
                tgt = next(target_iter)

            src = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in src.items()
            }
            tgt = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in tgt.items()
            }

            src_n = int(src["pixel_values"].shape[0])
            tgt = crop_batch_to_n(tgt, src_n)

            lambd = grl_lambda(
                global_step_counter,
                total_planned_steps,
            )
            lambda_values.append(lambd)

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

            # Source labels are used here.
            class_loss = class_criterion(
                src_class_logits,
                src["label"],
            )

            # Target CLASS labels are never accessed.
            domain_logits = torch.cat(
                [src_domain_logits, tgt_domain_logits],
                dim=0,
            )

            src_domain_y = torch.zeros(
                src_domain_logits.shape[0],
                dtype=torch.long,
                device=device,
            )
            tgt_domain_y = torch.ones(
                tgt_domain_logits.shape[0],
                dtype=torch.long,
                device=device,
            )
            domain_y = torch.cat(
                [src_domain_y, tgt_domain_y],
                dim=0,
            )

            domain_loss = domain_criterion(
                domain_logits,
                domain_y,
            )

            weighted_domain_loss = DOMAIN_LOSS_WEIGHT * domain_loss
            loss = class_loss + weighted_domain_loss

            if method == "pcgrad":
                # V2: asymmetric PCGrad restricted to the SHARED representation.
                # The classifier and domain heads keep their own gradient
                # exactly; only the shared slice is deconflicted. Do NOT call
                # loss.backward() here; the function writes deconflicted .grad.
                stats = dann_pcgrad_step(
                    class_loss,
                    weighted_domain_loss,
                    model,
                )
                total_pcgrad_conflicts += float(
                    stats.get("pcgrad_conflicts", 0)
                )
                total_pcgrad_cosine += float(
                    stats.get("pcgrad_cosine", 0.0)
                )
                total_pcgrad_proj_scale += float(
                    stats.get("pcgrad_projection_scale", 0.0)
                )
            else:
                # Standard DANN baseline: ordinary summed-loss backward.
                loss.backward()

            optimizer.step()

            with torch.no_grad():
                domain_pred = domain_logits.argmax(dim=1)
                total_domain_correct += int(
                    (domain_pred == domain_y).sum().item()
                )
                total_domain_n += int(domain_y.numel())

            total_loss += float(loss.item())
            total_class_loss += float(class_loss.item())
            total_domain_loss += float(domain_loss.item())
            global_step_counter += 1

        # STRICT-UDA checkpoint selection: source_val only.
        source_val_metrics_epoch = evaluate_split(
            "_temporary_source_val_epoch",
            source_val_df,
            source_val_loader,
            save_outputs=False,
        )
        source_val_f1 = float(
            source_val_metrics_epoch["macro_f1"]
        )

        scheduler.step(source_val_f1)

        row = {
            "epoch": epoch,
            "n_steps": n_steps_per_epoch,
            "train_total_loss": (
                total_loss / n_steps_per_epoch
            ),
            "train_source_class_loss": (
                total_class_loss / n_steps_per_epoch
            ),
            "train_domain_loss": (
                total_domain_loss / n_steps_per_epoch
            ),
            "train_domain_accuracy": (
                total_domain_correct
                / max(total_domain_n, 1)
            ),
            "pcgrad_conflicts_total": (
                total_pcgrad_conflicts if method == "pcgrad" else 0.0
            ),
            "pcgrad_conflicts_per_step": (
                total_pcgrad_conflicts / n_steps_per_epoch
                if method == "pcgrad" else 0.0
            ),
            "pcgrad_cosine_mean": (
                total_pcgrad_cosine / n_steps_per_epoch
                if method == "pcgrad" else 0.0
            ),
            "pcgrad_projection_scale_mean": (
                total_pcgrad_proj_scale / n_steps_per_epoch
                if method == "pcgrad" else 0.0
            ),
            "grl_lambda_first": (
                float(lambda_values[0])
                if lambda_values
                else float("nan")
            ),
            "grl_lambda_last": (
                float(lambda_values[-1])
                if lambda_values
                else float("nan")
            ),
            "lr": float(
                optimizer.param_groups[0]["lr"]
            ),
            **{
                f"source_val_{k}": v
                for k, v in source_val_metrics_epoch.items()
            },
        }
        history.append(row)
        print(row, flush=True)

        if source_val_f1 > best_source_val_f1:
            best_source_val_f1 = source_val_f1
            best_epoch = epoch
            patience_left = DANN_EARLY_STOP_PATIENCE

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "checkpoint_role": (
                        f"best_{method}_dann_by_source_val"
                    ),
                    "epoch": epoch,
                    "best_source_val_macro_f1": (
                        best_source_val_f1
                    ),
                    **run_config,
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best epoch={best_epoch}, "
                    f"best source_val Macro-F1="
                    f"{best_source_val_f1:.6f}",
                    flush=True,
                )
                break

    pd.DataFrame(history).to_csv(
        out / "history.csv",
        index=False,
    )

    # Terminal state kept separately. Never use it for open-world.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "checkpoint_role": f"last_{method}_dann",
            "epoch": (
                int(history[-1]["epoch"])
                if history
                else 0
            ),
            "best_epoch": best_epoch,
            "best_source_val_macro_f1": (
                best_source_val_f1
            ),
            **run_config,
        },
        last_path,
    )

    if not best_path.exists():
        raise RuntimeError(
            "No best_dann.pt was produced."
        )

    # -------------------------------------------------------------------------
    # Reload source-val-selected DANN checkpoint for all AFTER metrics.
    # -------------------------------------------------------------------------
    best_payload = torch.load(
        best_path,
        map_location=device,
    )
    model.load_state_dict(
        best_payload["model_state_dict"],
        strict=True,
    )

    after_source_val = evaluate_split(
        "after_dann_source_val",
        source_val_df,
        source_val_loader,
    )
    after_source_test = evaluate_split(
        "after_dann_source_test",
        source_test_df,
        source_test_loader,
    )
    after_target_val = evaluate_split(
        "after_dann_target_val_known",
        target_val_known_df,
        target_val_known_loader,
    )
    after_target_test = evaluate_split(
        "after_dann_target_test_known",
        target_test_known_df,
        target_test_known_loader,
    )

    summary = {
        **run_config,
        "best_epoch": int(best_epoch),
        "best_dann_checkpoint": str(best_path),
        "last_dann_checkpoint": str(last_path),
        "best_source_val_macro_f1": (
            float(best_source_val_f1)
        ),
        "after_source_val_macro_f1": (
            after_source_val["macro_f1"]
        ),
        "after_source_test_macro_f1": (
            after_source_test["macro_f1"]
        ),
        "before_target_val_known_macro_f1": (
            before_target_val["macro_f1"]
        ),
        "after_target_val_known_macro_f1": (
            after_target_val["macro_f1"]
        ),
        "delta_target_val_known_macro_f1": (
            after_target_val["macro_f1"]
            - before_target_val["macro_f1"]
        ),
        "before_target_test_known_accuracy": (
            before_target_test["accuracy"]
        ),
        "before_target_test_known_balanced_accuracy": (
            before_target_test["balanced_accuracy"]
        ),
        "before_target_test_known_macro_f1": (
            before_target_test["macro_f1"]
        ),
        "before_target_test_known_macro_auc_ovr": (
            before_target_test["macro_auc_ovr"]
        ),
        "after_target_test_known_accuracy": (
            after_target_test["accuracy"]
        ),
        "after_target_test_known_balanced_accuracy": (
            after_target_test["balanced_accuracy"]
        ),
        "after_target_test_known_macro_f1": (
            after_target_test["macro_f1"]
        ),
        "after_target_test_known_macro_auc_ovr": (
            after_target_test["macro_auc_ovr"]
        ),
        "delta_target_test_known_macro_f1": (
            after_target_test["macro_f1"]
            - before_target_test["macro_f1"]
        ),
        "delta_target_test_known_accuracy": (
            after_target_test["accuracy"]
            - before_target_test["accuracy"]
        ),
        "target_test_metrics_used_for_selection": False,
        "openworld_threshold_selected": False,
        "pcgrad_conflict_rate_mean": (
            float(np.mean([h["pcgrad_conflicts_per_step"] for h in history]))
            if method == "pcgrad" and history else 0.0
        ),
        "pcgrad_cosine_mean": (
            float(np.mean([h["pcgrad_cosine_mean"] for h in history]))
            if method == "pcgrad" and history else 0.0
        ),
        "pcgrad_projection_scale_mean": (
            float(np.mean([h["pcgrad_projection_scale_mean"] for h in history]))
            if method == "pcgrad" and history else 0.0
        ),
    }

    json_dump(summary, out / "run_summary.json")
    pd.DataFrame([summary]).to_csv(
        out / "run_summary.csv",
        index=False,
    )

    print(
        "\nCOMPLETE "
        f"{method.upper()} / {target_dataset} / {model_family} / "
        f"{text_col} / seed {seed}\n"
        f"before target-test known Macro-F1 = "
        f"{before_target_test['macro_f1']:.6f}\n"
        f"after  target-test known Macro-F1 = "
        f"{after_target_test['macro_f1']:.6f}\n"
        f"delta = "
        f"{summary['delta_target_test_known_macro_f1']:+.6f}\n"
        f"best DANN selected ONLY by source_val: {best_path}",
        flush=True,
    )

    del (
        model,
        optimizer,
        scheduler,
        source_train_loader,
        target_adapt_loader,
        source_val_loader,
        source_test_loader,
        target_val_known_loader,
        target_test_known_loader,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_results() -> None:
    import numpy as np
    import pandas as pd

    rows = []
    reused_standard = 0
    for method in METHODS:
        for target in TARGET_DATASETS:
            for model_family in MODEL_FAMILIES:
                for text_col in TEXT_COLS:
                    for seed in SEEDS:
                        p = (
                            run_dir(method, target, model_family, text_col, seed)
                            / "run_summary.json"
                        )
                        # Per the review, standard DANN is FROZEN: it is not
                        # rerun for v2. If a standard run is absent from the v2
                        # output tree, fall back to the v1 tree so the pairing
                        # reuses the identical, already-computed baseline.
                        if not p.exists() and method == "standard":
                            legacy = (
                                FROZEN_STANDARD_ROOT
                                / "runs"
                                / method
                                / sanitize(target)
                                / model_family
                                / text_col
                                / f"seed_{seed}"
                                / "run_summary.json"
                            )
                            if legacy.exists():
                                p = legacy
                                reused_standard += 1
                        if p.exists():
                            with p.open("r", encoding="utf-8") as f:
                                rows.append(json.load(f))

    if reused_standard:
        log(
            f"Reused {reused_standard} FROZEN standard-DANN runs from "
            f"{FROZEN_STANDARD_ROOT} (standard DANN is intentionally not rerun)."
        )

    if not rows:
        log("No paired DANN runs available to aggregate.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_ROOT / "all_dann_paired_runs.csv", index=False)

    expected_total = (
        len(METHODS) * len(TARGET_DATASETS) * len(MODEL_FAMILIES)
        * len(TEXT_COLS) * len(SEEDS)
    )

    key_cols = ["method", "target_dataset", "model_family", "text_col", "seed"]
    duplicates = df.duplicated(subset=key_cols, keep=False)
    if duplicates.any():
        raise RuntimeError(
            "Duplicate paired-DANN run keys found:\n"
            + df.loc[duplicates, key_cols].to_string(index=False)
        )

    metrics = [
        "before_target_test_known_accuracy",
        "before_target_test_known_balanced_accuracy",
        "before_target_test_known_macro_f1",
        "before_target_test_known_macro_auc_ovr",
        "after_target_test_known_accuracy",
        "after_target_test_known_balanced_accuracy",
        "after_target_test_known_macro_f1",
        "after_target_test_known_macro_auc_ovr",
        "delta_target_test_known_macro_f1",
        "delta_target_test_known_accuracy",
        "after_source_val_macro_f1",
        "after_source_test_macro_f1",
        "pcgrad_conflict_rate_mean",
        "pcgrad_cosine_mean",
        "pcgrad_projection_scale_mean",
    ]

    agg_spec = {
        "n_seeds": ("seed", "nunique"),
        "seeds": (
            "seed",
            lambda x: ",".join(str(int(v)) for v in sorted(set(x))),
        ),
        "best_epoch_mean": ("best_epoch", "mean"),
        "best_epoch_std": ("best_epoch", "std"),
    }
    for metric in metrics:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")

    method_summary = (
        df.groupby(
            ["method", "target_dataset", "model_family", "text_col"],
            as_index=False,
        )
        .agg(**agg_spec)
        .sort_values(
            ["target_dataset", "model_family", "text_col", "method"],
            ascending=True,
        )
    )
    method_summary.to_csv(
        OUTPUT_ROOT / "dann_method_mean_std.csv", index=False
    )

    # ------------------------------------------------------------------
    # Strict paired comparison: PCGrad - Standard for SAME seed/cell.
    # ------------------------------------------------------------------
    pair_keys = ["target_dataset", "model_family", "text_col", "seed"]
    pair_metric_cols = [
        "after_target_test_known_macro_f1",
        "after_target_test_known_accuracy",
        "after_target_test_known_balanced_accuracy",
        "after_target_test_known_macro_auc_ovr",
        "delta_target_test_known_macro_f1",
        "after_source_val_macro_f1",
        "after_source_test_macro_f1",
        "best_epoch",
    ]

    # v2 geometry diagnostics carried through the pairing, so the mechanistic
    # claim ("conflict causes negative transfer, surgery removes it") can be
    # tested against cosine/projection magnitude and not just conflict counts.
    pcgrad_diag_cols = [
        c
        for c in [
            "pcgrad_conflict_rate_mean",
            "pcgrad_cosine_mean",
            "pcgrad_projection_scale_mean",
        ]
        if c in df.columns
    ]

    standard = df[df["method"] == "standard"][pair_keys + pair_metric_cols].copy()
    pcgrad = df[df["method"] == "pcgrad"][
        pair_keys + pair_metric_cols + pcgrad_diag_cols
    ].copy()

    standard = standard.rename(
        columns={c: f"standard_{c}" for c in pair_metric_cols}
    )
    pcgrad = pcgrad.rename(
        columns={c: f"pcgrad_{c}" for c in pair_metric_cols}
    )

    paired = standard.merge(pcgrad, on=pair_keys, how="outer", indicator=True)
    paired["pair_complete"] = paired["_merge"].eq("both")

    for c in pair_metric_cols:
        paired[f"pcgrad_minus_standard_{c}"] = (
            paired[f"pcgrad_{c}"] - paired[f"standard_{c}"]
        )

    paired.to_csv(
        OUTPUT_ROOT / "pcgrad_vs_standard_paired_per_seed.csv", index=False
    )

    complete_pairs = paired[paired["pair_complete"]].copy()
    delta_cols = [
        c for c in complete_pairs.columns
        if c.startswith("pcgrad_minus_standard_")
    ]

    paired_agg = {
        "n_paired_seeds": ("seed", "nunique"),
        "paired_seeds": (
            "seed",
            lambda x: ",".join(str(int(v)) for v in sorted(set(x))),
        ),
    }
    for c in pcgrad_diag_cols:
        paired_agg[c] = (c, "mean")
        paired_agg[c.replace("_mean", "_std")] = (c, "std")
    for c in delta_cols:
        paired_agg[f"{c}_mean"] = (c, "mean")
        paired_agg[f"{c}_std"] = (c, "std")

    if not complete_pairs.empty:
        paired_summary = (
            complete_pairs.groupby(
                ["target_dataset", "model_family", "text_col"],
                as_index=False,
            )
            .agg(**paired_agg)
            .sort_values(
                [
                    "target_dataset",
                    "pcgrad_minus_standard_after_target_test_known_macro_f1_mean",
                ],
                ascending=[True, False],
            )
        )
    else:
        paired_summary = pd.DataFrame()

    paired_summary.to_csv(
        OUTPUT_ROOT / "pcgrad_vs_standard_mean_std.csv", index=False
    )

    # Combined checkpoint registry for the later open-world stage.
    registry_cols = [
        "method",
        "target_dataset",
        "model_family",
        "text_col",
        "seed",
        "closed_checkpoint",
        "best_dann_checkpoint",
        "best_epoch",
        "best_source_val_macro_f1",
        "before_target_test_known_macro_f1",
        "after_target_test_known_macro_f1",
        "delta_target_test_known_macro_f1",
        "pcgrad_conflict_rate_mean",
    ]
    df[registry_cols].to_csv(
        OUTPUT_ROOT / "dann_paired_best_checkpoint_registry.csv", index=False
    )

    print("\n" + "=" * 124)
    print("PAIRED STANDARD DANN vs SHARED-PARAMETER PCGRAD-DANN (v2)")
    print("=" * 124)

    display_cols = [
        c
        for c in [
            "target_dataset",
            "model_family",
            "text_col",
            "n_paired_seeds",
            "pcgrad_minus_standard_after_target_test_known_macro_f1_mean",
            "pcgrad_minus_standard_after_target_test_known_macro_f1_std",
            "pcgrad_minus_standard_after_source_val_macro_f1_mean",
            "pcgrad_conflict_rate_mean",
            "pcgrad_cosine_mean",
        ]
        if c in paired_summary.columns
    ]
    if not paired_summary.empty:
        print(paired_summary[display_cols].to_string(index=False))

    print("\n" + "=" * 124)
    print(f"Completed run files: {len(df)}/{expected_total}")
    print(
        f"Complete standard/PCGrad seed pairs: "
        f"{int(complete_pairs.shape[0])}/108"
    )
    print(
        "Primary paired artifact: "
        f"{OUTPUT_ROOT / 'pcgrad_vs_standard_mean_std.csv'}"
    )
    print(
        "Checkpoint registry for later open-world evaluation: "
        f"{OUTPUT_ROOT / 'dann_paired_best_checkpoint_registry.csv'}"
    )
    print(
        "Target test differences are descriptive final evaluation; they did NOT "
        "select either DANN or PCGrad checkpoints."
    )
    print("=" * 124)


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def orchestrate(method_filter: str = "both") -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Mandatory provenance/data/checkpoint validation before GPU work.
    resolved_roots = preflight(force_scan=False)

    import pandas as pd
    registry = pd.read_csv(CLOSED_CHECKPOINT_REGISTRY)

    lookup = {
        (str(r.model_family), str(r.text_col), int(r.seed)): str(r.best_checkpoint)
        for r in registry.itertuples(index=False)
    }

    if method_filter == "both":
        selected_methods = METHODS
    elif method_filter in METHODS:
        selected_methods = [method_filter]
    else:
        raise ValueError(f"Unknown method filter: {method_filter}")

    # Pair-adjacent ordering: for each exact closed checkpoint/target/seed,
    # run Standard first and PCGrad second. Both subprocesses reset the same seed.
    jobs = []
    for target in TARGET_DATASETS:
        for model_family in MODEL_FAMILIES:
            for text_col in TEXT_COLS:
                for seed in SEEDS:
                    checkpoint = lookup[(model_family, text_col, seed)]
                    for method in selected_methods:
                        jobs.append(
                            (method, target, model_family, text_col, seed, checkpoint)
                        )

    total = len(jobs)
    script_path = Path(__file__).resolve()

    log("=" * 104)
    log("PAIRED STANDARD DANN + USER ASYMMETRIC PCGRAD-DANN")
    log(f"Methods this launch: {selected_methods}")
    log(f"Total worker runs this launch: {total}")
    log(
        "Everything is matched; only gradient combination differs: "
        "standard summed backward vs user's dann_pcgrad_step."
    )
    log(
        "Selection: source_val Macro-F1 ONLY; target labels not used for "
        "sampling/classification optimization."
    )
    log(f"Resolved target roots: {resolved_roots}")
    log("=" * 104)

    for i, (method, target, model_family, text_col, seed, checkpoint) in enumerate(jobs, 1):
        label = (
            f"[{i}/{total}] {method.upper()} / {target} / "
            f"{model_family} / {text_col} / seed {seed}"
        )

        if run_complete(method, target, model_family, text_col, seed):
            log(f"SKIP completed {label}")
            continue

        log(f"START {label}")

        cmd = [
            sys.executable,
            str(script_path),
            "--worker",
            "--method", method,
            "--target-dataset", target,
            "--model-family", model_family,
            "--text-col", text_col,
            "--seed", str(seed),
            "--closed-checkpoint", checkpoint,
        ]

        result = subprocess.run(cmd, cwd=str(ROOT), env=child_env())
        if result.returncode != 0:
            raise RuntimeError(
                f"{method} DANN worker failed (return code {result.returncode}): {label}"
            )

        cooldown(label)

    aggregate_results()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate V5 splits, 54 same-seed closed checkpoints, image roots, "
            "and snapshot/hash the user's pcgrad_dann.py without training."
        ),
    )
    p.add_argument(
        "--force-root-scan",
        action="store_true",
        help="Force filesystem image-root discovery during preflight.",
    )
    p.add_argument(
        "--methods",
        choices=["both", "standard", "pcgrad"],
        default="both",
        help="Default 'both' runs the complete paired ablation (216 workers).",
    )

    p.add_argument("--worker", action="store_true")
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--target-dataset", choices=TARGET_DATASETS)
    p.add_argument("--model-family", choices=MODEL_FAMILIES)
    p.add_argument("--text-col", choices=TEXT_COLS)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--closed-checkpoint")
    p.add_argument("--aggregate-only", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.aggregate_only:
        aggregate_results()

    elif args.preflight:
        preflight(force_scan=args.force_root_scan)

    elif args.worker:
        required = {
            "--method": args.method,
            "--target-dataset": args.target_dataset,
            "--model-family": args.model_family,
            "--text-col": args.text_col,
            "--seed": args.seed,
            "--closed-checkpoint": args.closed_checkpoint,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(f"--worker missing required args: {missing}")

        worker(
            method=args.method,
            target_dataset=args.target_dataset,
            model_family=args.model_family,
            text_col=args.text_col,
            seed=int(args.seed),
            closed_checkpoint=args.closed_checkpoint,
        )

    else:
        orchestrate(method_filter=args.methods)
