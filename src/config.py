from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# ============================================================
# Global constants
# ============================================================

SEED: int = 42

KNOWN_CLASSES: List[str] = ["AK", "BCC", "MEL", "NEV", "SCC", "SK"]
UNKNOWN_LABEL_NAME: str = "UNKNOWN"

LABEL_TO_ID = {label: idx for idx, label in enumerate(KNOWN_CLASSES)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
NUM_CLASSES: int = len(KNOWN_CLASSES)


# ============================================================
# Experiment configuration
# ============================================================

@dataclass
class ExperimentConfig:
    """
    Central configuration used by the modular training/evaluation scripts.

    The scripts still expose important options through argparse. This class
    stores stable defaults shared across:
    - train_closed_set.py
    - evaluate_closed_set_transfer.py
    - train_dann.py
    - evaluate_openworld.py
    - tune_hyperparameters.py
    """

    # Reproducibility / runtime
    seed: int = SEED
    num_workers: int = 0

    # Backbones
    image_model_name: str = "mobilevit_s.cvnets_in1k"
    resnet_model_name: str = "resnet50"
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"

    # Input sizes
    image_size: int = 224
    max_text_len: int = 96

    # Training
    batch_size: int = 16
    epochs: int = 50
    patience: int = 7
    lr: float = 7.724577449307428e-05
    weight_decay: float = 4.383323192443001e-05

    # Scheduler
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3

    # Fusion head
    fusion_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.3
    freeze_backbones: bool = False

    # Class imbalance handling
    balance_beta: float = 0.9851359506352776
    use_soft_weighted_sampler: bool = True
    use_weighted_loss: bool = True

    # DANN/domain adaptation
    domain_loss_weight: float = 1.0
    dann_lambda_max: float = 1.0

    # Energy-based open-world evaluation
    energy_temperature: float = 1.0

    # Experiment text columns
    text_experiments: List[str] = field(
        default_factory=lambda: ["text_full", "text_core", "text_missing_explicit"]
    )

    # GradCAM/t-SNE
    n_gradcam_examples: int = 6
    show_plots: bool = False
