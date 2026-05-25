from dataclasses import dataclass, field
from pathlib import Path
from typing import List


SEED = 42
KNOWN_CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SK"]
UNKNOWN_LABEL_NAME = "UNKNOWN"
TEXT_EXPERIMENTS = ["text_core", "text_full", "text_missing_explicit"]


@dataclass
class ExperimentConfig:
    padufes_csv: Path = Path("D:/Deep Learning/metadata.csv")
    padufes_image_dir: Path = Path("D:/Deep Learning/images")
    standardized_csv: Path = Path("D:/Deep Learning/preprocessed_outputs/all_preprocessed_splits_standardized_text.csv")
    output_dir: Path = Path("D:/Deep Learning/output/clean_code")
    target_image_roots: List[Path] = field(default_factory=list)

    image_model_name: str = "mobilevit_s.cvnets_in1k"
    resnet_model_name: str = "resnet50"
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    model_family: str = "mobilevit_gated"

    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 50
    patience: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_text_len: int = 96
    fusion_dim: int = 256
    num_heads: int = 4
    balance_beta: float = 0.99
    dann_lambda_max: float = 1.0
    domain_loss_weight: float = 1.0

    use_soft_weighted_sampler: bool = True
    use_weighted_loss: bool = True

    @property
    def label_to_id(self):
        return {label: idx for idx, label in enumerate(KNOWN_CLASSES)}

    @property
    def id_to_label(self):
        return {idx: label for label, idx in self.label_to_id.items()}
