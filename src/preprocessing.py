from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from config import KNOWN_CLASSES, SEED
except ImportError:  # Keeps this module importable during standalone preprocessing checks.
    KNOWN_CLASSES = ["AK", "BCC", "MEL", "NEV", "SCC", "SK"]
    SEED = 42

# Core known classes learned from PAD-UFES.
DEFAULT_KNOWN_CLASSES = list(KNOWN_CLASSES)

# Unknown/open-world target classes used in the original preprocessing script.
ISIC_UNKNOWN_CLASSES = ["ANG", "DF", "UNK"]
MCR_UNKNOWN_CLASSES = ["ANG", "ATY", "DF"]
UNKNOWN_LABEL_NAME = "UNKNOWN"

# The original script used RANDOM_STATE. The modular package uses SEED.
RANDOM_STATE = SEED

# Supported extensions for image path resolution.
IMAGE_EXTS = ["", ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

# Full harmonisation map from the closed-set notebook + raw preprocessing script.
LABEL_MAP = {
    "MEL": "MEL",
    "MELANOMA": "MEL",
    "BCC": "BCC",
    "SCC": "SCC",
    "DF": "DF",
    "ATY": "ATY",
    "UNK": "UNK",

    "NV": "NEV",
    "NEV": "NEV",
    "NEVUS": "NEV",

    # Important: ACK/AK must become AK because KNOWN_CLASSES uses AK.
    "AK": "AK",
    "ACK": "AK",
    "ACTINIC_KERATOSIS": "AK",

    "SK": "SK",
    "SEK": "SK",
    "BKL": "SK",
    "SEBORRHEIC_KERATOSIS": "SK",
    "BENIGN_KERATOSIS": "SK",

    "ANG": "ANG",
    "VASC": "ANG",
    "ANGIOMA": "ANG",

    "BOWEN_CARCINOMA": "SCC",
}

COMMON_COLS = [
    "dataset", "role", "split",
    "sample_id", "image_id", "image_file",
    "label_raw", "label_harmonized", "known_unknown",
    "age_clean", "sex_clean", "site_clean",
    "text_core", "text_full", "text_missing_explicit",
]

PAD_EXTRA_COLS = [
    "patient_id", "lesion_id",
    "smoke", "drink", "pesticide",
    "skin_cancer_history", "cancer_history",
    "fitspatrick",
    "diameter_1_clean", "diameter_2_clean",
    "itch", "grew", "hurt", "changed", "bleed", "elevation",
    "biopsed",
]

ISIC_EXTRA_COLS = [
    "lesion_id", "age_approx", "anatom_site_general", "sex",
    "score_weight", "validation_weight",
]

MCR_EXTRA_COLS = [
    "lesion_id", "subject_id",
    "referral_diagnosis",
    "lesion_status_when_captured",
    "location", "location_group",
    "diameter_clean",
    "malignancy",
    "natural_hair_color",
    "skin_reaction_to_sun",
    "moles_body_18", "moles_bigger_5mm", "moles_bigger_20cm", "moles_body",
    "sunburn_number", "sunburn_number_group",
    "sunbed",
    "h_cancer", "h_skin_cancer", "h_skin_cancer_relatives",
    "organ_transplant", "immunosuppresion",
]

TEXT_COLS = ["text_core", "text_full", "text_missing_explicit"]


def normalize_label(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().upper()
    x = x.replace("-", "_").replace("/", "_").replace(" ", "_")
    return x


def harmonize_label(x):
    x = normalize_label(x)
    if pd.isna(x):
        return np.nan
    return LABEL_MAP.get(x, x)


def is_missing(x) -> bool:
    if pd.isna(x):
        return True
    return str(x).strip().lower() in {
        "", "nan", "none", "null", "unknown", "unspecified", "na", "n/a",
    }


def safe_text(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip()


def clean_text(x) -> str:
    return "unknown" if is_missing(x) else str(x).strip()


def clean_bool(x) -> str:
    if is_missing(x):
        return "unknown"
    x = str(x).strip().lower()
    if x in {"true", "1", "yes", "y"}:
        return "yes"
    if x in {"false", "0", "no", "n"}:
        return "no"
    return "unknown"


def clean_numeric(x, min_val=None, max_val=None) -> str:
    if is_missing(x):
        return "unknown"
    try:
        v = float(x)
        if min_val is not None and v < min_val:
            return "unknown"
        if max_val is not None and v > max_val:
            return "unknown"
        return str(int(v)) if v.is_integer() else f"{v:.1f}"
    except Exception:
        return "unknown"


def first_available(row: pd.Series, columns: Sequence[str], default=np.nan):
    """Return the first non-missing value from a row across possible column names."""
    for col in columns:
        if col in row.index and not is_missing(row.get(col)):
            return row.get(col)
    return default


def standardize_sex(x) -> str:
    if is_missing(x):
        return "unknown"
    x = str(x).strip().lower()
    if x in {"male", "m", "man"}:
        return "male"
    if x in {"female", "f", "woman"}:
        return "female"
    return "unknown"


def standardize_location(x) -> str:
    if is_missing(x):
        return "unknown"
    x = str(x).strip().lower().replace("_", " ").replace("-", " ")
    mapping = {
        "face": "face",
        "scalp": "scalp",
        "ear": "ear",
        "neck": "neck",
        "head neck": "head or neck",
        "head/neck": "head or neck",
        "head or neck": "head or neck",
        "chest": "chest",
        "abdomen": "abdomen",
        "back": "back",
        "torso": "torso",
        "trunk": "torso",
        "upper extremity": "upper limb",
        "upper limb": "upper limb",
        "arm": "upper limb",
        "forearm": "upper limb",
        "hand": "hand",
        "lower extremity": "lower limb",
        "lower limb": "lower limb",
        "leg": "lower limb",
        "thigh": "lower limb",
        "foot": "foot",
        "palms/soles": "palms or soles",
        "palms soles": "palms or soles",
        "oral/genital": "oral or genital",
        "oral genital": "oral or genital",
    }
    return mapping.get(x, x)


def build_metadata_text(row: pd.Series) -> tuple[str, str, str]:
    """
    Build the three text inputs used by the closed-set and target-domain scripts.

    This is robust to PAD-UFES raw columns and standardized ISIC/MCR columns.
    """
    age = clean_numeric(
        first_available(row, ["age_clean", "age", "age_approx"]),
        min_val=0,
        max_val=120,
    )

    sex = standardize_sex(
        first_available(row, ["sex_clean", "gender", "sex"]),
    )

    location = standardize_location(
        first_available(row, ["site_clean", "region", "anatom_site_general", "location_group", "location"]),
    )

    d1 = clean_numeric(
        first_available(row, ["diameter_1_clean", "diameter_1"]),
        min_val=0,
        max_val=300,
    )
    d2 = clean_numeric(
        first_available(row, ["diameter_2_clean", "diameter_2"]),
        min_val=0,
        max_val=300,
    )
    diameter_clean = clean_numeric(
        first_available(row, ["diameter_clean", "diameter"]),
        min_val=0,
        max_val=300,
    )

    if d1 != "unknown" and d2 != "unknown":
        diameter = f"{d1} by {d2} mm"
    elif d1 != "unknown":
        diameter = f"{d1} mm"
    elif d2 != "unknown":
        diameter = f"{d2} mm"
    elif diameter_clean != "unknown":
        diameter = f"{diameter_clean} mm"
    else:
        diameter = "unknown"

    meta = {
        "age": age,
        "sex": sex,
        "location": location,
        "diameter": diameter,
        "fitzpatrick": clean_numeric(first_available(row, ["fitspatrick", "fitzpatrick"]), min_val=1, max_val=6),
        "itch": clean_bool(row.get("itch", np.nan)),
        "grew": clean_bool(row.get("grew", np.nan)),
        "hurt": clean_bool(row.get("hurt", np.nan)),
        "changed": clean_bool(row.get("changed", np.nan)),
        "bleed": clean_bool(row.get("bleed", np.nan)),
        "elevation": clean_bool(row.get("elevation", np.nan)),
        "smoking": clean_bool(row.get("smoke", np.nan)),
        "alcohol": clean_bool(row.get("drink", np.nan)),
        "pesticide_exposure": clean_bool(row.get("pesticide", np.nan)),
        "personal_skin_cancer_history": clean_bool(first_available(row, ["skin_cancer_history", "h_skin_cancer"])),
        "general_cancer_history": clean_bool(first_available(row, ["cancer_history", "h_cancer"])),
        "family_skin_cancer_history": clean_bool(row.get("h_skin_cancer_relatives", np.nan)),
        "sunbed_use": clean_bool(row.get("sunbed", np.nan)),
        "immunosuppression": clean_bool(first_available(row, ["immunosuppresion", "immunosuppression"])),
    }

    age_text = "an unknown age" if meta["age"] == "unknown" else f"{meta['age']} years old"
    sex_text = "unspecified biological sex" if meta["sex"] == "unknown" else meta["sex"]
    loc_text = "an unspecified anatomical location" if meta["location"] == "unknown" else meta["location"]

    text_core = (
        f"A clinical image of a skin lesion located on {loc_text} "
        f"from a patient who is {age_text} with {sex_text}."
    )

    text_full = (
        text_core + " "
        f"Diameter: {meta['diameter']}. "
        f"Fitzpatrick skin type: {meta['fitzpatrick']}. "
        f"Symptoms include itching: {meta['itch']}, growth: {meta['grew']}, pain: {meta['hurt']}, "
        f"change: {meta['changed']}, bleeding: {meta['bleed']}, elevation: {meta['elevation']}. "
        f"Smoking: {meta['smoking']}. Alcohol: {meta['alcohol']}. "
        f"Pesticide exposure: {meta['pesticide_exposure']}. "
        f"Personal skin cancer history: {meta['personal_skin_cancer_history']}. "
        f"General cancer history: {meta['general_cancer_history']}. "
        f"Family skin cancer history: {meta['family_skin_cancer_history']}. "
        f"Sunbed use: {meta['sunbed_use']}. "
        f"Immunosuppression: {meta['immunosuppression']}."
    )

    text_missing_explicit = text_core + " " + "; ".join(
        f"{k}: {v}" for k, v in sorted(meta.items())
    )

    return text_core, text_full, text_missing_explicit


def add_text_columns(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """Add text_core, text_full, and text_missing_explicit to any standardized dataframe."""
    df = df.copy()
    missing_text_cols = [c for c in TEXT_COLS if c not in df.columns]
    if overwrite or missing_text_cols:
        texts = df.apply(build_metadata_text, axis=1)
        if overwrite or "text_core" not in df.columns:
            df["text_core"] = [x[0] for x in texts]
        if overwrite or "text_full" not in df.columns:
            df["text_full"] = [x[1] for x in texts]
        if overwrite or "text_missing_explicit" not in df.columns:
            df["text_missing_explicit"] = [x[2] for x in texts]
    return df


def validate_text_columns(df: pd.DataFrame, required_cols: Sequence[str] = TEXT_COLS) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "standardized CSV is missing required text column(s): "
            + ", ".join(missing)
            + ". Run add_text_columns() or prepare_all_standardized_splits(...)."
        )


def onehot_to_label(df: pd.DataFrame, label_cols: Sequence[str]) -> pd.Series:
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing one-hot label column(s): {missing}")

    def row_to_label(row):
        active = [c for c in label_cols if row[c] == 1 or row[c] == 1.0]
        if len(active) == 1:
            return active[0]
        if len(active) == 0:
            return np.nan
        return "|".join(active)

    return df[list(label_cols)].apply(row_to_label, axis=1)


def split_source(df: pd.DataFrame, label_col: str = "label_harmonized", seed: int = SEED):
    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df[label_col],
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["split"] = "source_train"
    val_df["split"] = "source_val"
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def split_target_train_pool_for_adapt_val(
    df: pd.DataFrame,
    known_classes: Sequence[str],
    unknown_classes: Sequence[str],
    label_col: str = "label_harmonized",
    seed: int = SEED,
):
    """
    Used for ISIC official TRAINING file.

    Logic:
    - target_adapt gets known classes only.
    - target_val gets known + unknown classes.
    - No target_test is created from the training file.
    """
    keep_classes = list(known_classes) + list(unknown_classes)
    df = df[df[label_col].isin(keep_classes)].copy()
    df["known_unknown"] = np.where(df[label_col].isin(known_classes), "known", "unknown")

    known_df = df[df["known_unknown"] == "known"].copy()
    unknown_df = df[df["known_unknown"] == "unknown"].copy()

    known_adapt, known_val = train_test_split(
        known_df,
        test_size=0.20,
        random_state=seed,
        stratify=known_df[label_col],
    )

    target_adapt = known_adapt.copy()
    target_val = pd.concat([known_val, unknown_df], ignore_index=True)

    target_adapt = target_adapt.sample(frac=1, random_state=seed).reset_index(drop=True)
    target_val = target_val.sample(frac=1, random_state=seed).reset_index(drop=True)

    target_adapt["split"] = "target_adapt"
    target_val["split"] = "target_val"

    return target_adapt, target_val


def prepare_official_target_test(
    df: pd.DataFrame,
    known_classes: Sequence[str],
    unknown_classes: Sequence[str],
    label_col: str = "label_harmonized",
    seed: int = SEED,
):
    """Used for official ISIC test file. Keeps known + unknown classes and marks known/unknown."""
    keep_classes = list(known_classes) + list(unknown_classes)
    df = df[df[label_col].isin(keep_classes)].copy()
    df["known_unknown"] = np.where(df[label_col].isin(known_classes), "known", "unknown")
    df["split"] = "target_test"
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def split_target_no_official_test(
    df: pd.DataFrame,
    known_classes: Sequence[str],
    unknown_classes: Sequence[str],
    label_col: str = "label_harmonized",
    seed: int = SEED,
):
    """
    Used for MCR-SL because it has no official train/test partition.

    Logic:
    - known target samples: 60/20/20 into adapt/val/test.
    - unknown target samples: no adaptation, split into val/test only.
    """
    keep_classes = list(known_classes) + list(unknown_classes)
    df = df[df[label_col].isin(keep_classes)].copy()
    df["known_unknown"] = np.where(df[label_col].isin(known_classes), "known", "unknown")

    target_known = df[df["known_unknown"] == "known"].copy()
    target_unknown = df[df["known_unknown"] == "unknown"].copy()

    known_adapt, known_temp = train_test_split(
        target_known,
        test_size=0.40,
        random_state=seed,
        stratify=target_known[label_col],
    )

    known_val, known_test = train_test_split(
        known_temp,
        test_size=0.50,
        random_state=seed,
        stratify=known_temp[label_col],
    )

    unknown_val_parts = []
    unknown_test_parts = []

    for _, grp in target_unknown.groupby(label_col):
        grp = grp.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(grp)
        if n == 1:
            val_part = grp.iloc[:0]
            test_part = grp.iloc[:1]
        else:
            n_val = n // 2
            val_part = grp.iloc[:n_val]
            test_part = grp.iloc[n_val:]
        unknown_val_parts.append(val_part)
        unknown_test_parts.append(test_part)

    unknown_val = (
        pd.concat(unknown_val_parts, ignore_index=True)
        if unknown_val_parts else pd.DataFrame(columns=df.columns)
    )
    unknown_test = (
        pd.concat(unknown_test_parts, ignore_index=True)
        if unknown_test_parts else pd.DataFrame(columns=df.columns)
    )

    target_adapt = known_adapt.copy()
    target_val = pd.concat([known_val, unknown_val], ignore_index=True)
    target_test = pd.concat([known_test, unknown_test], ignore_index=True)

    target_adapt = target_adapt.sample(frac=1, random_state=seed).reset_index(drop=True)
    target_val = target_val.sample(frac=1, random_state=seed).reset_index(drop=True)
    target_test = target_test.sample(frac=1, random_state=seed).reset_index(drop=True)

    target_adapt["split"] = "target_adapt"
    target_val["split"] = "target_val"
    target_test["split"] = "target_test"

    return target_adapt, target_val, target_test


def report(df: pd.DataFrame, name: str) -> None:
    print(f"\n{'=' * 80}")
    print(name)
    print(f"{'=' * 80}")
    print("Shape:", df.shape)
    if "label_harmonized" in df.columns:
        print("\nClass distribution:")
        print(df["label_harmonized"].value_counts().sort_index())
    if "known_unknown" in df.columns:
        print("\nKnown/Unknown:")
        print(df["known_unknown"].value_counts())


def keep_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].copy()


def resolve_image_path(
    image_file: str,
    image_roots: Sequence[str | Path],
    exts: Sequence[str] = IMAGE_EXTS,
) -> str | None:
    image_file = str(image_file)
    for root in image_roots:
        root = Path(root)
        for ext in exts:
            candidate = root / image_file
            if ext and not str(candidate).lower().endswith(ext.lower()):
                candidate = root / f"{image_file}{ext}"
            if candidate.exists():
                return str(candidate)
    return None


def prepare_padufes_dataframe(
    csv_path: str | Path,
    image_dir: str | Path,
    known_classes: Sequence[str] = DEFAULT_KNOWN_CLASSES,
) -> pd.DataFrame:
    """Prepare PAD-UFES for the closed-set baseline train/val/test split."""
    df = pd.read_csv(csv_path)
    df["label_harmonized"] = df["diagnostic"].apply(harmonize_label)
    df = df[df["label_harmonized"].isin(known_classes)].copy()

    label_to_id = {label: i for i, label in enumerate(known_classes)}
    df["label_id"] = df["label_harmonized"].map(label_to_id)
    df["image_file"] = df["img_id"].astype(str)
    df = add_text_columns(df, overwrite=True)
    df["dataset"] = "PAD-UFES"
    df["image_path"] = df["image_file"].apply(lambda x: resolve_image_path(x, [image_dir]))
    return df[df["image_path"].notna()].reset_index(drop=True)


def stratified_train_val_test_split(
    df: pd.DataFrame,
    label_col: str = "label_id",
    seed: int = SEED,
):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df[label_col],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df[label_col],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_padufes_source_splits(
    padufes_csv: str | Path,
    known_classes: Sequence[str] = DEFAULT_KNOWN_CLASSES,
    seed: int = SEED,
):
    """Prepare PAD-UFES source_train/source_val rows for domain adaptation."""
    pad = pd.read_csv(padufes_csv)

    pad["dataset"] = "PAD-UFES-20"
    pad["role"] = "source"
    pad["sample_id"] = pad["img_id"]
    pad["image_id"] = pad["img_id"]
    pad["image_file"] = pad["img_id"]

    pad["label_raw"] = pad["diagnostic"]
    pad["label_harmonized"] = pad["label_raw"].apply(harmonize_label)

    pad["age_clean"] = pad["age"]
    pad["sex_clean"] = pad["gender"].apply(safe_text)
    pad["site_clean"] = pad["region"].apply(safe_text)

    pad["diameter_1_clean"] = pad.get("diameter_1", np.nan)
    pad["diameter_2_clean"] = pad.get("diameter_2", np.nan)
    pad["known_unknown"] = "known"

    pad_source = pad[pad["label_harmonized"].isin(known_classes)].copy()
    pad_source = add_text_columns(pad_source, overwrite=True)
    return split_source(pad_source, seed=seed)


def prepare_isic_splits(
    train_groundtruth_csv: str | Path,
    train_metadata_csv: str | Path,
    test_groundtruth_csv: str | Path,
    test_metadata_csv: str | Path,
    known_classes: Sequence[str] = DEFAULT_KNOWN_CLASSES,
    unknown_classes: Sequence[str] = ISIC_UNKNOWN_CLASSES,
    seed: int = SEED,
):
    """Prepare ISIC target_adapt/target_val/target_test rows from official train/test files."""
    isic_label_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

    isic_train_gt = pd.read_csv(train_groundtruth_csv)
    isic_train_meta = pd.read_csv(train_metadata_csv)
    isic_train_gt["label_raw"] = onehot_to_label(isic_train_gt, isic_label_cols)
    isic_train_gt["label_harmonized"] = isic_train_gt["label_raw"].apply(harmonize_label)

    isic_train = isic_train_gt.merge(isic_train_meta, on="image", how="left")
    isic_train["dataset"] = "ISIC 2019"
    isic_train["role"] = "target"
    isic_train["sample_id"] = isic_train["image"]
    isic_train["image_id"] = isic_train["image"]
    isic_train["image_file"] = isic_train["image"].astype(str) + ".jpg"
    isic_train["age_clean"] = isic_train.get("age_approx", np.nan)
    isic_train["sex_clean"] = isic_train.get("sex", np.nan).apply(safe_text)
    isic_train["site_clean"] = isic_train.get("anatom_site_general", np.nan).apply(safe_text)
    isic_train = add_text_columns(isic_train, overwrite=True)

    isic_adapt, isic_val = split_target_train_pool_for_adapt_val(
        isic_train,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        seed=seed,
    )

    isic_test_gt = pd.read_csv(test_groundtruth_csv)
    isic_test_meta = pd.read_csv(test_metadata_csv)
    isic_test_gt["label_raw"] = onehot_to_label(isic_test_gt, isic_label_cols)
    isic_test_gt["label_harmonized"] = isic_test_gt["label_raw"].apply(harmonize_label)

    isic_test = isic_test_gt.merge(isic_test_meta, on="image", how="left")
    isic_test["dataset"] = "ISIC 2019"
    isic_test["role"] = "target"
    isic_test["sample_id"] = isic_test["image"]
    isic_test["image_id"] = isic_test["image"]
    isic_test["image_file"] = isic_test["image"].astype(str) + ".jpg"
    isic_test["age_clean"] = isic_test.get("age_approx", np.nan)
    isic_test["sex_clean"] = isic_test.get("sex", np.nan).apply(safe_text)
    isic_test["site_clean"] = isic_test.get("anatom_site_general", np.nan).apply(safe_text)
    isic_test = add_text_columns(isic_test, overwrite=True)

    isic_test = prepare_official_target_test(
        isic_test,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        seed=seed,
    )

    return isic_adapt, isic_val, isic_test


def prepare_mcr_splits(
    unified_xlsx: str | Path,
    lesion_xlsx: str | Path,
    subject_xlsx: str | Path,
    image_xlsx: str | Path | None = None,
    dermatology_diagnosis_xlsx: str | Path | None = None,
    histopathology_diagnosis_xlsx: str | Path | None = None,
    known_classes: Sequence[str] = DEFAULT_KNOWN_CLASSES,
    unknown_classes: Sequence[str] = MCR_UNKNOWN_CLASSES,
    include_mcr_unk: bool = False,
    seed: int = SEED,
):
    """Prepare MCR-SL target_adapt/target_val/target_test rows."""
    if include_mcr_unk and "UNK" not in unknown_classes:
        unknown_classes = list(unknown_classes) + ["UNK"]

    # The image/dermatology/histopathology files are accepted for API compatibility with
    # the original script. The original logic only used unified + lesion + subject.
    _ = image_xlsx, dermatology_diagnosis_xlsx, histopathology_diagnosis_xlsx

    mcr_unified = pd.read_excel(unified_xlsx)
    mcr_lesion = pd.read_excel(lesion_xlsx)
    mcr_subject = pd.read_excel(subject_xlsx)

    mcr = mcr_unified.copy()
    mcr["label_raw"] = mcr["unified_diagnosis"]
    mcr["label_harmonized"] = mcr["label_raw"].apply(harmonize_label)

    mcr = mcr.merge(mcr_lesion, on="lesion_id", how="left", suffixes=("", "_lesion"))
    mcr = mcr.merge(mcr_subject, on="subject_id", how="left", suffixes=("", "_subject"))

    mcr["dataset"] = "MCR-SL"
    mcr["role"] = "target"
    mcr["sample_id"] = mcr["lesion_id"]

    # Default from the original script: diagnosis image id.
    mcr["image_id"] = mcr["diagnosis_image_id"]
    mcr["image_file"] = mcr["diagnosis_image_id"]

    mcr["age_clean"] = mcr.get("age", np.nan)
    mcr["sex_clean"] = mcr.get("sex", np.nan).apply(safe_text)
    mcr["site_clean"] = mcr.get("location_group", np.nan).apply(safe_text)
    mcr["diameter_clean"] = mcr.get("diameter", np.nan)
    mcr = add_text_columns(mcr, overwrite=True)

    return split_target_no_official_test(
        mcr,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        seed=seed,
    )


def prepare_all_standardized_splits(
    output_dir: str | Path,
    padufes_csv: str | Path,
    isic_train_groundtruth_csv: str | Path,
    isic_train_metadata_csv: str | Path,
    isic_test_groundtruth_csv: str | Path,
    isic_test_metadata_csv: str | Path,
    mcr_unified_xlsx: str | Path,
    mcr_lesion_xlsx: str | Path,
    mcr_subject_xlsx: str | Path,
    mcr_image_xlsx: str | Path | None = None,
    mcr_dermatology_diagnosis_xlsx: str | Path | None = None,
    mcr_histopathology_diagnosis_xlsx: str | Path | None = None,
    known_classes: Sequence[str] = DEFAULT_KNOWN_CLASSES,
    isic_unknown_classes: Sequence[str] = ISIC_UNKNOWN_CLASSES,
    mcr_unknown_classes: Sequence[str] = MCR_UNKNOWN_CLASSES,
    include_mcr_unk: bool = False,
    seed: int = SEED,
    write_individual_csvs: bool = True,
    print_reports: bool = True,
) -> pd.DataFrame:
    """
    Build the full standardized dataframe expected by the DANN/open-world scripts.

    Outputs include the three text columns expected by model code:
    text_core, text_full, text_missing_explicit.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_train, source_val = prepare_padufes_source_splits(
        padufes_csv=padufes_csv,
        known_classes=known_classes,
        seed=seed,
    )

    isic_adapt, isic_val, isic_test = prepare_isic_splits(
        train_groundtruth_csv=isic_train_groundtruth_csv,
        train_metadata_csv=isic_train_metadata_csv,
        test_groundtruth_csv=isic_test_groundtruth_csv,
        test_metadata_csv=isic_test_metadata_csv,
        known_classes=known_classes,
        unknown_classes=isic_unknown_classes,
        seed=seed,
    )

    mcr_adapt, mcr_val, mcr_test = prepare_mcr_splits(
        unified_xlsx=mcr_unified_xlsx,
        lesion_xlsx=mcr_lesion_xlsx,
        subject_xlsx=mcr_subject_xlsx,
        image_xlsx=mcr_image_xlsx,
        dermatology_diagnosis_xlsx=mcr_dermatology_diagnosis_xlsx,
        histopathology_diagnosis_xlsx=mcr_histopathology_diagnosis_xlsx,
        known_classes=known_classes,
        unknown_classes=mcr_unknown_classes,
        include_mcr_unk=include_mcr_unk,
        seed=seed,
    )

    source_train_final = keep_cols(source_train, COMMON_COLS + PAD_EXTRA_COLS)
    source_val_final = keep_cols(source_val, COMMON_COLS + PAD_EXTRA_COLS)
    isic_adapt_final = keep_cols(isic_adapt, COMMON_COLS + ISIC_EXTRA_COLS)
    isic_val_final = keep_cols(isic_val, COMMON_COLS + ISIC_EXTRA_COLS)
    isic_test_final = keep_cols(isic_test, COMMON_COLS + ISIC_EXTRA_COLS)
    mcr_adapt_final = keep_cols(mcr_adapt, COMMON_COLS + MCR_EXTRA_COLS)
    mcr_val_final = keep_cols(mcr_val, COMMON_COLS + MCR_EXTRA_COLS)
    mcr_test_final = keep_cols(mcr_test, COMMON_COLS + MCR_EXTRA_COLS)

    if print_reports:
        report(source_train_final, "SOURCE TRAIN: PAD-UFES")
        report(source_val_final, "SOURCE VAL: PAD-UFES")
        report(isic_adapt_final, "TARGET ADAPT: ISIC OFFICIAL TRAINING FILE")
        report(isic_val_final, "TARGET VAL: ISIC OFFICIAL TRAINING FILE")
        report(isic_test_final, "TARGET TEST: ISIC OFFICIAL TEST FILE")
        report(mcr_adapt_final, "TARGET ADAPT: MCR-SL")
        report(mcr_val_final, "TARGET VAL: MCR-SL")
        report(mcr_test_final, "TARGET TEST: MCR-SL")

    if write_individual_csvs:
        source_train_final.to_csv(output_dir / "padufes_source_train.csv", index=False)
        source_val_final.to_csv(output_dir / "padufes_source_val.csv", index=False)
        isic_adapt_final.to_csv(output_dir / "isic_target_adapt.csv", index=False)
        isic_val_final.to_csv(output_dir / "isic_target_val.csv", index=False)
        isic_test_final.to_csv(output_dir / "isic_target_test_official.csv", index=False)
        mcr_adapt_final.to_csv(output_dir / "mcr_target_adapt.csv", index=False)
        mcr_val_final.to_csv(output_dir / "mcr_target_val.csv", index=False)
        mcr_test_final.to_csv(output_dir / "mcr_target_test.csv", index=False)

    combined = pd.concat(
        [
            source_train_final,
            source_val_final,
            isic_adapt_final,
            isic_val_final,
            isic_test_final,
            mcr_adapt_final,
            mcr_val_final,
            mcr_test_final,
        ],
        ignore_index=True,
    )

    combined = add_text_columns(combined, overwrite=False)
    validate_text_columns(combined)

    # Save both names for compatibility with older scripts and the markdown notebooks.
    combined.to_csv(output_dir / "all_preprocessed_splits.csv", index=False)
    combined.to_csv(output_dir / "all_preprocessed_splits_standardized_text.csv", index=False)

    print(f"\nSaved preprocessing outputs to: {output_dir}")
    print(f"Combined shape: {combined.shape}")
    print(f"Combined standardized-text CSV: {output_dir / 'all_preprocessed_splits_standardized_text.csv'}")

    return combined


def load_standardized_splits(
    csv_path: str | Path,
    image_roots: Sequence[str | Path],
    dataset_name: str,
    required_text_col: str | None = None,
    add_missing_text: bool = True,
) -> pd.DataFrame:
    """
    Load one dataset from a standardized CSV and resolve image paths.

    If the CSV lacks text columns, they are generated by default. If required_text_col
    is supplied, this function validates that the requested column exists.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    if "dataset" not in df.columns:
        raise ValueError("standardized CSV must contain dataset")

    df = df[df["dataset"] == dataset_name].copy()

    if "label_harmonized" not in df.columns:
        raise ValueError("standardized CSV must contain label_harmonized")
    df["label_harmonized"] = df["label_harmonized"].apply(harmonize_label)

    if "image_file" not in df.columns:
        raise ValueError("standardized CSV must contain image_file")

    if add_missing_text:
        df = add_text_columns(df, overwrite=False)

    if required_text_col is not None and required_text_col != "image_only":
        validate_text_columns(df, [required_text_col])
    else:
        validate_text_columns(df, TEXT_COLS)

    df["image_path"] = df["image_file"].apply(lambda x: resolve_image_path(x, image_roots))
    return df[df["image_path"].notna()].reset_index(drop=True)


def add_known_unknown_columns(
    df: pd.DataFrame,
    known_classes: Sequence[str] = DEFAULT_KNOWN_CLASSES,
) -> pd.DataFrame:
    df = df.copy()
    if "label_harmonized" not in df.columns:
        raise ValueError("DataFrame must contain label_harmonized")

    df["label_harmonized"] = df["label_harmonized"].apply(harmonize_label)
    label_to_id = {label: i for i, label in enumerate(known_classes)}
    unknown_id = len(known_classes)

    df["is_unknown"] = ~df["label_harmonized"].isin(known_classes)
    df["label_id"] = df["label_harmonized"].map(label_to_id)
    df["label_open_id"] = df["label_id"]
    df.loc[df["is_unknown"], "label_open_id"] = unknown_id
    df["label_open_id"] = df["label_open_id"].astype(int)
    return df
