from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import KNOWN_CLASSES, SEED

LABEL_MAP = {
    "ACK": "ACK", "AK": "ACK", "ACTINIC_KERATOSIS": "ACK",
    "BCC": "BCC",
    "MEL": "MEL", "MELANOMA": "MEL",
    "NEV": "NEV", "NV": "NEV", "NEVUS": "NEV",
    "SCC": "SCC",
    "SEK": "SK", "SK": "SK", "BKL": "SK", "SEBORRHEIC_KERATOSIS": "SK",
}
IMAGE_EXTS = ["", ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def harmonize_label(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().upper().replace("-", "_").replace("/", "_").replace(" ", "_")
    return LABEL_MAP.get(x, x)


def is_missing(x) -> bool:
    if pd.isna(x):
        return True
    return str(x).strip().lower() in {"", "nan", "none", "null", "unknown", "unspecified", "na", "n/a"}


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


def standardize_location(x) -> str:
    if is_missing(x):
        return "unknown"
    x = str(x).strip().lower().replace("_", " ").replace("-", " ")
    mapping = {
        "face": "face", "scalp": "scalp", "ear": "ear", "neck": "neck",
        "head neck": "head or neck", "head/neck": "head or neck",
        "chest": "chest", "abdomen": "abdomen", "back": "back",
        "torso": "torso", "trunk": "torso",
        "arm": "upper limb", "forearm": "upper limb", "hand": "hand",
        "leg": "lower limb", "thigh": "lower limb", "foot": "foot",
    }
    return mapping.get(x, x)


def build_metadata_text(row: pd.Series) -> tuple[str, str, str]:
    age = clean_numeric(row.get("age", np.nan), min_val=0, max_val=120)
    gender = clean_text(row.get("gender", np.nan)).lower()
    if gender in {"male", "m", "man"}:
        sex = "male"
    elif gender in {"female", "f", "woman"}:
        sex = "female"
    else:
        sex = "unknown"

    location = standardize_location(row.get("region", np.nan))
    d1 = clean_numeric(row.get("diameter_1", np.nan), min_val=0, max_val=300)
    d2 = clean_numeric(row.get("diameter_2", np.nan), min_val=0, max_val=300)
    if d1 != "unknown" and d2 != "unknown":
        diameter = f"{d1} by {d2} mm"
    elif d1 != "unknown":
        diameter = f"{d1} mm"
    elif d2 != "unknown":
        diameter = f"{d2} mm"
    else:
        diameter = "unknown"

    meta = {
        "age": age,
        "sex": sex,
        "location": location,
        "diameter": diameter,
        "fitzpatrick": clean_numeric(row.get("fitspatrick", np.nan), min_val=1, max_val=6),
        "itch": clean_bool(row.get("itch", np.nan)),
        "grew": clean_bool(row.get("grew", np.nan)),
        "hurt": clean_bool(row.get("hurt", np.nan)),
        "changed": clean_bool(row.get("changed", np.nan)),
        "bleed": clean_bool(row.get("bleed", np.nan)),
        "elevation": clean_bool(row.get("elevation", np.nan)),
        "smoking": clean_bool(row.get("smoke", np.nan)),
        "alcohol": clean_bool(row.get("drink", np.nan)),
        "pesticide_exposure": clean_bool(row.get("pesticide", np.nan)),
        "personal_skin_cancer_history": clean_bool(row.get("skin_cancer_history", np.nan)),
        "general_cancer_history": clean_bool(row.get("cancer_history", np.nan)),
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
        f"Diameter: {meta['diameter']}. Fitzpatrick skin type: {meta['fitzpatrick']}. "
        f"Symptoms include itching: {meta['itch']}, growth: {meta['grew']}, pain: {meta['hurt']}, "
        f"change: {meta['changed']}, bleeding: {meta['bleed']}, elevation: {meta['elevation']}. "
        f"Smoking: {meta['smoking']}. Alcohol: {meta['alcohol']}. "
        f"Pesticide exposure: {meta['pesticide_exposure']}. "
        f"Personal skin cancer history: {meta['personal_skin_cancer_history']}. "
        f"General cancer history: {meta['general_cancer_history']}."
    )
    text_missing_explicit = text_core + " " + "; ".join(f"{k}: {v}" for k, v in sorted(meta.items()))
    return text_core, text_full, text_missing_explicit


def resolve_image_path(image_file: str, image_roots: Sequence[str | Path], exts: Sequence[str] = IMAGE_EXTS) -> str | None:
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


def prepare_padufes_dataframe(csv_path: str | Path, image_dir: str | Path, known_classes=KNOWN_CLASSES) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["label_harmonized"] = df["diagnostic"].apply(harmonize_label)
    df = df[df["label_harmonized"].isin(known_classes)].copy()
    label_to_id = {label: i for i, label in enumerate(known_classes)}
    df["label_id"] = df["label_harmonized"].map(label_to_id)
    df["image_file"] = df["img_id"].astype(str)
    texts = df.apply(build_metadata_text, axis=1)
    df["text_core"] = [x[0] for x in texts]
    df["text_full"] = [x[1] for x in texts]
    df["text_missing_explicit"] = [x[2] for x in texts]
    df["dataset"] = "PAD-UFES"
    df["image_path"] = df["image_file"].apply(lambda x: resolve_image_path(x, [image_dir]))
    return df[df["image_path"].notna()].reset_index(drop=True)


def stratified_train_val_test_split(df: pd.DataFrame, label_col="label_id", seed: int = SEED):
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=df[label_col])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df[label_col])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_standardized_splits(csv_path: str | Path, image_roots: Sequence[str | Path], dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["dataset"] == dataset_name].copy()
    if "label_harmonized" in df.columns:
        df["label_harmonized"] = df["label_harmonized"].apply(harmonize_label)
    else:
        raise ValueError("standardized CSV must contain label_harmonized")
    if "image_file" not in df.columns:
        raise ValueError("standardized CSV must contain image_file")
    df["image_path"] = df["image_file"].apply(lambda x: resolve_image_path(x, image_roots))
    return df[df["image_path"].notna()].reset_index(drop=True)


def add_known_unknown_columns(df: pd.DataFrame, known_classes=KNOWN_CLASSES) -> pd.DataFrame:
    df = df.copy()
    label_to_id = {label: i for i, label in enumerate(known_classes)}
    unknown_id = len(known_classes)
    df["is_unknown"] = ~df["label_harmonized"].isin(known_classes)
    df["label_id"] = df["label_harmonized"].map(label_to_id)
    df["label_open_id"] = df["label_id"]
    df.loc[df["is_unknown"], "label_open_id"] = unknown_id
    df["label_open_id"] = df["label_open_id"].astype(int)
    return df
