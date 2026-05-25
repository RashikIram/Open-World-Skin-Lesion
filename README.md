# Skin Lesion Open-World Domain Adaptation

Clean Python version of the PAD-UFES closed-set, DANN adaptation, and open-world evaluation pipeline.

The project supports four image-text fusion model families and their DANN versions:

1. MobileViT + Cross Attention
2. MobileViT + Gated Fusion
3. ResNet50 + Cross Attention
4. ResNet50 + Gated Fusion

It also supports image-only baselines and three metadata text variants: `text_core`, `text_full`, and `text_missing_explicit`.

## Structure

```text
skin_lesion_openworld/
├── README.md
├── requirements.txt
├── config.py
├── src/
│   ├── preprocessing.py
│   ├── datasets.py
│   ├── transforms.py
│   ├── models.py
│   ├── train_closed_set.py
│   ├── train_dann.py
│   ├── evaluate_openworld.py
│   ├── metrics.py
│   ├── visualization.py
│   └── utils.py
├── notebooks/
│   └── demo_run.ipynb
└── outputs/
    └── README_outputs.md
```

## Example commands

Train PAD-UFES closed-set models:

```bash
python -m src.train_closed_set \
  --padufes-csv "D:/Deep Learning/metadata.csv" \
  --padufes-image-dir "D:/Deep Learning/images" \
  --output-dir "D:/Deep Learning/output/clean_code" \
  --model-family mobilevit_gated \
  --text-col text_full
```

Run DANN adaptation:

```bash
python -m src.train_dann \
  --standardized-csv "D:/Deep Learning/preprocessed_outputs/all_preprocessed_splits_standardized_text.csv" \
  --source-dataset PAD-UFES \
  --source-image-roots "D:/Deep Learning/images" \
  --target-dataset MCR-SL \
  --target-image-roots "D:/Deep Learning/MCR-SL_dataset/dermoscopic" "D:/Deep Learning/MCR-SL_dataset/images" \
  --checkpoint "D:/Deep Learning/output/clean_code/text_full/best.pt" \
  --output-dir "D:/Deep Learning/output/clean_code/mcr_dann" \
  --model-family mobilevit_gated \
  --text-col text_full
```

Evaluate open-world unknown detection:

```bash
python -m src.evaluate_openworld \
  --standardized-csv "D:/Deep Learning/preprocessed_outputs/all_preprocessed_splits_standardized_text.csv" \
  --target-dataset MCR-SL \
  --target-image-roots "D:/Deep Learning/MCR-SL_dataset/dermoscopic" "D:/Deep Learning/MCR-SL_dataset/images" \
  --checkpoint "D:/Deep Learning/output/clean_code/mcr_dann/text_full/best_dann.pt" \
  --output-dir "D:/Deep Learning/output/clean_code/openworld" \
  --model-family mobilevit_gated \
  --text-col text_full
```

## Notes

Update paths in `config.py` or pass CLI arguments. The scripts save metrics, predictions, plots, and checkpoints under the selected output directory.
