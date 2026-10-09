# Skin Lesion Open-World Domain Adaptation

Clean Python version of the PAD-UFES closed-set, DANN adaptation, and open-world evaluation pipeline.

The project supports four image-text fusion model families and their DANN versions:

1. MobileViT + Cross Attention
2. MobileViT + Gated Fusion
3. MobileViT + Concatenation
4. ResNet50 + Cross Attention
5. ResNet50 + Gated Fusion
6. ResNet50 + Concatenation

It also supports image-only baselines and three metadata text variants: `text_core`, `text_full`, and `text_missing_explicit`.

All six fusion families can also be adapted with **UADAL** (Unknown-Aware Domain Adversarial Learning, Jang et al., NeurIPS 2022), an open-set domain adaptation method. See [UADAL open-set adaptation](#uadal-open-set-adaptation).

## Structure

```text
skin_lesion_openworld/
├── README.md
├── requirements.txt
├── config.py
├── demo.ipynb
├── src/
│   ├── preprocessing.py
│   ├── datasets.py
│   ├── transforms.py
│   ├── models.py
│   ├── train_closed_set.py
│   ├── train_dann.py
│   ├── train_uadal.py
│   ├── uadal.py
│   ├── evaluate_openworld.py
│   ├── metrics.py
│   ├── visualization.py
│   └── utils.py
├── notebooks/
│   └── notebooks_related_to_the_steps.ipynb
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

## UADAL open-set adaptation

`src/train_uadal.py` ports UADAL from the official repository
(<https://github.com/JoonHo-Jang/UADAL>, `models/model_UADAL.py`) into this pipeline.

DANN aligns every target image with the source, including images from classes the source never saw. UADAL instead:

1. **Warms up on the source.** It trains a (K+1)-way classifier `C` (index K = UNKNOWN) and a K-way open-set recognizer `E`.
2. **Estimates p(unknown) for every unlabeled target image.** It fits a two-component beta mixture to `E`'s normalized prediction entropy. The fit is repeated every `--bmm-update-every` epochs, and `E` is re-initialized after each fit.
3. **Adapts with a 3-way discriminator** (source / target-known / target-unknown). Target images it believes are known are pulled toward the source. Images it believes are unknown are pushed away and taught to predict UNKNOWN.
4. **Adds a pseudo-label consistency loss** between a weak and a strong (RandAugment) view of each target image.

Protocol differences from `train_dann.py`:

- The `target_adapt` set **keeps its unknown-class images** because open-set adaptation needs them. Target labels never enter a training loss.
- Target batches are plainly shuffled. Only the source loader is class-balanced.
- Open-world prediction needs no threshold: an image is UNKNOWN when the argmax over the K+1 outputs is UNKNOWN (`*_native` outputs). For comparison with DANN's energy protocol, a threshold on p(unknown) tuned on `target_val` is also reported (`*_tuned` outputs).

Run UADAL, starting from a closed-set checkpoint (recommended):

```bash
python -m src.train_uadal \
  --standardized-csv "D:/Deep Learning/preprocessed_outputs/all_preprocessed_splits_standardized_text.csv" \
  --source-dataset PAD-UFES \
  --source-image-roots "D:/Deep Learning/images" \
  --target-dataset MCR-SL \
  --target-image-roots "D:/Deep Learning/MCR-SL_dataset/dermoscopic" "D:/Deep Learning/MCR-SL_dataset/images" \
  --checkpoint "D:/Deep Learning/output/clean_code/mobilevit_gated/text_full/best.pt" \
  --output-dir "D:/Deep Learning/output/clean_code/mcr_uadal" \
  --model-family mobilevit_gated \
  --text-col text_full \
  --seed 42
```

The K-way checkpoint is loaded into the first K rows of the K+1 classifier. Only the UNKNOWN output, `E` and the discriminator start from scratch. A DANN checkpoint also works; its 2-way domain head is ignored.

Main UADAL options (defaults follow the official `run_scripts.sh`, rescaled to this pipeline's epoch budget):

| Option | Default | Meaning |
|---|---|---|
| `--warmup-steps` | 500 | Source-only warm-up iterations (`warmup_iter`) |
| `--bmm-update-every` | 2 | Epochs between posterior refits (`update_term`) |
| `--pseudo-threshold` | 0.85 | Confidence needed for a target pseudo-label |
| `--ls-eps` | 0.1 | Label smoothing |
| `--head-lr-mult` | 10 | Learning-rate multiplier for C, E and the discriminator |
| `--adv-weight` / `--pseudo-weight` / `--unknown-weight` | 0.5 / 0.5 / 0.2 | Loss weights from the paper |
| `--selection-metric` | `known_macro_f1` | `known_macro_f1` matches DANN; `hos` uses target_val HOS; `last` uses no target labels |

Outputs, next to the usual DANN-style folders:

- `before_uadal_*` and `after_uadal_*`: known-only metrics computed on the first K logits.
- `openworld_target_{val,test}_{native,tuned}/`: open-world metrics and predictions, including `os_star`, `unk`, `hos`, `unknown_auroc` and `oscr`.
- `uadal_metrics_summary.csv` and `uadal_summary.json`.
- `history.csv`: per-epoch losses and `pseudo_mask_rate`.
- `posterior_history.csv`: beta-mixture fit after every refit.

Check `posterior_history.csv` after the first run. `w_unk_mean` should stay close to the true unknown rate (about 2 to 4% here). The `diag_*` columns use target labels **for logging only**. If `w_unk_mean` is much larger, the beta mixture is labelling hard known images as unknown, and known-class F1 will drop. If a fit degenerates, `bmm_valid` is `False` and that round falls back to plain known-class alignment.

UADAL runs three forward passes per step (source, weak target and strong target). Lower `--batch-size` if you run out of GPU memory.

## Notes

Update paths in `config.py` or pass CLI arguments. The scripts save metrics, predictions, plots, and checkpoints under the selected output directory.
