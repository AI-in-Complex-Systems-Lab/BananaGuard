# Model weights

## weapon_detection.pt (current)

- **Architecture:** YOLO11x-seg (Ultralytics), 100 epochs, trained on
  2x NVIDIA RTX A6000
- **Classes:** 4 distinct classes — `handgun`, `rifle`, `shotgun`, `knife`
  (the single-merged-class bug from the previous version is fixed)
- **Trained on:** 140GB / 36 videos of real police training footage
  (firearms range drills, force-on-force scenarios, room clearing),
  auto-labeled with SAM3 (`facebook/sam3` via Hugging Face Transformers)
  into ~41,000 usable frames, split by video (26 train / 5 val / 5 test)
  so no frame from a test video ever appears in training.
- **Test-set metrics** (held-out videos, never seen during training or
  tuning):

  | Class | Precision | Recall | mAP50 | mAP50-95 |
  |---|---|---|---|---|
  | handgun | 0.691 | 0.746 | 0.711 | 0.556 |
  | rifle | 0.567 | 0.474 | 0.487 | 0.372 |
  | shotgun | 1.000 | 0.000 | 0.000 | 0.000 |
  | knife | — | — | — | — (0 test instances) |
  | overall (box) | 0.753 | 0.407 | 0.399 | 0.309 |

### Known limitation

The source footage is almost entirely handgun and rifle drills — only
17 shotgun and 3 knife instances exist across the *entire* dataset.
Shotgun and knife detection are effectively non-functional as a
result (not a pipeline bug — there simply isn't enough footage of
those weapon types yet). Handgun and rifle detection are solid and
usable. Improving shotgun/knife requires collecting more footage
containing those weapons, then re-running the same pipeline.

The model is loaded via segmentation, but BananaGuard's detection
pipeline only reads bounding boxes (`result.boxes`) from it — mask
data is present in the model's output but unused.

## weapon_detection_seg_x_stable_v0_single_class.pt (superseded)

The original version of this model trained on the same footage but
with a `data.yaml` bug that collapsed all 4 weapon classes into one
merged `weapon` label, so it could detect a weapon but not identify
its type. Superseded by the current model above. Not kept in the repo
— see git history prior to the multi-class retraining if needed.
