# Model weights

## weapon_detection.pt

- **Architecture:** YOLO11x-seg (Ultralytics), 100 epochs
- **Classes:** single merged `weapon` class (handgun/rifle/shotgun/knife
  training data was collapsed into one class — see note below)
- **Validation metrics (best epoch, epoch 79):**
  - Box: precision 0.777, recall 0.683, mAP50 0.744, mAP50-95 0.547
  - Mask: precision 0.728, recall 0.623, mAP50 0.657, mAP50-95 0.375
- **Trained on:** real police training footage (firearms range drills,
  force-on-force scenarios, room clearing) via a SAM3 auto-labeling
  pipeline (`vertex_sam3_video_to_yolo.ipynb`, run on 8x H100 GPUs)

### Known limitation

The auto-labeling pipeline generated 4 distinct classes (handgun,
rifle, shotgun, knife), but the `data.yaml` used for this specific
training run declared only 1 class (`weapon`), collapsing all four
into one. The notebook that produced this dataset has since been
fixed for this bug — a future training run should be able to recover
the 4-class distinction. This model does not distinguish weapon type.

The model is loaded via segmentation, but BananaGuard's detection
pipeline only reads bounding boxes (`result.boxes`) from it — mask
data is present in the model's output but unused.
