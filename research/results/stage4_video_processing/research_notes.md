# Stage 4 Research Notes — Grounding DINO uploaded-video processing

Real end-to-end test through the actual `/api/videos` pipeline (not a standalone
script), running on the lab GPU server with the full backend booted
(`DETECTOR_TYPE` defaults to `yolo`; `grounding_dino` selected per-job via the
new `detector_type` upload form field).

No ground-truth annotations exist for either video, so nothing here is a
precision/recall/accuracy claim — only directly observed agreements,
disagreements, and errors, each anchored to a specific frame that was visually
checked.

## Video 1 — `short_gun_clip.mp4` (real footage, 90 frames / 3.0s / 1920x1080)

Both YOLO (every frame) and Grounding DINO (multiple sampling rates, including
every frame) found **zero detections**. Visual inspection of several frames
confirms this is a genuinely hard/ambiguous case: an officer filmed from
behind during a room-clearing drill, weapon (if any) not clearly visible to
the human eye either. Both detectors agreeing on "nothing here" is itself a
useful, honest data point — neither hallucinated a detection on an ambiguous
negative.

## Video 2 — `force_on_force.mp4` (real footage, 4417 frames / 147.4s / 1920x1080)

- **YOLO** (every frame, current production multi-class model): 393 detections
  across 390 frames, **100% labeled "handgun"** — no rifle/shotgun/knife
  detections at all, despite the model having those classes.
- **Grounding DINO** (4 default prompts, sample_fps=2 → stride 15, 295 sampled
  frames): 44 detections — 12 handgun, 25 shotgun, 7 rifle.

### Where they agree

Both detectors independently concentrate the large majority of their
detections in the same ~110–146s window of the video (YOLO: 391/393
detections there; Grounding DINO: ~34/44). Two completely different detection
approaches agreeing on *when* the video's real weapon activity happens is a
meaningful cross-check, even without ground truth.

### Where they disagree

1. **Grounding DINO found activity YOLO missed**: a cluster of "shotgun"
   detections at t≈64–78s, a period where YOLO reports almost nothing
   (1 detection at 60–65s, nothing 65–110s). Checked frame 1935 (t=64.56s,
   GDINO said "shotgun" at 0.51) visually — **it's an empty room, no person,
   no weapon at all.** This is a confirmed Grounding DINO false positive, not
   a real detection YOLO missed. See
   `comparison_frames/01_gdino_false_positive_empty_room_shotgun_0.51.jpg`.

2. **Category disagreement inside the shared hot zone**: YOLO calls
   everything in this window "handgun." Grounding DINO calls the same period
   a mix of handgun/rifle/shotgun. Checked frame 4110 (t=137.14s, GDINO said
   "shotgun" at 0.65) visually — **it's clearly a handgun, held two-handed.**
   Confirmed mislabel. See
   `comparison_frames/02_gdino_mislabeled_handgun_as_shotgun_0.65.jpg`.
   Checked frame 4380 (t=146.15s, GDINO said "handgun" at 0.50) — **correctly
   a handgun**, single-handed grip. See
   `comparison_frames/03_gdino_correct_handgun_0.50.jpg`.

   So on the same physical object type, a few sampled frames apart, Grounding
   DINO's per-frame classification is inconsistent (shotgun, then correctly
   handgun). YOLO was consistent (always handgun) across the entire video —
   though that consistency could itself reflect the training-data imbalance
   documented in `webapp/backend/models/README.md` (22,727 handgun vs. 16,964
   rifle vs. 17 shotgun vs. 3 knife instances) rather than genuinely correct
   discrimination; this video alone can't tell those two explanations apart.

### Honest summary

Neither model should be called "more accurate" from this alone. What's
observed: YOLO was fast, consistent, and stayed silent outside the real
action window on this video, but showed no rifle/shotgun/knife
discrimination at all here. Grounding DINO caught the same overall
timeframe, occasionally identified rifle/shotgun where YOLO didn't, but did
so inconsistently and produced at least one clear false positive on an empty
room. Both behaviors are consistent with the two systems' known weaknesses
going into this test (YOLO: severe shotgun/knife training-data scarcity;
Grounding DINO: prompt-dependent, sometimes-unstable zero-shot
classification, as already seen in Stage 2B).
