"""
BananaGuard Stage 2B — per-class prompt strategy experiment.

Standalone, outside the FastAPI app (no dependency on server.py or
YoloDetector). Extends Stage 2 (research/grounding_dino_test.py) to
answer one question: should BananaGuard run Grounding DINO with one
combined multi-class prompt, or as several independent single-class
prompts merged after inference?

Compares three strategies on the same real images at the same
thresholds (box=0.30, text=0.25, matching Stage 2's baseline):
  A. one combined multi-class prompt
  B. seven independent single-class prompts, normalized + NMS-merged
  C. the best single known prompt for that image (from Stage 2/B)

Also measures the 4-prompt subset (handgun/rifle/shotgun/knife) against
the full 7-prompt set (adding pistol/revolver/machete synonyms), and
whether the HF Grounding DINO API allows reusing image features across
prompts (it does not, at the API level — see REUSE_FINDING below).
"""

import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

CHECKPOINT = "IDEA-Research/grounding-dino-tiny"
DEVICE = "cuda:1"
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
IOU_MERGE_THRESHOLD = 0.5

PROMPTS_7 = ["handgun.", "pistol.", "revolver.", "rifle.", "shotgun.", "knife.", "machete."]
PROMPTS_4 = ["handgun.", "rifle.", "shotgun.", "knife."]
COMBINED_PROMPT = "handgun. pistol. revolver. rifle. shotgun. knife. machete."

NORMALIZE = {
    "handgun": "FIREARM_HANDGUN",
    "pistol": "FIREARM_HANDGUN",
    "revolver": "FIREARM_HANDGUN",
    "rifle": "FIREARM_RIFLE",
    "shotgun": "FIREARM_SHOTGUN",
    "knife": "BLADED_WEAPON",
    "machete": "BLADED_WEAPON",
}

IMAGES = [
    {"tag": "01_handgun_range", "path": "test_images/01_handgun_range.jpg", "best_known_prompt": "handgun."},
    {"tag": "02_shotgun_colored_lighting", "path": "test_images/02_shotgun_colored_lighting.jpg", "best_known_prompt": "shotgun."},
    {"tag": "03_difficult_motion_blur_crowd", "path": "test_images/03_difficult_motion_blur_crowd.jpg", "best_known_prompt": None},
]

REUSE_FINDING = (
    "Inspected transformers/models/grounding_dino/modeling_grounding_dino.py "
    "(GroundingDinoModel.forward): the vision backbone call "
    "`self.backbone(pixel_values, pixel_mask)` takes no text input and runs "
    "before any image/text fusion, so the image features ARE architecturally "
    "independent of the prompt. However, the public model.forward()/__call__ "
    "API recomputes the backbone from raw pixel_values on every single call "
    "and exposes no parameter to inject precomputed vision features back in. "
    "So: reuse is architecturally possible, but NOT available through normal "
    "use of the HF API — it would require subclassing/monkey-patching the "
    "model to call the backbone once and feed cached features into the rest "
    "of the forward pass manually. Not attempted here (would be optimization, "
    "not measurement)."
)


def load_model():
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

    processor = GroundingDinoProcessor.from_pretrained(CHECKPOINT)
    model = GroundingDinoForObjectDetection.from_pretrained(CHECKPOINT).to(DEVICE)
    model.eval()
    return processor, model


def gpu_mem_mib():
    return round(torch.cuda.memory_allocated(DEVICE) / 1024**2, 1)


def run_one_prompt(processor, model, image, prompt):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)

    torch.cuda.synchronize(DEVICE)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    torch.cuda.synchronize(DEVICE)
    latency = time.perf_counter() - start

    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        x1, y1, x2, y2 = [round(v, 2) for v in box.tolist()]
        detections.append(
            {
                "raw_prompt": prompt,
                "raw_label": label,
                "score": round(float(score), 4),
                "box_xyxy": [x1, y1, x2, y2],
            }
        )

    return detections, latency


def run_combined(processor, model, image, prompt):
    return run_one_prompt(processor, model, image, prompt)


def run_separate(processor, model, image, prompts):
    all_detections = []
    per_prompt_latency = {}

    start_total = time.perf_counter()
    for prompt in prompts:
        detections, latency = run_one_prompt(processor, model, image, prompt)
        per_prompt_latency[prompt] = round(latency, 4)
        all_detections.extend(detections)
    total_latency = time.perf_counter() - start_total

    return all_detections, per_prompt_latency, total_latency


def normalize_detection(detection):
    key = detection["raw_prompt"].rstrip(".").strip().lower()
    detection["normalized_category"] = NORMALIZE.get(key, "UNKNOWN")
    return detection


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def merge_duplicates(detections, iou_threshold=IOU_MERGE_THRESHOLD):
    """
    Greedy NMS-style merge across ALL detections from all prompts for one
    image: sort by score descending, keep a box if it doesn't overlap
    (IoU > threshold) an already-kept box, and record which other
    raw_prompt/category hits were folded into it as duplicates of the same
    physical object.
    """
    duplicate_count_before = len(detections)

    ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    suppressed_log = []

    for candidate in ordered:
        merged_into = None
        for winner in kept:
            if iou(candidate["box_xyxy"], winner["box_xyxy"]) > iou_threshold:
                merged_into = winner
                break

        if merged_into is None:
            candidate["other_matching_prompts"] = []
            kept.append(candidate)
        else:
            merged_into["other_matching_prompts"].append(
                {
                    "raw_prompt": candidate["raw_prompt"],
                    "normalized_category": candidate["normalized_category"],
                    "score": candidate["score"],
                }
            )
            suppressed_log.append(candidate)

    return kept, duplicate_count_before, len(kept), suppressed_log


def draw_annotations(image, detections, label_field="normalized_category"):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["box_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline="#00ff66", width=4)
        caption = f"{detection[label_field]} ({detection['score']:.2f})"
        draw.text((x1, max(y1 - 22, 0)), caption, fill="#00ff66", font=font)

    return annotated


def main():
    output_dir = Path("results/grounding_dino_stage2b")
    annotated_dir = output_dir / "annotated"
    json_dir = output_dir / "json"
    logs_dir = output_dir / "logs"
    for d in (annotated_dir, json_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    processor, model = load_model()

    comparison_rows = []
    per_image_records = {}

    for image_info in IMAGES:
        tag = image_info["tag"]
        image = Image.open(image_info["path"]).convert("RGB")
        print(f"\n=== {tag} ===")

        # --- Strategy A: combined multi-class prompt ---
        combined_detections, combined_latency = run_combined(processor, model, image, COMBINED_PROMPT)
        for d in combined_detections:
            d["normalized_category"] = "UNKNOWN"  # combined prompt has no single raw_prompt to map cleanly
        print(f"[A: combined] {combined_latency:.3f}s, {len(combined_detections)} raw detections")

        # --- Strategy B: 7 separate prompts, normalized + merged ---
        raw_7, per_prompt_latency_7, total_latency_7 = run_separate(processor, model, image, PROMPTS_7)
        raw_7 = [normalize_detection(d) for d in raw_7]
        merged_7, dup_before_7, dup_after_7, suppressed_7 = merge_duplicates(raw_7)
        print(
            f"[B: 7-prompt] total {total_latency_7:.3f}s, "
            f"{dup_before_7} raw -> {dup_after_7} after NMS"
        )

        # --- 4-prompt subset (handgun/rifle/shotgun/knife only) ---
        raw_4, per_prompt_latency_4, total_latency_4 = run_separate(processor, model, image, PROMPTS_4)
        raw_4 = [normalize_detection(d) for d in raw_4]
        merged_4, dup_before_4, dup_after_4, suppressed_4 = merge_duplicates(raw_4)
        print(
            f"[4-prompt subset] total {total_latency_4:.3f}s, "
            f"{dup_before_4} raw -> {dup_after_4} after NMS"
        )

        # --- Strategy C: best single known prompt ---
        best_prompt = image_info["best_known_prompt"]
        if best_prompt is None:
            # Fall back to whichever of the 4/7 single prompts scored highest
            # on THIS image, if any fired at all.
            candidates = raw_7
            if candidates:
                best_hit = max(candidates, key=lambda d: d["score"])
                best_prompt = best_hit["raw_prompt"]
            else:
                best_prompt = None

        if best_prompt is not None:
            strategy_c_detections, strategy_c_latency = run_one_prompt(processor, model, image, best_prompt)
            strategy_c_detections = [normalize_detection(d) for d in strategy_c_detections]
        else:
            strategy_c_detections, strategy_c_latency = [], None
        print(f"[C: best single = {best_prompt!r}] {len(strategy_c_detections)} detections")

        # Annotated images
        draw_annotations(image, combined_detections).save(annotated_dir / f"{tag}_A_combined.jpg", quality=95)
        draw_annotations(image, merged_7).save(annotated_dir / f"{tag}_B_7prompt_merged.jpg", quality=95)
        draw_annotations(image, merged_4).save(annotated_dir / f"{tag}_4prompt_merged.jpg", quality=95)
        if strategy_c_detections:
            draw_annotations(image, strategy_c_detections).save(annotated_dir / f"{tag}_C_best_single.jpg", quality=95)

        record = {
            "tag": tag,
            "image_path": image_info["path"],
            "strategy_A_combined": {
                "prompt": COMBINED_PROMPT,
                "latency_seconds": round(combined_latency, 4),
                "detections": combined_detections,
            },
            "strategy_B_7prompt": {
                "prompts": PROMPTS_7,
                "per_prompt_latency_seconds": per_prompt_latency_7,
                "total_latency_seconds": round(total_latency_7, 4),
                "raw_detection_count_before_nms": dup_before_7,
                "detection_count_after_nms": dup_after_7,
                "merged_detections": merged_7,
                "suppressed_as_duplicates": suppressed_7,
            },
            "strategy_4prompt_subset": {
                "prompts": PROMPTS_4,
                "per_prompt_latency_seconds": per_prompt_latency_4,
                "total_latency_seconds": round(total_latency_4, 4),
                "raw_detection_count_before_nms": dup_before_4,
                "detection_count_after_nms": dup_after_4,
                "merged_detections": merged_4,
                "suppressed_as_duplicates": suppressed_4,
            },
            "strategy_C_best_single": {
                "prompt": best_prompt,
                "latency_seconds": round(strategy_c_latency, 4) if strategy_c_latency else None,
                "detections": strategy_c_detections,
            },
        }
        per_image_records[tag] = record

        with open(json_dir / f"{tag}.json", "w") as f:
            json.dump(record, f, indent=2)

        comparison_rows.append(
            {
                "tag": tag,
                "strategy": "A_combined",
                "prompt_or_prompts": COMBINED_PROMPT,
                "detections_final": len(combined_detections),
                "raw_before_nms": len(combined_detections),
                "latency_seconds": round(combined_latency, 4),
            }
        )
        comparison_rows.append(
            {
                "tag": tag,
                "strategy": "B_7prompt_merged",
                "prompt_or_prompts": " | ".join(PROMPTS_7),
                "detections_final": dup_after_7,
                "raw_before_nms": dup_before_7,
                "latency_seconds": round(total_latency_7, 4),
            }
        )
        comparison_rows.append(
            {
                "tag": tag,
                "strategy": "4prompt_subset_merged",
                "prompt_or_prompts": " | ".join(PROMPTS_4),
                "detections_final": dup_after_4,
                "raw_before_nms": dup_before_4,
                "latency_seconds": round(total_latency_4, 4),
            }
        )
        comparison_rows.append(
            {
                "tag": tag,
                "strategy": "C_best_single",
                "prompt_or_prompts": best_prompt or "(none found)",
                "detections_final": len(strategy_c_detections),
                "raw_before_nms": len(strategy_c_detections),
                "latency_seconds": round(strategy_c_latency, 4) if strategy_c_latency else None,
            }
        )

    # --- Latency / throughput summary ---
    single_prompt_latencies = []
    for record in per_image_records.values():
        single_prompt_latencies.extend(record["strategy_B_7prompt"]["per_prompt_latency_seconds"].values())
    avg_single = sum(single_prompt_latencies) / len(single_prompt_latencies)

    avg_7prompt_total = sum(r["strategy_B_7prompt"]["total_latency_seconds"] for r in per_image_records.values()) / len(per_image_records)
    avg_4prompt_total = sum(r["strategy_4prompt_subset"]["total_latency_seconds"] for r in per_image_records.values()) / len(per_image_records)
    avg_combined = sum(r["strategy_A_combined"]["latency_seconds"] for r in per_image_records.values()) / len(per_image_records)

    latency_summary = {
        "avg_single_prompt_latency_seconds": round(avg_single, 4),
        "avg_combined_prompt_latency_seconds": round(avg_combined, 4),
        "avg_7prompt_sequential_total_seconds": round(avg_7prompt_total, 4),
        "avg_4prompt_sequential_total_seconds": round(avg_4prompt_total, 4),
        "theoretical_fps_combined": round(1 / avg_combined, 2),
        "theoretical_fps_7prompt": round(1 / avg_7prompt_total, 2),
        "theoretical_fps_4prompt": round(1 / avg_4prompt_total, 2),
        "gpu_memory_mib_final": gpu_mem_mib(),
        "feature_reuse_finding": REUSE_FINDING,
    }

    with open(json_dir / "_latency_summary.json", "w") as f:
        json.dump(latency_summary, f, indent=2)

    with open(logs_dir / "latency_summary.txt", "w") as f:
        f.write(json.dumps(latency_summary, indent=2))

    import csv
    with open(output_dir / "comparison_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("\n=== LATENCY SUMMARY ===")
    print(json.dumps(latency_summary, indent=2))
    print(f"\nWrote comparison_summary.csv with {len(comparison_rows)} rows")


if __name__ == "__main__":
    main()
