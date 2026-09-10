"""
Standalone Grounding DINO smoke test for BananaGuard's Stage 2 research
spike (see research/vlm-integration branch).

Deliberately outside the FastAPI app: this script has no dependency on
server.py, YoloDetector, or anything in webapp/. Its only job is to
answer "does open-vocabulary detection work at all on real BananaGuard
frames, and how does prompt wording affect it" before any integration
work happens.

Usage:
    python grounding_dino_test.py \
        --image /path/to/frame.jpg \
        --prompt "handgun. pistol. firearm. rifle. shotgun. knife." \
        --box-threshold 0.30 \
        --text-threshold 0.25 \
        --repeat 3 \
        --output-dir results/grounding_dino_stage2 \
        --tag handgun_range

Outputs (under --output-dir):
    annotated/<tag>.jpg   - image with predicted boxes/labels drawn
    json/<tag>.json       - full machine-readable result record
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to a real BananaGuard frame/image")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt, GroundingDINO-style period-separated phrases, e.g. 'handgun. rifle.'",
    )
    parser.add_argument("--box-threshold", type=float, default=0.30, help="Detection confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Text-matching threshold")
    parser.add_argument("--repeat", type=int, default=3, help="Number of inference repeats after warm-up")
    parser.add_argument("--output-dir", required=True, help="Base output directory")
    parser.add_argument("--tag", required=True, help="Short identifier for this run, used in output filenames")
    parser.add_argument(
        "--checkpoint",
        default="IDEA-Research/grounding-dino-tiny",
        help="Hugging Face checkpoint to load",
    )
    parser.add_argument("--device", default="cuda:1", help="Torch device to run on")
    return parser.parse_args()


def gpu_memory_mib(device):
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.memory_allocated(device) / 1024**2, 1)


def load_model(checkpoint, device):
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

    processor = GroundingDinoProcessor.from_pretrained(checkpoint)
    model = GroundingDinoForObjectDetection.from_pretrained(checkpoint).to(device)
    model.eval()
    return processor, model


def run_once(processor, model, image, prompt, box_threshold, text_threshold, device):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model(**inputs)

    torch.cuda.synchronize(device)
    latency_seconds = time.perf_counter() - start

    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        x1, y1, x2, y2 = [round(v, 2) for v in box.tolist()]
        detections.append(
            {
                "label": label,
                "score": round(float(score), 4),
                "box_xyxy": [x1, y1, x2, y2],
            }
        )

    return detections, latency_seconds


def draw_annotations(image, detections):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["box_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline="#00ff66", width=4)
        caption = f"{detection['label']} ({detection['score']:.2f})"
        draw.text((x1, max(y1 - 22, 0)), caption, fill="#00ff66", font=font)

    return annotated


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    annotated_dir = output_dir / "annotated"
    json_dir = output_dir / "json"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"WARNING: CUDA not available, falling back to {device}")

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")

    mem_before_load = gpu_memory_mib(device)
    load_start = time.perf_counter()
    processor, model = load_model(args.checkpoint, device)
    load_time = time.perf_counter() - load_start
    mem_after_load = gpu_memory_mib(device)

    print(f"[{args.tag}] model loaded in {load_time:.2f}s onto {device}")
    print(f"[{args.tag}] GPU memory before load: {mem_before_load} MiB, after load: {mem_after_load} MiB")

    # Warm-up run (first CUDA kernel launches / cuDNN autotune are slow and
    # would distort the "real" latency if counted with the rest).
    warmup_detections, warmup_latency = run_once(
        processor, model, image, args.prompt, args.box_threshold, args.text_threshold, device
    )
    print(f"[{args.tag}] warm-up run latency: {warmup_latency:.3f}s ({len(warmup_detections)} detections)")

    subsequent_runs = []
    for i in range(args.repeat):
        detections, latency = run_once(
            processor, model, image, args.prompt, args.box_threshold, args.text_threshold, device
        )
        subsequent_runs.append({"latency_seconds": round(latency, 4), "detections": detections})
        print(f"[{args.tag}] run {i + 1}/{args.repeat}: {latency:.3f}s, {len(detections)} detections")

    mem_after_inference = gpu_memory_mib(device)

    final_detections = subsequent_runs[-1]["detections"] if subsequent_runs else warmup_detections
    annotated = draw_annotations(image, final_detections)
    annotated_path = annotated_dir / f"{args.tag}.jpg"
    annotated.save(annotated_path, quality=95)

    result_record = {
        "tag": args.tag,
        "image_path": str(image_path),
        "image_size_wh": list(image.size),
        "checkpoint": args.checkpoint,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "prompt": args.prompt,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "model_load_time_seconds": round(load_time, 3),
        "gpu_memory_mib_before_load": mem_before_load,
        "gpu_memory_mib_after_load": mem_after_load,
        "gpu_memory_mib_after_inference": mem_after_inference,
        "warmup_run": {
            "latency_seconds": round(warmup_latency, 4),
            "detections": warmup_detections,
        },
        "subsequent_runs": subsequent_runs,
        "annotated_image_path": str(annotated_path),
    }

    json_path = json_dir / f"{args.tag}.json"
    with open(json_path, "w") as f:
        json.dump(result_record, f, indent=2)

    print(f"[{args.tag}] annotated image saved to {annotated_path}")
    print(f"[{args.tag}] JSON result saved to {json_path}")


if __name__ == "__main__":
    main()
