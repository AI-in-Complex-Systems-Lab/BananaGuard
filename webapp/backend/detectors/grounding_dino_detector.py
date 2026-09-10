import os
import threading

from .base import Detector, DetectorUnavailableError

DEFAULT_CHECKPOINT = "IDEA-Research/grounding-dino-tiny"
DEFAULT_DEVICE = "cuda"
DEFAULT_PROMPTS = ["handgun.", "rifle.", "shotgun.", "knife."]
DEFAULT_BOX_THRESHOLD = 0.30
DEFAULT_TEXT_THRESHOLD = 0.25
DEFAULT_IOU_MERGE_THRESHOLD = 0.5

# Research-level category normalization proven in Stage 2B: several
# prompts can refer to the same physical class of weapon, so their
# raw hits collapse into one shared category after inference.
NORMALIZE_CATEGORY = {
    "handgun": "FIREARM_HANDGUN",
    "pistol": "FIREARM_HANDGUN",
    "revolver": "FIREARM_HANDGUN",
    "rifle": "FIREARM_RIFLE",
    "shotgun": "FIREARM_SHOTGUN",
    "knife": "BLADED_WEAPON",
    "machete": "BLADED_WEAPON",
}


def _iou(box_a, box_b):
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


class GroundingDinoDetector(Detector):
    """
    Open-vocabulary weapon detector using Grounding DINO (Hugging Face
    transformers implementation), run as several independent
    single-class prompts and merged after inference — the strategy
    proven out in the Stage 2B research experiment
    (research/grounding_dino_stage2b.py). A combined multi-class
    prompt was deliberately rejected there: it suppressed detections
    that succeeded when run as separate prompts.

    Not registered as the default detector anywhere, and not wired to
    any live inference path yet — this class exists so the platform's
    detector abstraction is *capable* of using it, proven via a
    standalone test, before any integration decision is made.
    """

    name = "grounding_dino"
    detector_type = "grounding_dino"
    supports_text_prompts = True
    requires_gpu = True

    def __init__(
        self,
        checkpoint=None,
        device=None,
        prompts=None,
        box_threshold=None,
        text_threshold=None,
        iou_merge_threshold=None,
        allow_cpu=None,
    ):
        checkpoint = checkpoint or os.environ.get("GROUNDING_DINO_CHECKPOINT", DEFAULT_CHECKPOINT)
        device = device or os.environ.get("GROUNDING_DINO_DEVICE", DEFAULT_DEVICE)

        if prompts is None:
            prompts_env = os.environ.get("GROUNDING_DINO_PROMPTS")
            prompts = (
                [p.strip() for p in prompts_env.split(",") if p.strip()]
                if prompts_env
                else list(DEFAULT_PROMPTS)
            )

        self.prompts = prompts
        self.box_threshold = (
            box_threshold
            if box_threshold is not None
            else float(os.environ.get("GROUNDING_DINO_BOX_THRESHOLD", DEFAULT_BOX_THRESHOLD))
        )
        self.text_threshold = (
            text_threshold
            if text_threshold is not None
            else float(os.environ.get("GROUNDING_DINO_TEXT_THRESHOLD", DEFAULT_TEXT_THRESHOLD))
        )
        self.iou_merge_threshold = (
            iou_merge_threshold
            if iou_merge_threshold is not None
            else float(os.environ.get("GROUNDING_DINO_IOU_MERGE_THRESHOLD", DEFAULT_IOU_MERGE_THRESHOLD))
        )

        if allow_cpu is None:
            allow_cpu = os.environ.get("GROUNDING_DINO_ALLOW_CPU", "").lower() in {"1", "true", "yes"}

        try:
            import torch
            from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
        except ImportError as error:
            raise DetectorUnavailableError(
                "GroundingDinoDetector requires the 'torch' and 'transformers' "
                "packages, which are not installed in this environment. "
                "Install backend/requirements-research.txt to enable it."
            ) from error

        if device.startswith("cuda") and not torch.cuda.is_available():
            if not allow_cpu:
                raise DetectorUnavailableError(
                    f"GroundingDinoDetector requested device {device!r} but CUDA "
                    "is not available on this machine. Set "
                    "GROUNDING_DINO_ALLOW_CPU=true to force (very slow) CPU "
                    "inference for research use, or run this on a GPU machine."
                )
            print(
                "WARNING: GroundingDinoDetector falling back to CPU — "
                "inference will be roughly an order of magnitude slower."
            )
            device = "cpu"

        self.checkpoint = checkpoint
        self.device = device
        self._lock = threading.Lock()
        self._torch = torch

        self.processor = GroundingDinoProcessor.from_pretrained(checkpoint)
        self.model = GroundingDinoForObjectDetection.from_pretrained(checkpoint).to(device)
        self.model.eval()

    @property
    def names(self):
        categories = sorted({NORMALIZE_CATEGORY.get(p.rstrip(".").strip().lower(), "UNKNOWN") for p in self.prompts})
        return dict(enumerate(categories))

    def detect(self, image, **kwargs):
        pil_image = self._to_pil(image)

        prompts = kwargs.get("prompts", self.prompts)
        box_threshold = kwargs.get("confidence_threshold", self.box_threshold)
        text_threshold = kwargs.get("text_threshold", self.text_threshold)

        raw_detections = []
        for prompt in prompts:
            raw_detections.extend(
                self._run_one_prompt(pil_image, prompt, box_threshold, text_threshold)
            )

        merged = self._merge_duplicates(raw_detections)

        return [self._to_common_format(detection) for detection in merged]

    @staticmethod
    def _to_pil(image):
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _run_one_prompt(self, pil_image, prompt, box_threshold, text_threshold):
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)

        with self._lock:
            with self._torch.no_grad():
                outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        key = prompt.rstrip(".").strip().lower()
        normalized_category = NORMALIZE_CATEGORY.get(key, "UNKNOWN")

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
            x1, y1, x2, y2 = [round(v, 2) for v in box.tolist()]
            detections.append(
                {
                    "raw_prompt": prompt,
                    "raw_label": label,
                    "normalized_category": normalized_category,
                    "score": round(float(score), 4),
                    "box_xyxy": [x1, y1, x2, y2],
                }
            )

        return detections

    def _merge_duplicates(self, detections):
        """
        Greedy NMS-style merge, proven in Stage 2B: sort by score
        descending, keep a box unless it overlaps (IoU over threshold)
        an already-kept box, and fold suppressed hits from other
        prompts into the winner's other_matching_prompts so a single
        physical weapon is never counted once per prompt that found it.
        """
        ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
        kept = []

        for candidate in ordered:
            winner = next(
                (k for k in kept if _iou(candidate["box_xyxy"], k["box_xyxy"]) > self.iou_merge_threshold),
                None,
            )

            if winner is None:
                candidate["other_matching_prompts"] = []
                kept.append(candidate)
            else:
                winner["other_matching_prompts"].append(
                    {
                        "raw_prompt": candidate["raw_prompt"],
                        "normalized_category": candidate["normalized_category"],
                        "score": candidate["score"],
                    }
                )

        return kept

    @staticmethod
    def _to_common_format(detection):
        x1, y1, x2, y2 = detection["box_xyxy"]

        return {
            # Common Detector contract — every consumer of any detector
            # (batch video processing, live-camera WebSocket) can rely
            # on exactly these three keys, unchanged.
            "label": detection["normalized_category"],
            "score": detection["score"],
            "box": [
                round(x1, 2),
                round(y1, 2),
                round(x2 - x1, 2),
                round(y2 - y1, 2),
            ],
            # Additive research metadata — safe to ignore for any
            # consumer that only reads label/score/box.
            "raw_prompt": detection["raw_prompt"],
            "normalized_category": detection["normalized_category"],
            "model_name": "grounding_dino",
            "other_matching_prompts": detection["other_matching_prompts"],
        }
