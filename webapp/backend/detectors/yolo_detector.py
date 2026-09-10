import threading

from ultralytics import YOLO

from .base import Detector


class YoloDetector(Detector):
    """
    Wraps the platform's existing YOLO model. This is a structural
    move of the code that used to live directly in server.py (model
    loading, the inference thread lock, and box-to-wire-format
    conversion) — not a behavior change.
    """

    name = "yolo"
    detector_type = "yolo"
    supports_text_prompts = False
    requires_gpu = False

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(str(model_path))
        self._lock = threading.Lock()

    @property
    def names(self):
        return self.model.names

    def detect(self, image, **kwargs):
        confidence_threshold = kwargs.get(
            "confidence_threshold"
        )

        with self._lock:
            results = self.model(
                image,
                verbose=False,
                conf=confidence_threshold,
            )

        return self._serialize(results[0])

    def _serialize(self, result):
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = result.names[class_id]

            detections.append(
                {
                    "label": label,
                    "score": round(confidence, 4),
                    "box": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),
                        round(y2 - y1, 2),
                    ],
                }
            )

        return detections
