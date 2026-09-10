from abc import ABC, abstractmethod


class Detector(ABC):
    """
    Common interface every object detector on this platform implements,
    so the inference call sites (batch video processing and the
    live-camera WebSocket) can call any detector interchangeably
    without knowing which model backs it.

    A detection is a dict matching the platform's existing wire
    format, unchanged by this abstraction:
        {"label": str, "score": float, "box": [x, y, width, height]}

    Metadata attributes below are informational only (e.g. for a
    future /health or model-selection UI) and carry no behavior.
    """

    name: str = "unknown"
    detector_type: str = "unknown"
    supports_text_prompts: bool = False
    requires_gpu: bool = False

    @abstractmethod
    def detect(self, image, **kwargs):
        """
        Run detection on a single BGR image (as read by cv2) and
        return a list of detections in the format described above.
        """
        raise NotImplementedError
