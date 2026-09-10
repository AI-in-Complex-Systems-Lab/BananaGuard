from .yolo_detector import YoloDetector

AVAILABLE_DETECTOR_TYPES = ("yolo", "grounding_dino")


def create_detector(detector_type, **config):
    """
    Factory for the platform's detectors. Only ever constructs the
    detector actually requested.

    In particular, requesting "yolo" (the default everywhere, including
    production) never imports Grounding DINO's module — and therefore
    never imports transformers/torch's GroundingDINO code path — so
    production is completely unaffected by whether Grounding DINO's
    optional research dependencies are installed. That import only
    happens inside this function, and only on the "grounding_dino"
    branch below.
    """
    if detector_type == "yolo":
        model_path = config.get("model_path")
        if model_path is None:
            raise ValueError("YoloDetector requires a 'model_path'")
        return YoloDetector(model_path)

    if detector_type == "grounding_dino":
        from .grounding_dino_detector import GroundingDinoDetector

        return GroundingDinoDetector(
            checkpoint=config.get("checkpoint"),
            device=config.get("device"),
            prompts=config.get("prompts"),
        )

    raise ValueError(
        f"Unknown detector_type {detector_type!r}. "
        f"Available: {', '.join(AVAILABLE_DETECTOR_TYPES)}."
    )
