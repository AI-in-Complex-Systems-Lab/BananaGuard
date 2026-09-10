"""
Standalone tests for GroundingDinoDetector. Skips cleanly wherever its
optional research dependencies (torch/transformers) or a CUDA GPU
aren't available — i.e. this file is a no-op in production's test run
and in the normal local backend venv, and only actually exercises the
detector on a machine with both installed (the lab GPU server).

Point this at real BananaGuard test images via GROUNDING_DINO_TEST_IMAGES_DIR
(defaults to "test_images", matching the layout used for the Stage 2/2B
research scripts).
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA not available - GroundingDinoDetector tests require a GPU",
        allow_module_level=True,
    )

from detectors import DetectorUnavailableError, create_detector
from detectors.grounding_dino_detector import GroundingDinoDetector, NORMALIZE_CATEGORY

TEST_IMAGES_DIR = os.environ.get("GROUNDING_DINO_TEST_IMAGES_DIR", "test_images")
TEST_DEVICE = os.environ.get("GROUNDING_DINO_TEST_DEVICE", "cuda:1")


def _load_image(name):
    import cv2

    path = os.path.join(TEST_IMAGES_DIR, name)
    image = cv2.imread(path)
    assert image is not None, f"could not load real test image: {path}"
    return image


@pytest.fixture(scope="module")
def detector():
    return create_detector("grounding_dino", device=TEST_DEVICE)


def test_metadata(detector):
    assert detector.detector_type == "grounding_dino"
    assert detector.name == "grounding_dino"
    assert detector.supports_text_prompts is True
    assert detector.requires_gpu is True
    # Default prompt strategy per Stage 2B, NOT the 7-prompt research set.
    assert detector.prompts == ["handgun.", "rifle.", "shotgun.", "knife."]


def test_model_actually_on_cuda(detector):
    device = str(next(detector.model.parameters()).device)
    assert device.startswith("cuda"), f"expected model on CUDA, got {device}"


def test_handgun_image_finds_both_real_handguns(detector):
    image = _load_image("01_handgun_range.jpg")
    detections = detector.detect(image, confidence_threshold=0.30)

    handgun_hits = [d for d in detections if d["label"] == "FIREARM_HANDGUN"]
    assert len(handgun_hits) == 2, (
        f"expected 2 FIREARM_HANDGUN detections (matches Stage 2B), got: {detections}"
    )


def test_shotgun_image_finds_one_shotgun_not_four_duplicates(detector):
    image = _load_image("02_shotgun_colored_lighting.jpg")
    detections = detector.detect(image, confidence_threshold=0.30)

    shotgun_hits = [d for d in detections if d["label"] == "FIREARM_SHOTGUN"]
    assert len(shotgun_hits) == 1, (
        "expected exactly 1 shotgun after NMS merging (all 4 default "
        f"prompts hit the same object in Stage 2B), got: {detections}"
    )


def test_common_detector_contract(detector):
    image = _load_image("02_shotgun_colored_lighting.jpg")
    detections = detector.detect(image, confidence_threshold=0.30)

    assert len(detections) > 0
    for detection in detections:
        # The contract every existing consumer (batch video, WebSocket)
        # relies on — must never be broken.
        assert isinstance(detection["label"], str)
        assert isinstance(detection["score"], float)
        assert isinstance(detection["box"], list)
        assert len(detection["box"]) == 4
        assert all(isinstance(v, float) for v in detection["box"])

        # Additive research metadata, optional for any consumer.
        assert "raw_prompt" in detection
        assert "normalized_category" in detection
        assert detection["normalized_category"] in set(NORMALIZE_CATEGORY.values())


def test_latency_is_recorded_and_sane(detector):
    import time

    image = _load_image("01_handgun_range.jpg")

    start = time.perf_counter()
    detector.detect(image, confidence_threshold=0.30)
    elapsed = time.perf_counter() - start

    print(f"\nGroundingDinoDetector.detect() latency (4 prompts): {elapsed:.3f}s")
    assert elapsed < 10, "sanity bound only, not a strict performance assertion"


def test_prompts_are_configurable_not_hardcoded():
    custom = create_detector(
        "grounding_dino",
        device=TEST_DEVICE,
        prompts=["handgun.", "pistol.", "revolver.", "rifle.", "shotgun.", "knife.", "machete."],
    )
    assert len(custom.prompts) == 7


def test_cuda_required_unless_allow_cpu(monkeypatch):
    # Force the "no CUDA available" branch even on a GPU-equipped
    # machine, to test the production-safety guard itself.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(DetectorUnavailableError):
        GroundingDinoDetector(device="cuda:1")
