import io
import zipfile

from dataset_export import build_yolo_export


def fake_jpeg_bytes(marker):
    return f"fake-jpeg-{marker}".encode("utf-8")


def test_returns_none_when_nothing_qualifies():
    records = [
        {
            "frame": 0,
            "status": "pending",
            "label": "gun",
            "box": [1, 2, 3, 4],
        },
        {
            "frame": 1,
            "status": "rejected",
            "label": "gun",
            "box": [1, 2, 3, 4],
        },
    ]

    zip_bytes, frames, detections = build_yolo_export(
        records, 100, 100, {}
    )

    assert zip_bytes is None
    assert frames == 0
    assert detections == 0


def test_basic_export_contents():
    records = [
        {
            "frame": 0,
            "status": "approved",
            "label": "gun",
            "box": [10, 20, 40, 60],
        },
        {
            "frame": 0,
            "status": "pending",
            "label": "gun",
            "box": [1, 1, 1, 1],
        },
        {
            "frame": 5,
            "status": "corrected",
            "label": "rifle",
            "box": [0, 0, 50, 50],
        },
    ]

    frame_images = {
        0: fake_jpeg_bytes(0),
        5: fake_jpeg_bytes(5),
    }

    zip_bytes, frames, detections = build_yolo_export(
        records, 200, 100, frame_images
    )

    assert zip_bytes is not None
    assert frames == 2
    # the pending record on frame 0 must not count
    assert detections == 2

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = set(archive.namelist())

        assert "images/frame_0.jpg" in names
        assert "labels/frame_0.txt" in names
        assert "images/frame_5.jpg" in names
        assert "labels/frame_5.txt" in names
        assert "classes.txt" in names
        assert "data.yaml" in names

        assert (
            archive.read("images/frame_0.jpg")
            == fake_jpeg_bytes(0)
        )

        classes_content = archive.read(
            "classes.txt"
        ).decode("utf-8")
        assert classes_content == "gun\nrifle\n"

        # frame 0: box [10,20,40,60] on a 200x100 image
        # x_center=(10+20)/200=0.15, y_center=(20+30)/100=0.50
        # width=40/200=0.20, height=60/100=0.60, class "gun" -> index 0
        label_0 = archive.read(
            "labels/frame_0.txt"
        ).decode("utf-8").strip()
        assert label_0 == "0 0.150000 0.500000 0.200000 0.600000"

        # frame 5: box [0,0,50,50] on a 200x100 image, class "rifle" -> index 1
        label_5 = archive.read(
            "labels/frame_5.txt"
        ).decode("utf-8").strip()
        assert label_5 == "1 0.125000 0.250000 0.250000 0.500000"


def test_frames_without_extracted_images_are_skipped():
    records = [
        {
            "frame": 0,
            "status": "approved",
            "label": "gun",
            "box": [10, 20, 40, 60],
        },
        {
            "frame": 1,
            "status": "approved",
            "label": "gun",
            "box": [10, 20, 40, 60],
        },
    ]

    # only frame 0's image was successfully extracted
    frame_images = {0: fake_jpeg_bytes(0)}

    zip_bytes, frames, detections = build_yolo_export(
        records, 100, 100, frame_images
    )

    assert zip_bytes is not None
    assert frames == 1
    assert detections == 1

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = set(archive.namelist())
        assert "images/frame_1.jpg" not in names
        assert "labels/frame_1.txt" not in names


def test_multiple_detections_on_same_frame():
    records = [
        {
            "frame": 3,
            "status": "approved",
            "label": "gun",
            "box": [0, 0, 10, 10],
        },
        {
            "frame": 3,
            "status": "corrected",
            "label": "gun",
            "box": [50, 50, 10, 10],
        },
    ]

    frame_images = {3: fake_jpeg_bytes(3)}

    zip_bytes, frames, detections = build_yolo_export(
        records, 100, 100, frame_images
    )

    assert frames == 1
    assert detections == 2

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        label_lines = (
            archive.read("labels/frame_3.txt")
            .decode("utf-8")
            .strip()
            .splitlines()
        )
        assert len(label_lines) == 2


def test_all_frames_missing_images_returns_none():
    records = [
        {
            "frame": 0,
            "status": "approved",
            "label": "gun",
            "box": [10, 20, 40, 60],
        },
    ]

    zip_bytes, frames, detections = build_yolo_export(
        records, 100, 100, {}
    )

    assert zip_bytes is None
    assert frames == 0
    assert detections == 0
