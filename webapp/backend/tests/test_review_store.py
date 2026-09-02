import pytest

from review_store import ReviewStore


@pytest.fixture
def store(tmp_path):
    return ReviewStore(tmp_path / "reviews")


@pytest.fixture
def detection_events():
    return [
        {
            "frame": 0,
            "timestamp_seconds": 0.0,
            "detections": [
                {
                    "label": "gun",
                    "score": 0.9,
                    "box": [1, 2, 3, 4],
                }
            ],
        },
        {
            "frame": 10,
            "timestamp_seconds": 0.33,
            "detections": [
                {
                    "label": "gun",
                    "score": 0.7,
                    "box": [5, 6, 7, 8],
                },
                {
                    "label": "gun",
                    "score": 0.6,
                    "box": [9, 10, 11, 12],
                },
            ],
        },
    ]


def test_list_unknown_job_returns_none(store):
    assert store.list("unknown-job") is None
    assert store.summary("unknown-job") is None


def test_initialize_creates_pending_records(
    store, detection_events
):
    records = store.initialize("job-1", detection_events)

    assert len(records) == 3
    assert all(r["status"] == "pending" for r in records)
    assert records[0]["detection_id"] == "0-0"
    assert records[1]["detection_id"] == "10-0"
    assert records[2]["detection_id"] == "10-1"

    summary = store.summary("job-1")
    assert summary == {
        "total": 3,
        "pending": 3,
        "approved": 0,
        "rejected": 0,
        "corrected": 0,
    }


def test_update_approve_records_reviewer(
    store, detection_events
):
    store.initialize("job-2", detection_events)

    updated = store.update(
        job_id="job-2",
        detection_id="0-0",
        status="approved",
        reviewed_by="jdoe",
    )

    assert updated["status"] == "approved"
    assert updated["reviewed_by"] == "jdoe"
    assert updated["reviewed_at"] is not None

    summary = store.summary("job-2")
    assert summary["approved"] == 1
    assert summary["pending"] == 2


def test_update_corrected_changes_label_and_box(
    store, detection_events
):
    store.initialize("job-3", detection_events)

    updated = store.update(
        job_id="job-3",
        detection_id="10-1",
        status="corrected",
        reviewed_by="admin",
        label="rifle",
        box=[1, 1, 2, 2],
    )

    assert updated["label"] == "rifle"
    assert updated["box"] == [1, 1, 2, 2]
    assert updated["original_label"] == "gun"


def test_update_unknown_detection_returns_none(
    store, detection_events
):
    store.initialize("job-4", detection_events)

    result = store.update(
        job_id="job-4",
        detection_id="999-0",
        status="approved",
        reviewed_by="jdoe",
    )

    assert result is None


def test_update_unknown_job_returns_none(store):
    result = store.update(
        job_id="no-such-job",
        detection_id="0-0",
        status="approved",
        reviewed_by="jdoe",
    )

    assert result is None
