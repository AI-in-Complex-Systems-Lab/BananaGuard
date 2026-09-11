import time


def upload_and_wait(client, headers, video_bytes, filename="clip.mp4"):
    response = client.post(
        "/api/videos",
        headers=headers,
        files={"file": (filename, video_bytes, "video/mp4")},
    )

    assert response.status_code == 202
    job_id = response.json()["job_id"]

    for _ in range(60):
        status_response = client.get(
            f"/api/jobs/{job_id}", headers=headers
        )

        assert status_response.status_code == 200
        job = status_response.json()

        if job["status"] in {"completed", "failed"}:
            assert job["status"] == "completed"
            return job

        time.sleep(0.5)

    raise AssertionError("job never finished")


def test_compare_same_source_video_no_warning(
    client, admin_auth_headers, tiny_video_bytes
):
    job_a = upload_and_wait(
        client, admin_auth_headers, tiny_video_bytes, "same.mp4"
    )
    job_b = upload_and_wait(
        client, admin_auth_headers, tiny_video_bytes, "same.mp4"
    )

    response = client.get(
        "/api/research/compare",
        params={"job_a": job_a["job_id"], "job_b": job_b["job_id"]},
        headers=admin_auth_headers,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["warning"] is None
    assert data["ground_truth_evaluation"] == {
        "available": False,
        "message": "Ground-truth evaluation not available yet.",
        "metrics": None,
    }

    for side in ("job_a", "job_b"):
        metadata = data["jobs"][side]["metadata"]
        summary = data["jobs"][side]["summary"]

        assert metadata["detector_type"] == "yolo"
        assert metadata["supports_text_prompts"] is False
        assert metadata["prompts"] is None
        assert metadata["source_fps"] is not None
        assert metadata["source_duration_seconds"] is not None
        assert metadata["analyzed_frames"] == 5

        assert summary["total_detections"] == 0
        assert summary["average_confidence"] is None
        assert summary["detections_per_analyzed_frame"] == 0

    assert data["timeline"]["window_seconds"] == 2.0
    assert all(
        not b["job_a"] and not b["job_b"] and not b["both"]
        for b in data["timeline"]["bins"]
    )


def test_compare_different_source_videos_warns(
    client, admin_auth_headers, tiny_video_bytes
):
    job_a = upload_and_wait(
        client, admin_auth_headers, tiny_video_bytes, "video_one.mp4"
    )
    job_b = upload_and_wait(
        client, admin_auth_headers, tiny_video_bytes, "video_two.mp4"
    )

    response = client.get(
        "/api/research/compare",
        params={"job_a": job_a["job_id"], "job_b": job_b["job_id"]},
        headers=admin_auth_headers,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["warning"] is not None
    assert "not scientifically fair" in data["warning"]


def test_compare_unknown_job_returns_404(
    client, admin_auth_headers, tiny_video_bytes
):
    job_a = upload_and_wait(client, admin_auth_headers, tiny_video_bytes)

    response = client.get(
        "/api/research/compare",
        params={"job_a": job_a["job_id"], "job_b": "does-not-exist"},
        headers=admin_auth_headers,
    )

    assert response.status_code == 404


def test_compare_requires_auth(client):
    response = client.get(
        "/api/research/compare",
        params={"job_a": "a", "job_b": "b"},
    )

    assert response.status_code == 401


def test_compare_surfaces_known_research_notes_by_filename(
    client, admin_auth_headers, tiny_video_bytes
):
    job_a = upload_and_wait(
        client,
        admin_auth_headers,
        tiny_video_bytes,
        "force_on_force.mp4",
    )
    job_b = upload_and_wait(
        client,
        admin_auth_headers,
        tiny_video_bytes,
        "unrelated.mp4",
    )

    response = client.get(
        "/api/research/compare",
        params={"job_a": job_a["job_id"], "job_b": job_b["job_id"]},
        headers=admin_auth_headers,
    )

    assert response.status_code == 200
    notes = response.json()["qualitative_notes"]

    # job_a matches "force_on_force.mp4" twice: the yolo-specific note
    # and the detector-agnostic pair-level note. job_b ("unrelated.mp4")
    # matches nothing.
    assert len(notes) == 2
    assert all(note["job_id"] == job_a["job_id"] for note in notes)
    assert any("handgun" in note["note"] for note in notes)
