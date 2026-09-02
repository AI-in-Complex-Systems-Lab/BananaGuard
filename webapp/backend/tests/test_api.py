import time


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "confidence_threshold" in data


def test_protected_endpoint_requires_auth(client):
    response = client.get("/api/jobs")

    assert response.status_code == 401


def test_login_rejects_bad_password(client, admin_credentials):
    response = client.post(
        "/api/auth/login",
        json={
            "username": admin_credentials["username"],
            "password": "definitely-wrong",
        },
    )

    assert response.status_code == 401


def test_login_succeeds_with_correct_credentials(
    client, admin_credentials
):
    response = client.post(
        "/api/auth/login", json=admin_credentials
    )

    assert response.status_code == 200
    data = response.json()
    assert data["user"]["role"] == "admin"
    assert "password" not in data["user"]


def test_me_returns_current_user(
    client, admin_auth_headers
):
    response = client.get(
        "/api/auth/me", headers=admin_auth_headers
    )

    assert response.status_code == 200
    assert response.json()["role"] == "admin"


def test_officer_cannot_list_users(
    client, officer_auth_headers
):
    response = client.get(
        "/api/auth/users", headers=officer_auth_headers
    )

    assert response.status_code == 403


def test_admin_can_list_users(client, admin_auth_headers):
    response = client.get(
        "/api/auth/users", headers=admin_auth_headers
    )

    assert response.status_code == 200
    usernames = [u["username"] for u in response.json()]
    assert "admin" in usernames


def test_create_user_rejects_short_password(
    client, admin_auth_headers
):
    response = client.post(
        "/api/auth/users",
        headers=admin_auth_headers,
        json={
            "username": "shortpw-user",
            "password": "short",
            "role": "officer",
        },
    )

    assert response.status_code == 400


def test_create_user_rejects_invalid_role(
    client, admin_auth_headers
):
    response = client.post(
        "/api/auth/users",
        headers=admin_auth_headers,
        json={
            "username": "bad-role-user",
            "password": "LongEnoughPassword1",
            "role": "superuser",
        },
    )

    assert response.status_code == 400


def test_officer_cannot_create_user(
    client, officer_auth_headers
):
    response = client.post(
        "/api/auth/users",
        headers=officer_auth_headers,
        json={
            "username": "sneaky",
            "password": "LongEnoughPassword1",
            "role": "officer",
        },
    )

    assert response.status_code == 403


def test_admin_cannot_delete_self(
    client, admin_auth_headers, admin_credentials
):
    response = client.delete(
        f"/api/auth/users/{admin_credentials['username']}",
        headers=admin_auth_headers,
    )

    assert response.status_code == 400


def test_admin_can_delete_a_non_last_admin(
    client, admin_auth_headers
):
    # The last-admin guard only blocks a delete that would leave zero
    # admins; with the shared "admin" fixture account still present,
    # deleting a second admin here must succeed. The zero-admins case
    # is covered directly against UserStore in test_user_store.py,
    # since the API's own "can't delete yourself" rule makes that
    # scenario unreachable through the API (whoever calls the API is
    # necessarily one of the admins that would have to remain).
    create_response = client.post(
        "/api/auth/users",
        headers=admin_auth_headers,
        json={
            "username": "second-admin",
            "password": "LongEnoughPassword1",
            "role": "admin",
        },
    )

    assert create_response.status_code == 201

    delete_response = client.delete(
        "/api/auth/users/second-admin",
        headers=admin_auth_headers,
    )

    assert delete_response.status_code == 200


def test_upload_rejects_unsupported_extension(
    client, admin_auth_headers
):
    response = client.post(
        "/api/videos",
        headers=admin_auth_headers,
        files={
            "file": (
                "not-a-video.txt",
                b"hello",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_upload_requires_auth(client, tiny_video_bytes):
    response = client.post(
        "/api/videos",
        files={
            "file": (
                "clip.mp4",
                tiny_video_bytes,
                "video/mp4",
            )
        },
    )

    assert response.status_code == 401


def test_full_video_pipeline(
    client, admin_auth_headers, tiny_video_bytes
):
    upload_response = client.post(
        "/api/videos",
        headers=admin_auth_headers,
        files={
            "file": (
                "clip.mp4",
                tiny_video_bytes,
                "video/mp4",
            )
        },
    )

    assert upload_response.status_code == 202
    job_id = upload_response.json()["job_id"]

    final_job = None

    for _ in range(60):
        status_response = client.get(
            f"/api/jobs/{job_id}",
            headers=admin_auth_headers,
        )

        assert status_response.status_code == 200
        job = status_response.json()

        if job["status"] in {"completed", "failed"}:
            final_job = job
            break

        time.sleep(0.5)

    assert final_job is not None, "job never finished"
    assert final_job["status"] == "completed"
    assert final_job["processed_frames"] == 5

    reviews_response = client.get(
        f"/api/jobs/{job_id}/reviews",
        headers=admin_auth_headers,
    )

    assert reviews_response.status_code == 200
    assert reviews_response.json()["summary"]["total"] == 0

    frame_response = client.get(
        f"/api/jobs/{job_id}/frames/0",
        headers=admin_auth_headers,
    )

    assert frame_response.status_code == 200
    assert (
        frame_response.headers["content-type"]
        == "image/jpeg"
    )

    out_of_range_response = client.get(
        f"/api/jobs/{job_id}/frames/9999",
        headers=admin_auth_headers,
    )

    assert out_of_range_response.status_code == 404

    negative_frame_response = client.get(
        f"/api/jobs/{job_id}/frames/-1",
        headers=admin_auth_headers,
    )

    assert negative_frame_response.status_code == 400

    # media endpoints also accept the token as a query param,
    # since <img>/<video> tags cannot set an Authorization header
    admin_token = admin_auth_headers["Authorization"].split(
        " "
    )[1]

    query_auth_response = client.get(
        f"/api/jobs/{job_id}/frames/0",
        params={"token": admin_token},
    )

    assert query_auth_response.status_code == 200

    dashboard_response = client.get(
        "/api/dashboard", headers=admin_auth_headers
    )

    assert dashboard_response.status_code == 200
    assert dashboard_response.json()["totals"]["jobs"] >= 1


def test_job_not_found(client, admin_auth_headers):
    response = client.get(
        "/api/jobs/does-not-exist",
        headers=admin_auth_headers,
    )

    assert response.status_code == 404


def test_review_update_validation_and_tracking(
    client, admin_auth_headers, server_module
):
    job_id = "seeded-review-job"

    server_module.review_store.initialize(
        job_id,
        [
            {
                "frame": 0,
                "timestamp_seconds": 0.0,
                "detections": [
                    {
                        "label": "gun",
                        "score": 0.8,
                        "box": [1, 2, 3, 4],
                    }
                ],
            }
        ],
    )

    # corrected without a new label or box is rejected
    bad_correction = client.patch(
        f"/api/jobs/{job_id}/reviews/0-0",
        headers=admin_auth_headers,
        json={"status": "corrected"},
    )

    assert bad_correction.status_code == 400

    unknown_detection = client.patch(
        f"/api/jobs/{job_id}/reviews/999-0",
        headers=admin_auth_headers,
        json={"status": "approved"},
    )

    assert unknown_detection.status_code == 404

    approve_response = client.patch(
        f"/api/jobs/{job_id}/reviews/0-0",
        headers=admin_auth_headers,
        json={"status": "approved"},
    )

    assert approve_response.status_code == 200
    detection = approve_response.json()["detection"]
    assert detection["status"] == "approved"
    assert detection["reviewed_by"] == "admin"
    assert detection["reviewed_at"] is not None


def test_settings_get_and_patch(
    client, admin_auth_headers, officer_auth_headers
):
    get_response = client.get(
        "/api/settings", headers=admin_auth_headers
    )

    assert get_response.status_code == 200

    officer_patch = client.patch(
        "/api/settings",
        headers=officer_auth_headers,
        json={"confidence_threshold": 0.6},
    )

    assert officer_patch.status_code == 403

    out_of_range_patch = client.patch(
        "/api/settings",
        headers=admin_auth_headers,
        json={"confidence_threshold": 5.0},
    )

    assert out_of_range_patch.status_code == 400

    valid_patch = client.patch(
        "/api/settings",
        headers=admin_auth_headers,
        json={"confidence_threshold": 0.65},
    )

    assert valid_patch.status_code == 200
    assert valid_patch.json()["confidence_threshold"] == 0.65

    confirm_response = client.get(
        "/api/settings", headers=admin_auth_headers
    )

    assert (
        confirm_response.json()["confidence_threshold"]
        == 0.65
    )


def test_websocket_requires_token(client):
    try:
        with client.websocket_connect("/ws"):
            raise AssertionError(
                "connection should have been rejected"
            )
    except Exception:
        pass


def test_websocket_rejects_invalid_token(client):
    try:
        with client.websocket_connect(
            "/ws?token=not-a-real-token"
        ):
            raise AssertionError(
                "connection should have been rejected"
            )
    except Exception:
        pass
