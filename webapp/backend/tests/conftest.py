import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def app():
    with tempfile.TemporaryDirectory() as storage_dir:
        os.environ["STORAGE_DIR"] = storage_dir
        os.environ["AUTH_SECRET_KEY"] = (
            "test-only-secret-do-not-use-in-production"
        )
        os.environ.setdefault(
            "CONFIDENCE_THRESHOLD", "0.5"
        )

        import server

        yield server.app


@pytest.fixture(scope="session")
def server_module(app):
    import server

    return server


@pytest.fixture(scope="session")
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def admin_credentials(client):
    response = client.get("/api/auth/bootstrap-hint")
    data = response.json()

    assert data["available"] is True

    return {
        "username": data["username"],
        "password": data["password"],
    }


@pytest.fixture(scope="session")
def admin_token(client, admin_credentials):
    response = client.post(
        "/api/auth/login", json=admin_credentials
    )

    assert response.status_code == 200

    return response.json()["access_token"]


@pytest.fixture(scope="session")
def admin_auth_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="session")
def officer_token(client, admin_auth_headers):
    create_response = client.post(
        "/api/auth/users",
        headers=admin_auth_headers,
        json={
            "username": "test-officer",
            "password": "OfficerPassword123",
            "display_name": "Test Officer",
            "role": "officer",
        },
    )

    assert create_response.status_code == 201

    login_response = client.post(
        "/api/auth/login",
        json={
            "username": "test-officer",
            "password": "OfficerPassword123",
        },
    )

    assert login_response.status_code == 200

    return login_response.json()["access_token"]


@pytest.fixture(scope="session")
def officer_auth_headers(officer_token):
    return {"Authorization": f"Bearer {officer_token}"}


@pytest.fixture
def tiny_video_bytes(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video_path = tmp_path / "tiny.mp4"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )

    for index in range(5):
        frame = np.full(
            (48, 64, 3), index * 10, dtype=np.uint8
        )
        writer.write(frame)

    writer.release()

    return video_path.read_bytes()
