import time

import jwt
import pytest
from fastapi import HTTPException

from auth import AuthService, TOKEN_ALGORITHM
from user_store import UserStore


@pytest.fixture
def user_store(tmp_path):
    return UserStore(tmp_path / "users")


@pytest.fixture
def auth_service(tmp_path, user_store):
    return AuthService(
        secret_key="unit-test-secret",
        user_store=user_store,
        bootstrap_credentials_path=(
            tmp_path / "admin_bootstrap.txt"
        ),
    )


@pytest.fixture
def officer(user_store):
    return user_store.create(
        username="officer1",
        password="LongEnoughPassword1",
        display_name="Officer One",
        role="officer",
    )


def test_create_and_decode_token(auth_service, officer):
    token = auth_service.create_token(officer)
    payload = auth_service.decode_token(token)

    assert payload["sub"] == "officer1"
    assert payload["role"] == "officer"


def test_user_from_token_returns_current_role(
    auth_service, officer
):
    token = auth_service.create_token(officer)
    resolved = auth_service.user_from_token(token)

    assert resolved["username"] == "officer1"
    assert resolved["role"] == "officer"


def test_decode_token_rejects_garbage(auth_service):
    with pytest.raises(HTTPException) as excinfo:
        auth_service.decode_token("not-a-real-token")

    assert excinfo.value.status_code == 401


def test_decode_token_rejects_expired(
    auth_service, officer
):
    expired_payload = {
        "sub": officer["username"],
        "role": officer["role"],
        "display_name": officer["display_name"],
        "iat": int(time.time()) - 100,
        "exp": int(time.time()) - 50,
    }

    expired_token = jwt.encode(
        expired_payload,
        auth_service.secret_key,
        algorithm=TOKEN_ALGORITHM,
    )

    with pytest.raises(HTTPException) as excinfo:
        auth_service.decode_token(expired_token)

    assert excinfo.value.status_code == 401


def test_decode_token_rejects_wrong_secret(
    auth_service, officer
):
    token = jwt.encode(
        {
            "sub": officer["username"],
            "role": officer["role"],
            "display_name": officer["display_name"],
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        },
        "a-different-secret",
        algorithm=TOKEN_ALGORITHM,
    )

    with pytest.raises(HTTPException):
        auth_service.decode_token(token)


def test_user_from_token_rejects_deleted_user(
    auth_service, user_store, officer
):
    token = auth_service.create_token(officer)
    user_store.delete(officer["username"])

    with pytest.raises(HTTPException) as excinfo:
        auth_service.user_from_token(token)

    assert excinfo.value.status_code == 401


def test_bootstrap_credentials_roundtrip(
    auth_service, tmp_path
):
    assert auth_service.read_bootstrap_credentials() is None

    (tmp_path / "admin_bootstrap.txt").write_text(
        "username: admin\npassword: s3cret-value\n",
        encoding="utf-8",
    )

    credentials = auth_service.read_bootstrap_credentials()

    assert credentials == {
        "username": "admin",
        "password": "s3cret-value",
    }


def test_clear_bootstrap_credentials_only_if_matching(
    auth_service, tmp_path
):
    path = tmp_path / "admin_bootstrap.txt"
    path.write_text(
        "username: admin\npassword: s3cret-value\n",
        encoding="utf-8",
    )

    auth_service.clear_bootstrap_credentials_if_matching(
        "someone-else"
    )
    assert path.exists()

    auth_service.clear_bootstrap_credentials_if_matching(
        "admin"
    )
    assert not path.exists()
