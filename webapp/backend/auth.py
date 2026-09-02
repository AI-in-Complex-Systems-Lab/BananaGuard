import secrets
import time
from pathlib import Path

import jwt
from fastapi import Depends, HTTPException, Query
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from user_store import UserStore


TOKEN_ALGORITHM = "HS256"
TOKEN_LIFETIME_SECONDS = 12 * 60 * 60

bearer_scheme = HTTPBearer(auto_error=False)


def load_or_create_secret_key(path):
    path = Path(path)

    if path.exists():
        return path.read_text(encoding="utf-8").strip()

    secret_key = secrets.token_hex(32)
    path.write_text(secret_key, encoding="utf-8")

    return secret_key


class AuthService:
    def __init__(self, secret_key, user_store: UserStore):
        self.secret_key = secret_key
        self.user_store = user_store

    def create_token(self, user):
        payload = {
            "sub": user["username"],
            "role": user["role"],
            "display_name": user["display_name"],
            "iat": int(time.time()),
            "exp": int(
                time.time() + TOKEN_LIFETIME_SECONDS
            ),
        }

        return jwt.encode(
            payload,
            self.secret_key,
            algorithm=TOKEN_ALGORITHM,
        )

    def decode_token(self, token):
        try:
            return jwt.decode(
                token,
                self.secret_key,
                algorithms=[TOKEN_ALGORITHM],
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Session expired, please sign in again",
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token",
            )

    def user_from_token(self, token):
        payload = self.decode_token(token)
        user = self.user_store.get(payload["sub"])

        if user is None:
            raise HTTPException(
                status_code=401,
                detail="User no longer exists",
            )

        return {
            "username": user["username"],
            "display_name": user["display_name"],
            "role": user["role"],
        }


auth_service: AuthService | None = None


def configure_auth_service(service):
    global auth_service
    auth_service = service


def get_auth_service():
    return auth_service


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_scheme
    ),
):
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    return auth_service.user_from_token(
        credentials.credentials
    )


def get_current_user_flexible(
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_scheme
    ),
    token: str | None = Query(default=None),
):
    supplied_token = (
        credentials.credentials
        if credentials is not None
        else token
    )

    if supplied_token is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    return auth_service.user_from_token(supplied_token)


def require_admin(
    current_user: dict = Depends(get_current_user),
):
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Administrator access required",
        )

    return current_user
