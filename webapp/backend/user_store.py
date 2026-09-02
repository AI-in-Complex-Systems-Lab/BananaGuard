import hashlib
import json
import re
import secrets
import threading
import time
from pathlib import Path


PBKDF2_ITERATIONS = 200_000

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]{3,32}$")

VALID_ROLES = {"admin", "officer"}


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PBKDF2_ITERATIONS,
    )

    return salt, derived.hex()


def verify_password(password, salt, expected_hash):
    _, derived_hash = hash_password(password, salt)
    return secrets.compare_digest(derived_hash, expected_hash)


class UserStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "users.json"
        self.lock = threading.Lock()

    def _read(self):
        if not self.path.exists():
            return {}

        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, users):
        temporary = self.path.with_suffix(".tmp")

        temporary.write_text(
            json.dumps(users, indent=2),
            encoding="utf-8",
        )

        temporary.replace(self.path)

    def is_empty(self):
        with self.lock:
            return len(self._read()) == 0

    def get(self, username):
        with self.lock:
            return self._read().get(username.lower())

    def list(self):
        with self.lock:
            users = self._read()

        return [
            public_user(user)
            for user in users.values()
        ]

    def create(
        self,
        username,
        password,
        display_name,
        role,
    ):
        normalized_username = username.lower()

        if not USERNAME_PATTERN.match(normalized_username):
            raise ValueError(
                "Username must be 3-32 characters and "
                "contain only letters, numbers, dots, "
                "underscores, or hyphens"
            )

        if role not in VALID_ROLES:
            raise ValueError(
                f"Role must be one of {sorted(VALID_ROLES)}"
            )

        if len(password) < 10:
            raise ValueError(
                "Password must be at least 10 characters"
            )

        salt, password_hash = hash_password(password)

        with self.lock:
            users = self._read()

            if normalized_username in users:
                raise ValueError(
                    "A user with this username already exists"
                )

            users[normalized_username] = {
                "username": normalized_username,
                "display_name": display_name or username,
                "role": role,
                "password_salt": salt,
                "password_hash": password_hash,
                "created_at": time.time(),
            }

            self._write(users)

            return public_user(
                users[normalized_username]
            )

    def set_password(self, username, password):
        if len(password) < 10:
            raise ValueError(
                "Password must be at least 10 characters"
            )

        normalized_username = username.lower()
        salt, password_hash = hash_password(password)

        with self.lock:
            users = self._read()

            if normalized_username not in users:
                return False

            users[normalized_username][
                "password_salt"
            ] = salt

            users[normalized_username][
                "password_hash"
            ] = password_hash

            self._write(users)

            return True

    def delete(self, username):
        normalized_username = username.lower()

        with self.lock:
            users = self._read()

            if normalized_username not in users:
                return False

            admin_count = sum(
                1
                for user in users.values()
                if user["role"] == "admin"
            )

            is_last_admin = (
                users[normalized_username]["role"]
                == "admin"
                and admin_count <= 1
            )

            if is_last_admin:
                raise ValueError(
                    "Cannot delete the last remaining "
                    "admin account"
                )

            del users[normalized_username]
            self._write(users)

            return True

    def authenticate(self, username, password):
        user = self.get(username)

        if user is None:
            return None

        if not verify_password(
            password,
            user["password_salt"],
            user["password_hash"],
        ):
            return None

        return public_user(user)

    def bootstrap_admin_if_empty(self):
        with self.lock:
            users = self._read()

            if users:
                return None

            username = "admin"
            password = secrets.token_urlsafe(12)
            salt, password_hash = hash_password(password)

            users[username] = {
                "username": username,
                "display_name": "Administrator",
                "role": "admin",
                "password_salt": salt,
                "password_hash": password_hash,
                "created_at": time.time(),
            }

            self._write(users)

            return username, password


def public_user(user):
    return {
        "username": user["username"],
        "display_name": user["display_name"],
        "role": user["role"],
        "created_at": user.get("created_at"),
    }
