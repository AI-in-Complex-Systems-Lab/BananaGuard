import pytest

from user_store import UserStore, hash_password, verify_password


@pytest.fixture
def store(tmp_path):
    return UserStore(tmp_path / "users")


def test_hash_password_roundtrip():
    salt, password_hash = hash_password("correct-horse-battery")

    assert verify_password(
        "correct-horse-battery", salt, password_hash
    )

    assert not verify_password(
        "wrong-password", salt, password_hash
    )


def test_create_and_authenticate(store):
    created = store.create(
        username="Officer.Jones",
        password="LongEnoughPassword1",
        display_name="Officer Jones",
        role="officer",
    )

    assert created["username"] == "officer.jones"
    assert "password_hash" not in created

    authenticated = store.authenticate(
        "officer.jones", "LongEnoughPassword1"
    )

    assert authenticated is not None
    assert authenticated["role"] == "officer"

    assert (
        store.authenticate(
            "officer.jones", "wrong-password"
        )
        is None
    )


def test_authenticate_unknown_user(store):
    assert store.authenticate("nobody", "irrelevant") is None


def test_create_rejects_short_password(store):
    with pytest.raises(ValueError):
        store.create(
            username="shortpw",
            password="short",
            display_name="",
            role="officer",
        )


def test_create_rejects_bad_username(store):
    with pytest.raises(ValueError):
        store.create(
            username="in valid!",
            password="LongEnoughPassword1",
            display_name="",
            role="officer",
        )


def test_create_rejects_bad_role(store):
    with pytest.raises(ValueError):
        store.create(
            username="someone",
            password="LongEnoughPassword1",
            display_name="",
            role="superuser",
        )


def test_create_rejects_duplicate_username(store):
    store.create(
        username="dupe",
        password="LongEnoughPassword1",
        display_name="",
        role="officer",
    )

    with pytest.raises(ValueError):
        store.create(
            username="dupe",
            password="AnotherLongPassword1",
            display_name="",
            role="officer",
        )


def test_delete_last_admin_is_blocked(store):
    store.create(
        username="soleadmin",
        password="LongEnoughPassword1",
        display_name="",
        role="admin",
    )

    with pytest.raises(ValueError):
        store.delete("soleadmin")


def test_delete_non_last_admin_succeeds(store):
    store.create(
        username="admin-one",
        password="LongEnoughPassword1",
        display_name="",
        role="admin",
    )

    store.create(
        username="admin-two",
        password="LongEnoughPassword1",
        display_name="",
        role="admin",
    )

    assert store.delete("admin-one") is True
    assert store.get("admin-one") is None


def test_delete_unknown_user_returns_false(store):
    assert store.delete("ghost") is False


def test_bootstrap_admin_if_empty_only_once(store):
    result = store.bootstrap_admin_if_empty()

    assert result is not None
    username, password = result
    assert username == "admin"
    assert len(password) > 0

    assert store.bootstrap_admin_if_empty() is None


def test_set_password_updates_authentication(store):
    store.create(
        username="rotator",
        password="OriginalPassword1",
        display_name="",
        role="officer",
    )

    store.set_password("rotator", "BrandNewPassword1")

    assert (
        store.authenticate("rotator", "OriginalPassword1")
        is None
    )

    assert (
        store.authenticate("rotator", "BrandNewPassword1")
        is not None
    )
