from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import get_auth_service, get_current_user, require_admin
from user_store import VALID_ROLES


auth_router = APIRouter(prefix="/api/auth")


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    display_name: str = Field(default="")
    role: str = "officer"


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@auth_router.get("/bootstrap-hint")
async def bootstrap_hint():
    credentials = get_auth_service().read_bootstrap_credentials()

    if credentials is None:
        return {"available": False}

    return {
        "available": True,
        "username": credentials["username"],
        "password": credentials["password"],
    }


@auth_router.post("/login")
async def login(request: LoginRequest):
    auth_service = get_auth_service()

    user = auth_service.user_store.authenticate(
        request.username,
        request.password,
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    token = auth_service.create_token(user)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user,
    }


@auth_router.get("/me")
async def me(
    current_user: dict = Depends(get_current_user),
):
    return current_user


@auth_router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
):
    auth_service = get_auth_service()
    authenticated = auth_service.user_store.authenticate(
        current_user["username"],
        request.current_password,
    )

    if authenticated is None:
        raise HTTPException(
            status_code=401,
            detail="Current password is incorrect",
        )

    try:
        auth_service.user_store.set_password(
            current_user["username"],
            request.new_password,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    auth_service.clear_bootstrap_credentials_if_matching(
        current_user["username"]
    )

    return {"status": "ok"}


@auth_router.get("/users")
async def list_users(
    current_user: dict = Depends(require_admin),
):
    return get_auth_service().user_store.list()


@auth_router.post("/users", status_code=201)
async def create_user(
    request: CreateUserRequest,
    current_user: dict = Depends(require_admin),
):
    if request.role not in VALID_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Role must be one of {sorted(VALID_ROLES)}",
        )

    try:
        return get_auth_service().user_store.create(
            username=request.username,
            password=request.password,
            display_name=request.display_name,
            role=request.role,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@auth_router.delete("/users/{username}")
async def delete_user(
    username: str,
    current_user: dict = Depends(require_admin),
):
    if username.lower() == current_user["username"]:
        raise HTTPException(
            status_code=400,
            detail="You cannot delete your own account",
        )

    try:
        deleted = get_auth_service().user_store.delete(
            username
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    return {"status": "ok"}
