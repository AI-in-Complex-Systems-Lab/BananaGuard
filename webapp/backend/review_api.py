from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import get_current_user
from review_store import ReviewStore


review_router = APIRouter()

reviews_directory = (
    Path(__file__).resolve().parent
    / "storage"
    / "reviews"
)

review_store = ReviewStore(reviews_directory)


class DetectionReviewRequest(BaseModel):
    status: Literal[
        "approved",
        "rejected",
        "corrected",
    ]

    label: str | None = None

    box: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
    )

    notes: str | None = Field(
        default=None,
        max_length=1000,
    )


@review_router.get(
    "/api/jobs/{job_id}/reviews"
)
async def list_reviews(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    detections = review_store.list(job_id)

    if detections is None:
        raise HTTPException(
            status_code=404,
            detail="Review data was not found",
        )

    return {
        "job_id": job_id,
        "summary": review_store.summary(job_id),
        "detections": detections,
    }


@review_router.patch(
    "/api/jobs/{job_id}/reviews/{detection_id}"
)
async def update_review(
    job_id: str,
    detection_id: str,
    request: DetectionReviewRequest,
    current_user: dict = Depends(get_current_user),
):
    if (
        request.status == "corrected"
        and request.label is None
        and request.box is None
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "A corrected detection requires "
                "a new label or bounding box"
            ),
        )

    detection = review_store.update(
        job_id=job_id,
        detection_id=detection_id,
        status=request.status,
        reviewed_by=current_user["username"],
        label=request.label,
        box=request.box,
        notes=request.notes,
    )

    if detection is None:
        raise HTTPException(
            status_code=404,
            detail="Detection was not found",
        )

    return {
        "job_id": job_id,
        "summary": review_store.summary(job_id),
        "detection": detection,
    }