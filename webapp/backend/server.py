import asyncio
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from ultralytics import YOLO

import auth as auth_module
from auth import (
    AuthService,
    configure_auth_service,
    get_current_user,
    get_current_user_flexible,
    load_or_create_secret_key,
)
from auth_api import auth_router
from job_store import JobStore
from review_api import review_router, review_store
from user_store import UserStore


app = FastAPI(title="BananaGuard API")

server_directory = Path(__file__).resolve().parent
default_model_path = (
    server_directory.parent.parent / "yolo11.pt"
)
model_path = Path(
    os.environ.get("MODEL_PATH", default_model_path)
)

storage_directory = server_directory / "storage"
uploads_directory = storage_directory / "uploads"
outputs_directory = storage_directory / "outputs"
jobs_directory = storage_directory / "jobs"
users_directory = storage_directory / "users"

uploads_directory.mkdir(parents=True, exist_ok=True)
outputs_directory.mkdir(parents=True, exist_ok=True)
jobs_directory.mkdir(parents=True, exist_ok=True)
users_directory.mkdir(parents=True, exist_ok=True)

confidence_threshold = float(
    os.environ.get("CONFIDENCE_THRESHOLD", "0.50")
)

maximum_upload_size = 2 * 1024 * 1024 * 1024

allowed_video_extensions = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".webm",
}

model = YOLO(str(model_path))

model_lock = threading.Lock()
jobs_lock = threading.Lock()

job_store = JobStore(jobs_directory)
jobs = job_store.load_all()

for _job in jobs.values():
    if _job.get("status") in {"queued", "processing"}:
        _job["status"] = "failed"
        _job["message"] = "Processing was interrupted by a server restart"
        _job["error"] = "Processing was interrupted by a server restart"
        job_store.write(_job["job_id"], _job)


user_store = UserStore(users_directory)

secret_key = os.environ.get(
    "AUTH_SECRET_KEY"
) or load_or_create_secret_key(
    storage_directory / "auth_secret.key"
)

bootstrap_path = (
    storage_directory / "admin_bootstrap.txt"
)

configure_auth_service(
    AuthService(secret_key, user_store, bootstrap_path)
)

bootstrap_result = user_store.bootstrap_admin_if_empty()

if bootstrap_result is not None:
    bootstrap_username, bootstrap_password = (
        bootstrap_result
    )

    bootstrap_path.write_text(
        f"username: {bootstrap_username}\n"
        f"password: {bootstrap_password}\n",
        encoding="utf-8",
    )

    print(
        "=" * 60
        + "\nCreated default administrator account:\n"
        f"  username: {bootstrap_username}\n"
        f"  password: {bootstrap_password}\n"
        f"Also saved to {bootstrap_path}\n"
        "Sign in and change this password immediately.\n"
        + "=" * 60
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(review_router)


def update_job(job_id, **values):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(values)
            job_snapshot = dict(jobs[job_id])
        else:
            job_snapshot = None

    if job_snapshot is not None:
        job_store.write(job_id, job_snapshot)


def get_job_copy(job_id):
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            return None

        return dict(job)


def get_public_job(job):
    return {
        key: value
        for key, value in job.items()
        if key not in {"input_path", "output_path"}
    }


def run_inference(image):
    with model_lock:
        results = model(
            image,
            verbose=False,
            conf=confidence_threshold,
        )

    return results[0]


def serialize_detections(result):
    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = result.names[class_id]

        detections.append(
            {
                "label": label,
                "score": round(confidence, 4),
                "box": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2 - x1, 2),
                    round(y2 - y1, 2),
                ],
            }
        )

    return detections


def draw_detections(frame, detections):
    for detection in detections:
        x, y, width, height = detection["box"]

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + width)
        y2 = int(y + height)

        label = detection["label"]
        score = detection["score"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return frame


def process_video(job_id, input_path, output_path):
    video_capture = None
    video_writer = None

    try:
        update_job(
            job_id,
            status="processing",
            progress=0,
            message="Opening video",
        )

        video_capture = cv2.VideoCapture(
            str(input_path)
        )

        if not video_capture.isOpened():
            raise RuntimeError(
                "Unable to open the uploaded video"
            )

        total_frames = int(
            video_capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        frames_per_second = video_capture.get(
            cv2.CAP_PROP_FPS
        )

        if (
            not frames_per_second
            or frames_per_second <= 0
        ):
            frames_per_second = 30.0

        frame_width = int(
            video_capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        frame_height = int(
            video_capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        if frame_width <= 0 or frame_height <= 0:
            raise RuntimeError(
                "The video has invalid dimensions"
            )

        codec = cv2.VideoWriter_fourcc(*"mp4v")

        video_writer = cv2.VideoWriter(
            str(output_path),
            codec,
            frames_per_second,
            (frame_width, frame_height),
        )

        if not video_writer.isOpened():
            raise RuntimeError(
                "Unable to create the annotated "
                "output video"
            )

        frame_index = 0
        frames_with_detections = 0
        total_detections = 0
        detection_events = []
        previous_progress = -1

        while True:
            success, frame = video_capture.read()

            if not success or frame is None:
                break

            result = run_inference(frame)
            detections = serialize_detections(
                result
            )

            if detections:
                timestamp_seconds = (
                    frame_index / frames_per_second
                )

                frames_with_detections += 1
                total_detections += len(
                    detections
                )

                detection_events.append(
                    {
                        "frame": frame_index,
                        "timestamp_seconds": round(
                            timestamp_seconds,
                            3,
                        ),
                        "detections": detections,
                    }
                )

            annotated_frame = draw_detections(
                frame,
                detections,
            )

            video_writer.write(annotated_frame)
            frame_index += 1

            if total_frames > 0:
                progress = min(
                    int(
                        (
                            frame_index
                            / total_frames
                        )
                        * 100
                    ),
                    99,
                )
            else:
                progress = 0

            if progress != previous_progress:
                update_job(
                    job_id,
                    progress=progress,
                    processed_frames=frame_index,
                    message=(
                        f"Processing frame "
                        f"{frame_index}"
                        + (
                            f" of {total_frames}"
                            if total_frames > 0
                            else ""
                        )
                    ),
                )

                previous_progress = progress

        if frame_index == 0:
            raise RuntimeError(
                "The uploaded video did not "
                "contain readable frames"
            )

        review_store.initialize(
            job_id,
            detection_events,
        )

        review_summary = review_store.summary(
            job_id
        )

        update_job(
            job_id,
            status="completed",
            progress=100,
            message="Processing complete",
            processed_frames=frame_index,
            total_frames=(
                total_frames or frame_index
            ),
            frames_with_detections=(
                frames_with_detections
            ),
            total_detections=total_detections,
            detection_events=detection_events,
            review_summary=review_summary,
            reviews_url=(
                f"/api/jobs/{job_id}/reviews"
            ),
            result_url=(
                f"/api/jobs/{job_id}/result"
            ),
        )

    except Exception as error:
        update_job(
            job_id,
            status="failed",
            message=str(error),
            error=str(error),
        )

    finally:
        if video_capture is not None:
            video_capture.release()

        if video_writer is not None:
            video_writer.release()


async def process_video_in_background(
    job_id,
    input_path,
    output_path,
):
    await asyncio.to_thread(
        process_video,
        job_id,
        input_path,
        output_path,
    )


@app.get("/health")
async def health():
    with jobs_lock:
        active_jobs = sum(
            1
            for job in jobs.values()
            if job["status"]
            in {"queued", "processing"}
        )

    return {
        "status": "ok",
        "model": model_path.name,
        "classes": model.names,
        "confidence_threshold": (
            confidence_threshold
        ),
        "active_jobs": active_jobs,
    }


@app.post(
    "/api/videos",
    status_code=202,
)
async def upload_video(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    original_filename = (
        file.filename or "uploaded-video"
    )

    extension = Path(
        original_filename
    ).suffix.lower()

    if extension not in allowed_video_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Use MP4, MOV, AVI, MKV, "
                "M4V, or WEBM."
            ),
        )

    job_id = uuid.uuid4().hex

    input_path = (
        uploads_directory
        / f"{job_id}{extension}"
    )

    output_path = (
        outputs_directory
        / f"{job_id}.mp4"
    )

    uploaded_size = 0

    try:
        with input_path.open("wb") as destination:
            while chunk := await file.read(
                1024 * 1024
            ):
                uploaded_size += len(chunk)

                if (
                    uploaded_size
                    > maximum_upload_size
                ):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "The video exceeds "
                            "the 2 GB limit."
                        ),
                    )

                destination.write(chunk)

    except Exception:
        input_path.unlink(missing_ok=True)
        raise

    finally:
        await file.close()

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "filename": original_filename,
            "uploaded_bytes": uploaded_size,
            "uploaded_by": current_user["username"],
            "created_at": time.time(),
            "status": "queued",
            "progress": 0,
            "message": (
                "Waiting to begin processing"
            ),
            "processed_frames": 0,
            "total_frames": 0,
            "frames_with_detections": 0,
            "total_detections": 0,
            "detection_events": [],
            "review_summary": None,
            "input_path": str(input_path),
            "output_path": str(output_path),
        }

        job_snapshot = dict(jobs[job_id])

    job_store.write(job_id, job_snapshot)

    asyncio.create_task(
        process_video_in_background(
            job_id,
            input_path,
            output_path,
        )
    )

    return get_public_job(
        get_job_copy(job_id)
    )


@app.get("/api/jobs")
async def list_jobs(
    current_user: dict = Depends(get_current_user),
):
    with jobs_lock:
        all_jobs = [dict(job) for job in jobs.values()]

    all_jobs.sort(
        key=lambda job: job.get("created_at", 0),
        reverse=True,
    )

    return [get_public_job(job) for job in all_jobs]


@app.get("/api/dashboard")
async def get_dashboard(
    current_user: dict = Depends(get_current_user),
):
    with jobs_lock:
        all_jobs = [dict(job) for job in jobs.values()]

    all_jobs.sort(
        key=lambda job: job.get("created_at", 0),
        reverse=True,
    )

    totals = {
        "jobs": len(all_jobs),
        "completed_jobs": 0,
        "processing_jobs": 0,
        "failed_jobs": 0,
        "total_detections": 0,
        "pending_reviews": 0,
        "approved_reviews": 0,
        "rejected_reviews": 0,
        "corrected_reviews": 0,
    }

    for job in all_jobs:
        status = job.get("status")

        if status == "completed":
            totals["completed_jobs"] += 1
        elif status in {"queued", "processing"}:
            totals["processing_jobs"] += 1
        elif status == "failed":
            totals["failed_jobs"] += 1

        totals["total_detections"] += job.get(
            "total_detections", 0
        ) or 0

        review_summary = review_store.summary(
            job["job_id"]
        )

        if review_summary is not None:
            totals["pending_reviews"] += review_summary[
                "pending"
            ]
            totals["approved_reviews"] += review_summary[
                "approved"
            ]
            totals["rejected_reviews"] += review_summary[
                "rejected"
            ]
            totals["corrected_reviews"] += review_summary[
                "corrected"
            ]

    recent_jobs = [
        get_public_job(job) for job in all_jobs[:8]
    ]

    return {
        "totals": totals,
        "recent_jobs": recent_jobs,
    }


@app.get("/api/jobs/{job_id}")
async def get_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    job = get_job_copy(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return get_public_job(job)


@app.get("/api/jobs/{job_id}/result")
async def download_result(
    job_id: str,
    current_user: dict = Depends(
        get_current_user_flexible
    ),
):
    job = get_job_copy(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail="The result is not ready",
        )

    output_path = Path(job["output_path"])

    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail="The result file is missing",
        )

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=(
            f"bananaguard-{job_id}.mp4"
        ),
    )


def extract_frame(input_path, frame_number):
    video_capture = cv2.VideoCapture(str(input_path))

    try:
        if not video_capture.isOpened():
            return None

        video_capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number,
        )

        success, frame = video_capture.read()

        if not success or frame is None:
            return None

        encoded, buffer = cv2.imencode(".jpg", frame)

        if not encoded:
            return None

        return buffer.tobytes()

    finally:
        video_capture.release()


@app.get("/api/jobs/{job_id}/frames/{frame_number}")
async def get_frame(
    job_id: str,
    frame_number: int,
    current_user: dict = Depends(
        get_current_user_flexible
    ),
):
    if frame_number < 0:
        raise HTTPException(
            status_code=400,
            detail="frame_number must be non-negative",
        )

    job = get_job_copy(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    input_path = Path(job["input_path"])

    if not input_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Source video is no longer available",
        )

    frame_bytes = await asyncio.to_thread(
        extract_frame,
        input_path,
        frame_number,
    )

    if frame_bytes is None:
        raise HTTPException(
            status_code=404,
            detail="Frame not found",
        )

    return Response(
        content=frame_bytes,
        media_type="image/jpeg",
    )


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):
    token = websocket.query_params.get("token")

    if token is None:
        await websocket.close(code=4401)
        return

    try:
        auth_module.auth_service.user_from_token(token)
    except HTTPException:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    print("Webcam client connected")

    try:
        while True:
            data = (
                await websocket.receive_bytes()
            )

            image_array = np.frombuffer(
                data,
                np.uint8,
            )

            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR,
            )

            if image is None:
                continue

            result = await asyncio.to_thread(
                run_inference,
                image,
            )

            detections = serialize_detections(
                result
            )

            await websocket.send_json(
                detections
            )

    except WebSocketDisconnect:
        print("Webcam client disconnected")

    except Exception as error:
        print(f"WebSocket error: {error}")

        try:
            await websocket.close()
        except RuntimeError:
            pass


if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 8080)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )