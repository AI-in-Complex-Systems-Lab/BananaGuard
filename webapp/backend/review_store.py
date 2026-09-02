import json
import threading
from pathlib import Path


class ReviewStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def _path(self, job_id):
        return self.directory / f"{job_id}.json"

    def _write(self, job_id, records):
        destination = self._path(job_id)
        temporary = destination.with_suffix(".tmp")

        temporary.write_text(
            json.dumps(records, indent=2),
            encoding="utf-8",
        )

        temporary.replace(destination)

    def initialize(self, job_id, detection_events):
        records = []

        for event in detection_events:
            frame = event["frame"]
            timestamp = event["timestamp_seconds"]

            for index, detection in enumerate(
                event["detections"]
            ):
                records.append(
                    {
                        "detection_id": f"{frame}-{index}",
                        "job_id": job_id,
                        "frame": frame,
                        "timestamp_seconds": timestamp,
                        "original_label": detection["label"],
                        "label": detection["label"],
                        "score": detection["score"],
                        "box": detection["box"],
                        "status": "pending",
                        "notes": "",
                    }
                )

        with self.lock:
            self._write(job_id, records)

        return records

    def list(self, job_id):
        path = self._path(job_id)

        if not path.exists():
            return None

        with self.lock:
            return json.loads(
                path.read_text(encoding="utf-8")
            )

    def update(
        self,
        job_id,
        detection_id,
        status,
        label=None,
        box=None,
        notes=None,
    ):
        with self.lock:
            path = self._path(job_id)

            if not path.exists():
                return None

            records = json.loads(
                path.read_text(encoding="utf-8")
            )

            selected_record = None

            for record in records:
                if record["detection_id"] == detection_id:
                    record["status"] = status

                    if label is not None:
                        record["label"] = label

                    if box is not None:
                        record["box"] = box

                    if notes is not None:
                        record["notes"] = notes

                    selected_record = record
                    break

            if selected_record is None:
                return None

            self._write(job_id, records)

            return selected_record

    def summary(self, job_id):
        records = self.list(job_id)

        if records is None:
            return None

        summary = {
            "total": len(records),
            "pending": 0,
            "approved": 0,
            "rejected": 0,
            "corrected": 0,
        }

        for record in records:
            status = record["status"]

            if status in summary:
                summary[status] += 1

        return summary