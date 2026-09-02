import json
import threading
from pathlib import Path


class JobStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def _path(self, job_id):
        return self.directory / f"{job_id}.json"

    def write(self, job_id, job):
        with self.lock:
            destination = self._path(job_id)
            temporary = destination.with_suffix(".tmp")

            temporary.write_text(
                json.dumps(job, indent=2),
                encoding="utf-8",
            )

            temporary.replace(destination)

    def load_all(self):
        jobs = {}

        for path in self.directory.glob("*.json"):
            try:
                jobs[path.stem] = json.loads(
                    path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                continue

        return jobs
