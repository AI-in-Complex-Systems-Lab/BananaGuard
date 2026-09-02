from job_store import JobStore


def test_write_and_load_all(tmp_path):
    store = JobStore(tmp_path / "jobs")

    store.write(
        "job-1",
        {"job_id": "job-1", "status": "completed"},
    )

    store.write(
        "job-2",
        {"job_id": "job-2", "status": "failed"},
    )

    jobs = store.load_all()

    assert jobs["job-1"]["status"] == "completed"
    assert jobs["job-2"]["status"] == "failed"


def test_write_overwrites_existing(tmp_path):
    store = JobStore(tmp_path / "jobs")

    store.write("job-1", {"job_id": "job-1", "progress": 0})
    store.write("job-1", {"job_id": "job-1", "progress": 50})

    jobs = store.load_all()

    assert jobs["job-1"]["progress"] == 50


def test_load_all_skips_corrupted_files(tmp_path):
    directory = tmp_path / "jobs"
    store = JobStore(directory)

    store.write("good-job", {"job_id": "good-job"})

    corrupted_path = directory / "corrupted.json"
    corrupted_path.write_text("{not valid json", encoding="utf-8")

    jobs = store.load_all()

    assert "good-job" in jobs
    assert "corrupted" not in jobs


def test_load_all_on_empty_directory(tmp_path):
    store = JobStore(tmp_path / "jobs")

    assert store.load_all() == {}
