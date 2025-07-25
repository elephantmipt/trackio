import os
import sqlite3

from trackio.sqlite_storage import SQLiteStorage


def test_init_creates_metrics_table(temp_db):
    db_path = SQLiteStorage.init_db("proj1")
    assert os.path.exists(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM metrics")


def test_log_and_get_metrics(temp_db):
    metrics = {"acc": 0.9}
    SQLiteStorage.log(project="proj1", run="run1", metrics=metrics)
    results = SQLiteStorage.get_metrics(project="proj1", run="run1")
    assert len(results) == 1
    assert results[0]["acc"] == 0.9
    assert results[0]["step"] == 0
    assert "timestamp" in results[0]


def test_get_projects_and_runs(temp_db):
    SQLiteStorage.log(project="proj1", run="run1", metrics={"a": 1})
    SQLiteStorage.log(project="proj2", run="run2", metrics={"b": 2})
    projects = set(SQLiteStorage.get_projects())
    assert {"proj1", "proj2"}.issubset(projects)
    runs = set(SQLiteStorage.get_runs("proj1"))
    assert "run1" in runs


def test_log_image(temp_db, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"1234")

    SQLiteStorage.log_image(project="proj1", run="run1", image_path=str(img))
    images_dir = (
        SQLiteStorage.get_project_db_path("proj1").parent / "images" / "proj1" / "run1"
    )
    files = list(images_dir.iterdir())
    assert len(files) == 1

    with sqlite3.connect(SQLiteStorage.get_project_db_path("proj1")) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM images WHERE run_name=?", ("run1",))
        row = cursor.fetchone()
        assert row is not None
        assert row[0].endswith("img.png")

    images = SQLiteStorage.get_images(project="proj1", run="run1")
    assert len(images) == 1
    assert images[0]["image_path"].endswith("img.png")


def test_log_image_multiple_steps(temp_db, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"1234")

    SQLiteStorage.log_image(project="proj1", run="run1", image_path=str(img))
    SQLiteStorage.log_image(project="proj1", run="run1", image_path=str(img))
    images_dir = (
        SQLiteStorage.get_project_db_path("proj1").parent / "images" / "proj1" / "run1"
    )
    files = list(images_dir.iterdir())
    assert len(files) == 2

    images = SQLiteStorage.get_images(project="proj1", run="run1")
    assert [img_row["step"] for img_row in images] == [0, 1]
    assert images[0]["image_path"] != images[1]["image_path"]
