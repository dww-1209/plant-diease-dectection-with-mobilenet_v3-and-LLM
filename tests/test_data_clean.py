"""Smoke tests for the dedup pipeline.

The legacy ``process_repeat`` deletes files in place; we exercise the lower-level
``load_files`` + ``derepeat`` functions and a tiny end-to-end run via
``process_repeat`` on a temporary tree.
"""

from pathlib import Path

from plant_disease.data.data_clean import derepeat, load_files, process_repeat


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_load_files_skips_non_images(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    names, paths = load_files(str(tmp_path))
    assert sorted(names) == ["a.jpg", "b.png"]
    assert set(paths.keys()) == {"a.jpg", "b.png"}


def test_derepeat_drops_duplicate_basenames():
    out = derepeat(["a.jpg", "b.jpg", "b.jpg", "c.jpg"])
    # b.jpg appears twice, derepeat drops every duplicate
    assert "b.jpg" not in out
    assert "a.jpg" in out
    assert "c.jpg" in out


def test_process_repeat_removes_overlap(tmp_path):
    train = tmp_path / "train"
    val = tmp_path / "val"
    _touch(train / "shared.jpg")
    _touch(train / "train_only.jpg")
    _touch(val / "shared.jpg")
    _touch(val / "val_only.jpg")

    process_repeat(str(train), str(val))

    # The shared file should be gone from BOTH sides.
    assert not (train / "shared.jpg").exists()
    assert not (val / "shared.jpg").exists()
    assert (train / "train_only.jpg").exists()
    assert (val / "val_only.jpg").exists()
