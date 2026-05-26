from pathlib import Path

from plant_disease.data.class_map import (
    CLASS_DICTS,
    ClassInfo,
    load_class_map,
    lookup_class,
)


def _write_txt(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "classes.txt"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_class_map_basic(tmp_path):
    p = _write_txt(tmp_path, "0 0 0 0 0\n1 0 1 1 1\n")
    rows = load_class_map(p)
    assert len(rows) == 2
    assert rows[0] == ClassInfo(
        plant="苹果", health_status="未患病", disease_degree="健康", disease_name="健康"
    )
    assert rows[1].plant == "苹果"
    assert rows[1].disease_name == "苹果黑星病"


def test_load_class_map_missing_file(tmp_path):
    rows = load_class_map(tmp_path / "nope.txt")
    assert rows == []


def test_load_class_map_skips_malformed_rows(tmp_path):
    p = _write_txt(tmp_path, "0 0 0 0 0\nnot a row\n1 0 1 1 1\n")
    rows = load_class_map(p)
    assert len(rows) == 2


def test_load_class_map_utf8(tmp_path):
    p = _write_txt(tmp_path, "0 0 0 0 0\n")
    rows = load_class_map(p)
    assert "苹果" in rows[0].plant


def test_lookup_class_in_range():
    info = lookup_class([ClassInfo("a", "b", "c", "d")], 0)
    assert info.plant == "a"


def test_lookup_class_out_of_range_returns_placeholder():
    info = lookup_class([], 5)
    assert info.plant == "类别5"
    assert info.health_status == "未知"


def test_class_dicts_have_expected_keys():
    assert CLASS_DICTS["plant"][0] == "苹果"
    assert CLASS_DICTS["healthy"][0] == "未患病"
    assert CLASS_DICTS["degree"][0] == "健康"
    assert CLASS_DICTS["disease"][0] == "健康"
