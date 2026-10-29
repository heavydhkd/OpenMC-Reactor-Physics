"""数据集 schema 与断点续跑相关的测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DATASET_COLUMNS
from src.data import append_case, case_record, load_cases, load_dataset, records_to_frame, save_dataset


def test_failed_case_keeps_full_schema() -> None:
    """旧代码失败时只写 {case_id, status}，to_csv 会写出参差不齐的一行。"""
    record = case_record(7, {"enrichment": 3.1}, k_eff=None, std=None, status="failed")
    assert set(record) == set(DATASET_COLUMNS)
    assert record["status"] == "failed"
    assert pd.isna(record["k_eff"])
    assert record["case_id"] == 7


def test_case_record_copies_parameters() -> None:
    params = {"enrichment": 4.2, "boron_ppm": 800.0, "clad_r": 0.475, "fuel_temp": 900.0, "mod_temp": 600.0}
    record = case_record(3, params, k_eff=1.1, std=1e-3)
    assert record["enrichment"] == 4.2
    assert record["k_eff"] == 1.1
    assert record["status"] == "success"
    assert record["physics_version"] == "unknown"


def test_roundtrip_through_csv(tmp_path) -> None:
    rows = [
        case_record(1, {"enrichment": 2.0, "boron_ppm": 100.0, "clad_r": 0.475, "fuel_temp": 900.0,
                        "mod_temp": 600.0}, k_eff=1.05, std=1e-3),
        case_record(2, {"enrichment": 3.0, "boron_ppm": 200.0, "clad_r": 0.475, "fuel_temp": 900.0,
                        "mod_temp": 600.0}, k_eff=None, std=None, status="failed"),
    ]
    path = tmp_path / "dataset.csv"
    save_dataset(records_to_frame(rows), path)
    loaded = load_dataset(path, drop_failed=False, require_min_rows=1)
    assert len(loaded) == 2
    assert loaded["status"].tolist() == ["success", "failed"]
    dropped = load_dataset(path, drop_failed=True, require_min_rows=1)
    assert len(dropped) == 1


def test_load_dataset_rejects_empty(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=list(DATASET_COLUMNS)).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_dataset(path, require_min_rows=1)


def test_append_case_is_resumable(tmp_path) -> None:
    """每算完一个写一行 JSONL：中断后 --resume 能跳过已完成的算例。"""
    path = tmp_path / "cases.jsonl"
    append_case(case_record(1, {"enrichment": 2.0}, k_eff=1.05, std=1e-3), path)
    append_case(case_record(2, {"enrichment": 2.5}, k_eff=None, status="failed"), path)
    cases = load_cases(path)
    assert set(cases) == {1, 2}
    assert cases[2]["status"] == "failed"


def test_repo_dataset_is_usable(dataset) -> None:
    assert len(dataset) >= 50
    assert {"k_eff", "std", "enrichment", "boron_ppm"} <= set(dataset.columns)
    assert (dataset["k_eff"] > 1.0).all()
