"""配置加载与路径解析的测试。"""

from __future__ import annotations

import pytest

from src.config import DATASET_COLUMNS, FEATURE_COLUMNS, PATHS, load_config


def test_features_are_five_dimensional(config) -> None:
    assert tuple(config.features()) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 5


def test_dotted_access(config) -> None:
    assert config["openmc_run.batches"] == 100
    assert config["sampling.bounds.boron_ppm"][1] == 2500
    assert config.get("does.not.exist", "fallback") == "fallback"
    with pytest.raises(KeyError):
        config["nope"]


def test_bounds_are_ordered_and_cover_design_space(config) -> None:
    bounds = config.bounds()
    assert set(bounds) == set(FEATURE_COLUMNS)
    for lo, hi in bounds.values():
        assert float(lo) < float(hi)
    lo, hi = bounds["mod_temp"]
    assert hi < 617.94  # 必须低于 15.5 MPa 的饱和温度


def test_physics_section_is_vera_p1_like(config) -> None:
    physics = config.physics()
    assert physics["pellet_radius_cm"] == pytest.approx(0.4096)
    assert physics["clad_inner_radius_cm"] == pytest.approx(0.4180)
    assert physics["clad_outer_radius_cm"] == pytest.approx(0.4750)
    assert physics["pitch_cm"] == pytest.approx(1.26)


def test_design_bounds_only_cover_design_variables(config) -> None:
    design = config.design_bounds()
    assert set(design) == set(config.section("optimization")["design_vars"])


def test_paths_are_repo_relative(repo_root) -> None:
    assert PATHS.root == repo_root
    assert PATHS.dataset_csv.parent.parent.name == "data"
    assert set(DATASET_COLUMNS) >= {"case_id", "k_eff", "status", "physics_version"}


def test_missing_config_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
