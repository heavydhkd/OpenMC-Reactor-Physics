"""配置自检：几何、边界与物性换算的一致性（不需要 OpenMC）。"""

from __future__ import annotations

import yaml

from src.config import load_config
from src.validate import check_physics


def test_shipped_config_passes_self_check(config) -> None:
    report = check_physics(config)
    assert report["ok"] is True, report["problems"]


def test_self_check_detects_impossible_geometry(config, tmp_path) -> None:
    """把旧代码的 clad_r 下界塞回去，自检必须报错而不是静默通过。"""
    data = yaml.safe_load(config.resolve("configs/params.yaml").read_text(encoding="utf-8"))
    data["sampling"]["bounds"]["clad_r"] = [0.4504, 0.490]  # 包壳内半径 0.3934 < 芯块 0.4096
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    report = check_physics(load_config(path))
    assert report["ok"] is False
    assert report["problems"]


def test_self_check_detects_two_phase_coolant(config, tmp_path) -> None:
    """慢化剂温度超过 15.5 MPa 的饱和温度（617.94 K）就不再是单相水。"""
    data = yaml.safe_load(config.resolve("configs/params.yaml").read_text(encoding="utf-8"))
    data["sampling"]["bounds"]["mod_temp"] = [500.0, 700.0]
    path = tmp_path / "two_phase.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    report = check_physics(load_config(path))
    assert report["ok"] is False
