"""物性、几何与单位换算的正确性测试（不需要 OpenMC）。"""

from __future__ import annotations

import math

import pytest

from src.physics import (
    BENCHMARK_COOLANT_DENSITY,
    PHYSICS_VERSION,
    boron_mass_fraction,
    expected_b_h2o_ratio,
    iapws_water_density,
    k_uo2,
    k_zircaloy,
    saturation_pressure,
    saturation_temperature,
    validate_pin_geometry,
    water_density,
)


def test_physics_version_is_bumped() -> None:
    assert PHYSICS_VERSION == "2.0"


def test_saturation_temperature_matches_vera_p1_pressure() -> None:
    """15.5 MPa 的饱和温度 ≈ 617.9 K，这是"慢化剂 600 K 仍是单相水"的依据。"""
    assert saturation_temperature(15.5) == pytest.approx(617.94, abs=0.5)
    # 采样上限 615 K 必须小于饱和温度，否则进入两相区
    assert 615.0 < saturation_temperature(15.5)


def test_saturation_pressure_roundtrip() -> None:
    assert saturation_pressure(600.0) == pytest.approx(12.34, abs=0.2)
    assert saturation_pressure(saturation_temperature(15.5)) == pytest.approx(15.5, abs=0.05)


def test_water_density_decreases_with_temperature() -> None:
    rho = [iapws_water_density(t) for t in (500.0, 550.0, 600.0, 610.0)]
    assert all(a > b for a, b in zip(rho, rho[1:]))
    assert rho[2] == pytest.approx(0.661, abs=0.005)


def test_benchmark_density_is_not_the_physical_one() -> None:
    """VERA P1 的 0.743 g/cm³ 与 IAPWS 不符（对应约 567 K），所以保留两个模型。"""
    assert water_density(600.0, model="constant") == pytest.approx(BENCHMARK_COOLANT_DENSITY)
    assert water_density(600.0) < BENCHMARK_COOLANT_DENSITY - 0.05


def test_water_density_model_is_validated() -> None:
    with pytest.raises(ValueError):
        water_density(600.0, model="magic")


def test_thermal_conductivities_are_physical() -> None:
    assert k_uo2(1000.0) == pytest.approx(0.0394, rel=0.05)  # W/(cm·K)
    assert k_zircaloy(600.0) == pytest.approx(0.165, rel=0.05)
    assert k_uo2(600.0) > k_uo2(1200.0)  # UO2 导热随温度下降


def test_boron_mass_fraction_matches_ppm() -> None:
    assert boron_mass_fraction(1300.0) == pytest.approx(1.3e-3)


def test_b_h2o_ratio_closes_the_mass_balance() -> None:
    """1300 wppm 对应 B/H2O ≈ 2.17e-3；旧的 add_element(默认 ao) 只给了 1.30e-3。"""
    ratio = expected_b_h2o_ratio(1300.0, BENCHMARK_COOLANT_DENSITY)
    assert ratio == pytest.approx(2.169e-3, rel=2e-3)
    assert ratio / 1.3e-3 == pytest.approx(1.668, rel=5e-3)


def test_pin_geometry_rejects_legacy_radius() -> None:
    """旧采样下界 clad_r = 0.4504 会让包壳内半径 0.3934 < 芯块 0.4096，物理上不存在。"""
    problems = validate_pin_geometry(0.4096, 0.3934, 0.4504, 1.26)
    assert problems


def test_pin_geometry_accepts_vera_p1() -> None:
    assert validate_pin_geometry(0.4096, 0.4180, 0.4750, 1.26) == []


def test_pin_geometry_rejects_overlapping_cladding() -> None:
    assert validate_pin_geometry(0.4096, 0.60, 0.50, 1.26)
    assert validate_pin_geometry(0.4096, 0.4180, 0.4750, 0.90)  # 栅距装不下两根包壳


def test_saturation_temperature_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        saturation_temperature(-1.0)
    assert math.isfinite(saturation_pressure(300.0))
