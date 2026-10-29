"""一维热工模型：能量守恒、热阻串并联与 Picard 收敛。"""

from __future__ import annotations

import numpy as np
import pytest

from src.coupled_pin import fuel_center_temperature, solve_temperature


def test_energy_balance_closes(config) -> None:
    solution = solve_temperature(200.0, 600.0, config)
    total = solution.delta_fuel + solution.delta_gap + solution.delta_clad + solution.delta_film
    assert total == pytest.approx(solution.t_center - solution.mod_temp, abs=1e-6)


def test_temperature_rises_with_power(config) -> None:
    low = solve_temperature(150.0, 600.0, config)
    high = solve_temperature(250.0, 600.0, config)
    assert high.t_center > low.t_center
    assert high.t_fuel_avg > low.t_fuel_avg


def test_gap_drop_matches_fourier_law(config) -> None:
    solution = solve_temperature(200.0, 600.0, config)
    assert solution.delta_gap == pytest.approx(200.0 * solution.r_gap, rel=1e-9)


def test_picard_iteration_converges(config) -> None:
    solution = solve_temperature(200.0, 600.0, config)
    assert solution.converged is True
    assert 1 <= solution.iterations <= int(config["thermal.max_iterations"])
    # 收敛后最后两步的燃料平均温度变化应小于容差
    assert abs(solution.history[-1] - solution.history[-2]) < float(config["thermal.tolerance_k"])


def test_cladding_is_thinner_than_gap_conductance_ratio(config) -> None:
    solution = solve_temperature(200.0, 600.0, config)
    physics = config.physics()
    assert 0.0 < physics["clad_inner_radius_cm"] < physics["clad_outer_radius_cm"]
    assert solution.t_surface > solution.t_clad_inner > solution.t_clad_outer > solution.mod_temp


def test_center_temperature_is_above_surface(config) -> None:
    physics = config.physics()
    t_surface = 750.0
    t_center = fuel_center_temperature(200.0, float(physics["pellet_radius_cm"]), t_surface)
    assert t_center > t_surface
    # 线功率翻倍，中心-表面温差大约翻倍（UO2 导热随温度缓变）
    t_center2 = fuel_center_temperature(400.0, float(physics["pellet_radius_cm"]), t_surface)
    assert (t_center2 - t_surface) / (t_center - t_surface) == pytest.approx(2.0, rel=0.25)


def test_solution_serialises(config) -> None:
    solution = solve_temperature(200.0, 600.0, config)
    payload = solution.as_dict()
    assert payload["converged"] is True
    assert np.isfinite(payload["t_center"])
    assert payload["iterations"] == solution.iterations
    assert len(solution.history) == solution.iterations
