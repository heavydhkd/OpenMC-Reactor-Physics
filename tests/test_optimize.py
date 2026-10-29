"""临界设计寻优：目标函数的 pcm 标度、边界命中与外推告警。"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import FEATURE_COLUMNS
from src.optimize import criticality_pcm, check_extrapolation, optimize_criticality


class LinearSurrogate:
    """k = 1 + 1e-3·(B-1000) + 0.1·(E-3)，用于构造解析可验证的寻优点。"""

    def predict(self, X, return_std=True):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        k = 1.0 + 1.0e-3 * (X[:, 1] - 1000.0) + 0.1 * (X[:, 0] - 3.0)
        sigma = np.full(X.shape[0], 1.0e-4)
        return (k, sigma) if return_std else k


class StubBundle(dict):
    pass


def make_bundle(config) -> StubBundle:
    fixed = dict(config.section("optimization")["fixed"])
    ranges = {
        "enrichment": [1.5, 5.0],
        "boron_ppm": [0.0, 1200.0],
        "clad_r": [0.468, 0.490],
        "fuel_temp": [600.0, 1200.0],
        "mod_temp": [500.0, 615.0],
    }
    return StubBundle(
        pipeline=LinearSurrogate(),
        features=list(FEATURE_COLUMNS),
        feature_ranges=ranges,
        metrics={"test": {"rmse_pcm": 10.0}},
    )


def test_criticality_pcm_is_signed() -> None:
    assert criticality_pcm(make_bundle_simple(), np.array([3.0, 1000.0]), ["enrichment", "boron_ppm"],
                           {"clad_r": 0.475, "fuel_temp": 900.0, "mod_temp": 600.0}) == pytest.approx(0.0, abs=1e-6)


def make_bundle_simple() -> StubBundle:
    return StubBundle(pipeline=LinearSurrogate(), features=list(FEATURE_COLUMNS))


def test_optimizer_finds_the_critical_line(config) -> None:
    """旧代码的 abs(k-1) 目标停在初始点；改成 pcm 标度后才真的开始搜索。"""
    report = optimize_criticality(make_bundle(config), config, seed=0)
    assert report["success"] is True
    assert report["abs_residual_pcm"] < 1.0
    assert report["reached_target"] is True
    boron = float(report["optimum"]["boron_ppm"])
    enrichment = float(report["optimum"]["enrichment"])
    # 解析解：1e-3(B-1000) + 0.1(E-3) = 0
    assert 1.0e-3 * (boron - 1000.0) + 0.1 * (enrichment - 3.0) == pytest.approx(0.0, abs=1e-3)


def test_report_flags_extrapolation_beyond_training_range(config) -> None:
    bundle = make_bundle(config)
    outside = check_extrapolation(bundle, np.array([3.0, 2400.0]), list(FEATURE_COLUMNS))
    assert outside["outside_features"] == ["boron_ppm"]
    assert "boron_ppm" in outside["details"]

    inside = check_extrapolation(bundle, np.array([3.0, 800.0]), list(FEATURE_COLUMNS))
    assert inside["outside_features"] == []


def test_large_uncertainty_is_reported(config) -> None:
    class NoisySurrogate(LinearSurrogate):
        def predict(self, X, return_std=True):
            k = super().predict(X, return_std=False)
            return k, np.full(np.shape(k), 5.0e-3)  # 500 pcm，超出可信范围

    bundle = make_bundle(config)
    bundle["pipeline"] = NoisySurrogate()
    report = optimize_criticality(bundle, config, seed=0)
    assert any("1σ" in w or "置信" in w for w in report["warnings"])
