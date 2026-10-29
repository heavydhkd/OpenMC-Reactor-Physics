"""代理模型：核函数、指标、交叉验证与预测接口。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.surrogate import (
    MIN_TRAIN_ROWS,
    PCM,
    build_kernel,
    cross_validate,
    make_estimator,
    predict_frame,
    predict_k,
    regression_metrics,
    split_indices,
    to_pcm,
)


def synthetic_dataset(n: int = 48, seed: int = 0) -> pd.DataFrame:
    """k_eff 由 5 个参数线性决定 + 1e-3 量级的蒙特卡罗噪声。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "enrichment": rng.uniform(1.5, 5.0, n),
        "boron_ppm": rng.uniform(0.0, 2000.0, n),
        "clad_r": rng.uniform(0.468, 0.490, n),
        "fuel_temp": rng.uniform(600.0, 1200.0, n),
        "mod_temp": rng.uniform(500.0, 615.0, n),
    })
    k = (1.0
         + 0.05 * (df["enrichment"] - 3.0)
         - 2.0e-4 * (df["boron_ppm"] - 1000.0)
         - 1.0 * (df["clad_r"] - 0.475)
         - 2.0e-5 * (df["fuel_temp"] - 900.0)
         - 5.0e-5 * (df["mod_temp"] - 600.0))
    df["k_eff"] = k + rng.normal(0.0, 1.0e-3, n)
    df["std"] = 1.0e-3
    df["case_id"] = np.arange(n)
    df["status"] = "success"
    return df


def test_pcm_conversion() -> None:
    assert to_pcm(1.001) == pytest.approx(100.0)
    assert PCM == 1.0e5


@pytest.mark.parametrize("kind", ["rbf", "matern", "dot_rbf"])
def test_kernel_builds_for_any_dimension(kind: str, config) -> None:
    estimator = make_estimator(config, len(FEATURE_COLUMNS))
    assert estimator.named_steps["gpr"].kernel is not None
    df = synthetic_dataset(24)
    X = df[list(FEATURE_COLUMNS)].to_numpy()
    estimator.fit(X, df["k_eff"].to_numpy())
    mean, sigma = estimator.predict(X[:3], return_std=True)
    assert mean.shape == (3,) and np.all(sigma >= 0.0)
    assert np.all(np.isfinite(mean))


def test_bad_kernel_name_is_rejected(config) -> None:
    with pytest.raises(ValueError):
        build_kernel("quantum", len(FEATURE_COLUMNS))


def test_regression_metrics_are_in_pcm() -> None:
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([1.0 + 1e-4, 1.0 - 1e-4])
    sigma = np.full(2, 1e-4)
    mc = np.full(2, 1e-4)
    metrics = regression_metrics(y_true, y_pred, sigma, mc)
    assert metrics["rmse_pcm"] == pytest.approx(10.0)
    assert metrics["mae_pcm"] == pytest.approx(10.0)
    # 残差 10 pcm / 统计误差 10 pcm
    assert metrics["residual_over_mc_noise"] == pytest.approx(1.0)


def test_cross_validation_recovers_linear_surrogate(config) -> None:
    df = synthetic_dataset(60)
    cv = cross_validate(df, config, n_splits=5)
    assert len(cv["folds"]) == 5
    # 目标函数是线性的、噪声 100 pcm：交叉验证误差应当落在噪声地板附近
    assert cv["summary"]["rmse_pcm"] < 200.0
    assert cv["summary"]["residual_over_mc_noise"] < 2.0


def test_split_indices_are_disjoint_and_deterministic(config) -> None:
    train_idx, test_idx = split_indices(60, config)
    assert len(train_idx) + len(test_idx) == 60
    assert not set(train_idx) & set(test_idx)
    again = split_indices(60, config)
    assert np.array_equal(train_idx, again[0])


def test_predict_frame_keeps_rows(config) -> None:
    df = synthetic_dataset(30)
    bundle = {"features": list(FEATURE_COLUMNS), "pipeline": make_estimator(config, 5)}
    X = df[list(FEATURE_COLUMNS)].to_numpy()
    bundle["pipeline"].fit(X, df["k_eff"].to_numpy())
    frame = predict_frame(bundle, df)
    assert len(frame) == len(df)
    assert {"k_pred", "k_pred_std", "residual_pcm"} <= set(frame.columns)
    mean, sigma = predict_k(bundle, X[:2])
    assert mean.shape == sigma.shape == (2,)


def test_min_train_rows_constant() -> None:
    assert MIN_TRAIN_ROWS >= 8
