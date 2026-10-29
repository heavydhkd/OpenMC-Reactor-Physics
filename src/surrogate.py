"""高斯过程（GPR）代理模型：训练、评估、持久化、预测。

和历史版本的区别：

* **特征**：默认使用 ``configs/params.yaml`` 里的全部 5 个特征
  （富集度、硼浓度、包壳半径、燃料温度、慢化剂温度）。旧版只用前 3 个，
  把温度效应留给了噪声项，实测误差 1236.8 pcm（README 宣称 134.75 pcm）。
* **核函数**：不再把 ``length_scale=[1., 1., 1.]`` 写死成 3 维，
  而是按特征数自动生成，并可选 Matérn 5/2（对本问题拟合更好）。
* **噪声项**：``WhiteKernel`` 的噪声水平被约束在 OpenMC 统计误差量级
  （σ ≈ 1e-3 → 方差 ≈ 1e-6），防止 GP 用"大噪声"掩盖结构性误差。
* **报告口径**：误差统一折算成 pcm（``(k_eff - 1) × 1e5``），并与蒙特卡罗
  统计误差逐条对比 —— 这是判断代理模型"是否已经到噪声地板"的唯一可靠依据。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    Matern,
    RBF,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, FEATURE_LABELS, PATHS, Config, load_config
from .data import load_dataset

LOGGER = logging.getLogger(__name__)

PCM = 1.0e5  # k_eff → pcm 的换算系数
MIN_TRAIN_ROWS = 8


# ---------------------------------------------------------------------------
# 核函数与估计器
# ---------------------------------------------------------------------------
def build_kernel(kind: str, n_features: int) -> Any:
    """按配置构造核函数。``kind`` 取 ``rbf`` / ``matern`` / ``dot_rbf``。"""
    kind = (kind or "matern").lower()
    base = Matern(length_scale=np.ones(n_features), nu=2.5) if kind == "matern" else RBF(length_scale=np.ones(n_features))
    if kind == "dot_rbf":
        base = DotProduct(sigma_0=1.0) + base
    if kind not in {"rbf", "matern", "dot_rbf"}:
        raise ValueError(f"未知核函数：{kind}（可选 rbf / matern / dot_rbf）")
    return ConstantKernel(1.0, (1e-3, 1e3)) * base


def make_estimator(config: Config, n_features: int) -> Pipeline:
    """构造 ``StandardScaler + GPR`` 流水线（输入标准化，目标保持 k_eff 原尺度）。"""
    train_cfg = config.section("training")
    kernel = build_kernel(train_cfg.get("kernel", "matern"), n_features)
    noise = float(train_cfg.get("noise_level", 1e-6))
    noise_bounds = train_cfg.get("noise_level_bounds", (1e-9, 1e-4))
    kernel = kernel + WhiteKernel(noise_level=noise, noise_level_bounds=tuple(noise_bounds))
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=int(train_cfg.get("n_restarts_optimizer", 10)),
        normalize_y=False,  # 目标量级 ~1.0，标准化反而破坏 WhiteKernel 的物理解释
        random_state=int(train_cfg.get("random_state", 42)),
    )
    return Pipeline([("scaler", StandardScaler()), ("gpr", gpr)])


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------
def feature_matrix(df: pd.DataFrame, features: Sequence[str] = FEATURE_COLUMNS) -> np.ndarray:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"数据集缺少列：{missing}")
    return df[list(features)].to_numpy(dtype=float)


def to_pcm(k_eff: np.ndarray) -> np.ndarray:
    """k_eff 偏离 1.0 的量（pcm）。"""
    return (np.asarray(k_eff, dtype=float) - 1.0) * PCM


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray | None = None,
                       mc_sigma: np.ndarray | None = None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics: Dict[str, float] = {
        "n": int(y_true.size),
        "rmse_k": rmse,
        "rmse_pcm": rmse * PCM,
        "mae_pcm": float(mean_absolute_error(y_true, y_pred)) * PCM,
        "max_abs_err_pcm": float(np.max(np.abs(y_true - y_pred))) * PCM,
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
    }
    if sigma is not None and np.size(sigma):
        metrics["mean_pred_sigma_pcm"] = float(np.mean(sigma)) * PCM
    if mc_sigma is not None and np.size(mc_sigma):
        metrics["mean_mc_sigma_pcm"] = float(np.mean(mc_sigma)) * PCM
        # 用蒙特卡罗统计误差归一化后的残差：< ~2 说明代理模型已到噪声地板
        metrics["residual_over_mc_noise"] = float(rmse / max(float(np.mean(mc_sigma)), 1e-12))
    return metrics


def cross_validate(df: pd.DataFrame, config: Config, n_splits: int = 5) -> Dict[str, Any]:
    """K 折交叉验证，返回每折与汇总指标（pcm 口径）。"""
    features = config.features()
    X = feature_matrix(df, features)
    y = df["k_eff"].to_numpy(dtype=float)
    mc = df["std"].to_numpy(dtype=float) if "std" in df.columns else None

    n_splits = int(min(n_splits, len(df)))
    folds: List[Dict[str, float]] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(config["training.random_state"]))
    for fold_id, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        estimator = make_estimator(config, len(features)).fit(X[train_idx], y[train_idx])
        pred, sigma = estimator.predict(X[test_idx], return_std=True)
        m = regression_metrics(y[test_idx], pred, sigma, mc[test_idx] if mc is not None else None)
        m["fold"] = fold_id
        folds.append(m)
    summary = {
        key: float(np.nanmean([f[key] for f in folds]))
        for key in ("rmse_pcm", "mae_pcm", "r2", "mean_pred_sigma_pcm", "residual_over_mc_noise")
        if key in folds[0]
    }
    summary["n_splits"] = n_splits
    return {"folds": folds, "summary": summary}


# ---------------------------------------------------------------------------
# 训练与持久化
# ---------------------------------------------------------------------------
def train(config: Config | None = None) -> Dict[str, Any]:
    """用配置中的数据训练代理模型，返回可直接持久化的 bundle。"""
    config = config or load_config()
    features = config.features()
    paths = PATHS.ensure()
    df = load_dataset(paths.dataset_csv)
    if len(df) < MIN_TRAIN_ROWS:
        raise ValueError(f"有效样本只有 {len(df)} 条，至少需要 {MIN_TRAIN_ROWS} 条才能训练")

    X = feature_matrix(df, features)
    y = df["k_eff"].to_numpy(dtype=float)
    mc = df["std"].to_numpy(dtype=float) if "std" in df.columns else None

    test_size = float(config["training.test_size"])
    X_tr, X_te, y_tr, y_te, mc_tr, mc_te = _split(X, y, mc, test_size, int(config["training.random_state"]))

    estimator = make_estimator(config, len(features)).fit(X_tr, y_tr)
    pred_tr, sig_tr = estimator.predict(X_tr, return_std=True)
    pred_te, sig_te = estimator.predict(X_te, return_std=True)

    train_metrics = regression_metrics(y_tr, pred_tr, sig_tr, mc_tr)
    test_metrics = regression_metrics(y_te, pred_te, sig_te, mc_te)
    cv = cross_validate(df, config) if len(df) >= 12 else {"folds": [], "summary": {}}

    kernel = estimator.named_steps["gpr"].kernel_
    bundle: Dict[str, Any] = {
        "pipeline": estimator,
        "features": list(features),
        "feature_labels": {f: FEATURE_LABELS.get(f, f) for f in features},
        "metrics": {"train": train_metrics, "test": test_metrics, "cross_validation": cv},
        "kernel": str(kernel),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        # 训练域的每个特征范围：用于判断寻优结果是否属于外推（见 src/optimize.py）
        "feature_ranges": {
            f: [float(df[f].min()), float(df[f].max())] for f in features
        },
        "physics_version": str(df["physics_version"].iloc[0]) if "physics_version" in df.columns else "unknown",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": str(paths.dataset_csv),
    }

    LOGGER.info("训练集 %d 条 / 测试集 %d 条，特征 %s", len(y_tr), len(y_te), features)
    LOGGER.info("核函数：%s", kernel)
    LOGGER.info(
        "测试集：RMSE = %.1f pcm，R² = %.4f，最大误差 %.1f pcm（MC 1σ 平均 %.1f pcm）",
        test_metrics["rmse_pcm"], test_metrics["r2"], test_metrics["max_abs_err_pcm"],
        test_metrics.get("mean_mc_sigma_pcm", float("nan")),
    )
    if cv["summary"]:
        LOGGER.info("5 折交叉验证：RMSE = %.1f pcm（± %.1f）",
                    cv["summary"]["rmse_pcm"],
                    float(np.std([f["rmse_pcm"] for f in cv["folds"]])))
    return bundle


def split_indices(n_rows: int, config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """训练/测试划分的索引（评估与出图共用，保证看图看到的就是报数的那个划分）。"""
    idx = np.arange(n_rows)
    train_idx, test_idx = train_test_split(
        idx, test_size=float(config["training.test_size"]),
        random_state=int(config["training.random_state"]),
    )
    return np.sort(train_idx), np.sort(test_idx)


def _split(X, y, mc, test_size: float, seed: int):
    def _take(arr, idx):
        return None if arr is None else arr[idx]

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    return X[tr], X[te], y[tr], y[te], _take(mc, tr), _take(mc, te)


def save_model(bundle: Mapping[str, Any], path: Path | str | None = None) -> Path:
    path = Path(path) if path is not None else PATHS.ensure().model_joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(bundle), path)
    return path


def save_metrics(bundle: Mapping[str, Any], path: Path | str | None = None) -> Path:
    path = Path(path) if path is not None else PATHS.ensure().metrics_json
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in bundle.items() if k != "pipeline"}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_model(path: Path | str | None = None) -> Dict[str, Any]:
    path = Path(path) if path is not None else PATHS.ensure().model_joblib
    if not path.exists():
        raise FileNotFoundError(
            f"未找到已训练的模型：{path}\n请先运行 `python -m src.train_surrogate`。"
        )
    return joblib.load(path)


def predict_k(bundle: Mapping[str, Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """返回 ``(k_eff 均值, 1σ 标准差)``。"""
    estimator = bundle["pipeline"]
    mean, sigma = estimator.predict(np.atleast_2d(np.asarray(X, dtype=float)), return_std=True)
    return mean, sigma


def predict_frame(bundle: Mapping[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    features = bundle.get("features", list(FEATURE_COLUMNS))
    mean, sigma = predict_k(bundle, feature_matrix(df, features))
    out = df.copy()
    out["k_pred"] = mean
    out["k_pred_std"] = sigma
    if "k_eff" in out.columns:
        out["residual_pcm"] = to_pcm(out["k_eff"]) - to_pcm(out["k_pred"])
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    bundle = train()
    model_path = save_model(bundle)
    metrics_path = save_metrics(bundle)
    LOGGER.info("模型：%s", model_path)
    LOGGER.info("指标：%s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
