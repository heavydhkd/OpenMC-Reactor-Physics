"""训练 GP 代理模型，输出指标、对比图，并（可选）做临界设计寻优。

    python -m src.train_surrogate                 # 训练 + 出图
    python -m src.train_surrogate --optimize      # 训练后再寻优
    python -m src.train_surrogate --verify        # 训练 + 用 OpenMC 复核最优点（需要 OpenMC）

注意：仓库内现有的 ``data/processed/vera_p1_dataset.csv`` 是旧物理模型
（包壳内半径算错、硼按原子分数加入）算出的历史数据，只用于演示流程；
要得到可信的物理结果，请先安装 OpenMC，运行 ``python -m src.run_batch``
重新生成数据集。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .config import FEATURE_LABELS, PATHS, load_config
from .data import load_dataset
from .optimize import _print_report, optimize_criticality
from .plotting import configure_matplotlib
from .surrogate import PCM, cross_validate, predict_frame, save_metrics, save_model, split_indices, train

configure_matplotlib()  # 中文字体 + Agg 后端，必须在 pyplot 使用前调用

LOGGER = logging.getLogger(__name__)


def plot_parity(bundle: Dict[str, Any], df, out_path: Path, config=None) -> Path:
    """左图：预测 vs 真实（pcm 标高）；右图：残差与蒙特卡罗统计误差对比。

    训练点用灰色叉号、测试点用蓝色圆点 —— 只看训练点的拟合优度没有意义，
    报数用的 RMSE 只统计测试点。
    """
    config = config or load_config()
    frame = predict_frame(bundle, df)
    k_true = frame["k_eff"].to_numpy(dtype=float)
    k_pred = frame["k_pred"].to_numpy(dtype=float)
    sigma = frame["k_pred_std"].to_numpy(dtype=float)
    mc = frame["std"].to_numpy(dtype=float) if "std" in frame.columns else np.zeros_like(k_true)

    x = (k_true - 1.0) * PCM
    y = (k_pred - 1.0) * PCM
    residual = (k_true - k_pred) * PCM
    train_idx, test_idx = split_indices(len(frame), config)
    rmse_test = float(np.sqrt(np.mean(residual[test_idx] ** 2)))
    metric = bundle["metrics"]["test"]
    mc_sigma_pcm = float(np.mean(mc[test_idx]) * PCM)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    ax = axes[0]
    lo = float(min(x.min(), y.min())) - 400
    hi = float(max(x.max(), y.max())) + 400
    ax.fill_between([lo, hi], [lo - 2 * mc_sigma_pcm, hi - 2 * mc_sigma_pcm],
                    [lo + 2 * mc_sigma_pcm, hi + 2 * mc_sigma_pcm],
                    color="0.85", alpha=0.55, lw=0, label="±2σ 蒙特卡罗统计误差")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="预测 = 真实")
    ax.scatter(x[train_idx], y[train_idx], marker="x", s=26, c="0.55", label=f"训练集（{len(train_idx)}）")
    ax.errorbar(x[test_idx], y[test_idx], xerr=mc[test_idx] * PCM, yerr=sigma[test_idx] * PCM,
                fmt="o", ms=6, mfc="tab:blue", mec="white", ecolor="0.6", elinewidth=0.9,
                capsize=2, label=f"测试集（{len(test_idx)}，RMSE {rmse_test:.0f} pcm）")
    ax.set_xlabel("真实 $k_{eff}$ 偏离 1.0 (pcm)")
    ax.set_ylabel("GP 预测 $k_{eff}$ 偏离 1.0 (pcm)")
    ax.set_title(rf"测试集 RMSE = {rmse_test:.1f} pcm，$R^2$ = {metric['r2']:.4f}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    idx = np.arange(len(residual))
    ax.axhline(0.0, color="k", lw=1.0, ls="--")
    ax.fill_between([-0.5, len(residual) - 0.5], -2 * mc_sigma_pcm, 2 * mc_sigma_pcm,
                    color="tab:green", alpha=0.15, lw=0, label="±2σ 蒙特卡罗统计误差（测试集均值）")
    ax.scatter(idx[train_idx], residual[train_idx], marker="x", s=26, c="0.6", label="训练集残差")
    ax.errorbar(idx[test_idx], residual[test_idx], yerr=sigma[test_idx] * PCM, fmt="o", ms=6,
                color="tab:red", ecolor="0.6", elinewidth=0.9, capsize=2, label="测试集残差（含 GP 1σ）")
    ax.set_xlim(-0.5, len(residual) - 0.5)
    ax.set_xlabel("算例编号")
    ax.set_ylabel("残差 (pcm)")
    ax.set_title(f"残差与统计误差同量级即「已到噪声地板」（MC 1σ ≈ {mc_sigma_pcm:.0f} pcm）")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_cv(cv: Dict[str, Any], out_path: Path) -> Path:
    """交叉验证误差随折次的变化，附上蒙特卡罗统计误差参考线。"""
    folds = cv.get("folds") or []
    if len(folds) < 2:
        return out_path
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    xs = [f["fold"] for f in folds]
    ys = [f["rmse_pcm"] for f in folds]
    ax.bar(xs, ys, color="tab:blue", alpha=0.85, label="每折 RMSE")
    ax.axhline(cv["summary"]["rmse_pcm"], color="tab:red", ls="--", lw=1.2,
               label=f"平均 {cv['summary']['rmse_pcm']:.1f} pcm")
    if folds[0].get("mean_mc_sigma_pcm"):
        ax.axhline(folds[0]["mean_mc_sigma_pcm"], color="tab:green", ls=":", lw=1.2,
                   label=f"蒙特卡罗 1σ ≈ {folds[0]['mean_mc_sigma_pcm']:.1f} pcm")
    ax.set_xlabel("折")
    ax.set_ylabel("RMSE (pcm)")
    ax.set_title(f"{cv['summary']['n_splits']} 折交叉验证")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 GP 代理模型")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true", help="不出图（纯数值）")
    parser.add_argument("--optimize", action="store_true", help="训练后顺带做临界设计寻优")
    parser.add_argument("--verify", action="store_true", help="用 OpenMC 复核最优点（需要 OpenMC）")
    parser.add_argument("--cv", type=int, default=5, help="交叉验证折数，0 表示跳过")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    config = load_config(args.config)
    paths = PATHS.ensure()

    bundle = train(config)
    df = load_dataset(paths.dataset_csv)
    if args.cv > 0:
        bundle["metrics"]["cross_validation"] = cross_validate(df, config, n_splits=args.cv)

    model_path = save_model(bundle)
    metrics_path = save_metrics(bundle)
    LOGGER.info("模型已保存：%s", model_path)
    LOGGER.info("指标已保存：%s", metrics_path)

    if not args.no_plot:
        LOGGER.info("对比图：%s", plot_parity(bundle, df, paths.figures_dir / "prediction_comparison.png", config))
        LOGGER.info("交叉验证图：%s", plot_cv(bundle["metrics"]["cross_validation"], paths.figures_dir / "cross_validation.png"))

    test = bundle["metrics"]["test"]
    print("\n" + "-" * 72)
    print(f"测试集 ({test['n']} 个算例)：RMSE = {test['rmse_pcm']:.1f} pcm，"
          f"MAE = {test['mae_pcm']:.1f} pcm，R² = {test['r2']:.4f}")
    if "mean_mc_sigma_pcm" in test:
        print(f"对比：蒙特卡罗 1σ 平均 {test['mean_mc_sigma_pcm']:.1f} pcm，"
              f"残差/统计误差 = {test['residual_over_mc_noise']:.2f}")
    cv = bundle["metrics"].get("cross_validation") or {}
    if cv.get("summary"):
        print(f"{cv['summary']['n_splits']} 折交叉验证：RMSE = {cv['summary']['rmse_pcm']:.1f} pcm")
    print("-" * 72 + "\n")

    if args.optimize or args.verify:
        report = optimize_criticality(bundle, config)
        _print_report(report)
        out = paths.results_dir / "optimization.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("寻优结果：%s", out)
        if args.verify:
            from .validate import verify_design

            verify_report = verify_design(report["optimum"], config, paths=paths)
            print(json.dumps(verify_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
