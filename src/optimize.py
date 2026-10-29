"""在代理模型上做逆向寻优：求给定约束下使 k_eff → 1.0 的富集度与硼浓度。

历史版本的三个问题都在这里修掉：

1. **目标函数量纲**。旧版目标 ``abs(k_pred - 1.0)`` 的量级只有 1e-2，而 SLSQP
   用有限差分算梯度时，硼浓度方向上的步长约 1e-5 ppm，乘以灵敏度
   −6 pcm/ppm 之后函数变化只有 ~1e-10，低于浮点分辨率 —— 梯度恒等于 0，
   SLSQP 直接"原地不动"返回初始点，却打印 "Optimization terminated
   successfully"。现在目标函数统一换算成 pcm（乘 1e5），并且默认使用
   全局优化器 ``differential_evolution``。
2. **可行性**。旧版硼浓度上限 1200 ppm 时 k_eff 最低只能到 1.0138
   （残差 1376 pcm），"109 pcm 验证通过"是对训练域外的外推。现在上限
   提升到 2500 ppm，并显式报告结果是否落在边界、是否超出训练数据范围。
3. **最优解的不确定度**。结果同时给出 GP 预测 1σ，若该值远大于蒙特卡罗
   统计误差，说明最优点本身不可信，必须用 OpenMC 复核。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .config import FEATURE_COLUMNS, FEATURE_LABELS, PATHS, Config, load_config
from .surrogate import PCM, load_model, predict_k

LOGGER = logging.getLogger(__name__)

BOUND_TOL = 1e-3  # 判定"落在边界上"的容差（特征自身单位）


# ---------------------------------------------------------------------------
# 目标函数
# ---------------------------------------------------------------------------
def _assemble(x: Sequence[float], design_vars: Sequence[str], fixed: Mapping[str, float],
              features: Sequence[str]) -> np.ndarray:
    """把寻优变量与固定参数拼成完整特征向量（顺序与训练时一致）。"""
    params = dict(fixed)
    params.update({name: float(value) for name, value in zip(design_vars, x)})
    missing = [f for f in features if f not in params]
    if missing:
        raise KeyError(f"缺少固定参数：{missing}（请在 configs/params.yaml 的 optimization.fixed 中补齐）")
    return np.array([float(params[f]) for f in features], dtype=float)


def criticality_pcm(bundle: Mapping[str, Any], x: Sequence[float], design_vars: Sequence[str],
                    fixed: Mapping[str, float]) -> float:
    """代理模型预测的 ``(k_eff - 1) × 1e5``。"""
    features = bundle.get("features", list(FEATURE_COLUMNS))
    feature_vector = _assemble(x, design_vars, fixed, features)
    pred, _ = predict_k(bundle, feature_vector.reshape(1, -1))
    return float((pred[0] - 1.0) * PCM)


# ---------------------------------------------------------------------------
# 寻优
# ---------------------------------------------------------------------------
def optimize_criticality(bundle: Mapping[str, Any], config: Config,
                         seed: int | None = None) -> Dict[str, Any]:
    """在 ``optimization`` 配置段描述的二维设计空间中寻找临界设计。"""
    opt_cfg = config.section("optimization")
    features = bundle.get("features", list(FEATURE_COLUMNS))
    design_vars: List[str] = list(opt_cfg.get("design_vars", ["enrichment", "boron_ppm"]))
    fixed: Dict[str, float] = {k: float(v) for k, v in opt_cfg.get("fixed", {}).items()}
    bounds_cfg = opt_cfg.get("bounds", config.design_bounds())
    bounds = [(float(bounds_cfg[v][0]), float(bounds_cfg[v][1])) for v in design_vars]
    method = str(opt_cfg.get("method", "differential_evolution")).lower()
    seed = int(seed if seed is not None else config["training.random_state"])

    evaluations: List[float] = []

    def objective(x: np.ndarray) -> float:
        value = criticality_pcm(bundle, x, design_vars, fixed)
        evaluations.append(value)
        return abs(value)  # 目标：|k-1| 最小，量纲 pcm

    LOGGER.info("寻优变量 %s，边界 %s，方法 %s", design_vars, bounds, method)

    if method in {"differential_evolution", "de", "global"}:
        result = differential_evolution(
            objective, bounds, seed=seed, tol=1e-8, atol=0.0,
            popsize=25, maxiter=800, polish=True, init="sobol",
        )
        optimizer_name = "differential_evolution"
    elif method in {"slsqp", "local"}:
        x0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds])
        result = minimize(objective, x0=x0, method="SLSQP", bounds=bounds,
                          options={"maxiter": 500, "ftol": 1e-10})
        optimizer_name = "SLSQP"
    else:
        raise ValueError(f"未知寻优方法：{method}（可选 differential_evolution / slsqp）")

    x_best = np.atleast_1d(result.x)
    feature_vector = _assemble(x_best, design_vars, fixed, features)
    pred, sigma = predict_k(bundle, feature_vector.reshape(1, -1))
    k_pred = float(pred[0])
    sigma_pcm = float(sigma[0]) * PCM
    residual_pcm = float((k_pred - 1.0) * PCM)

    active = [
        {
            "feature": name,
            "value": float(value),
            "bound": float(bounds[i][0] if abs(value - bounds[i][0]) < BOUND_TOL else bounds[i][1]),
            "at_lower": bool(abs(value - bounds[i][0]) < BOUND_TOL),
        }
        for i, (name, value) in enumerate(zip(design_vars, x_best))
        if min(abs(value - bounds[i][0]), abs(value - bounds[i][1])) < BOUND_TOL
    ]

    report: Dict[str, Any] = {
        "method": optimizer_name,
        "design_vars": design_vars,
        "fixed": fixed,
        "bounds": {name: list(b) for name, b in zip(design_vars, bounds)},
        "optimum": {name: float(value) for name, value in zip(design_vars, x_best)},
        "k_pred": k_pred,
        "k_pred_sigma": float(sigma[0]),
        "k_pred_sigma_pcm": sigma_pcm,
        "residual_pcm": residual_pcm,
        "abs_residual_pcm": abs(residual_pcm),
        "objective_value_pcm": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "n_evaluations": int(len(evaluations)),
        "active_bounds": active,
        "reached_target": bool(abs(residual_pcm) <= 3.0 * max(sigma_pcm, 1.0)),
    }
    report["extrapolation"] = check_extrapolation(bundle, feature_vector, features)
    report["warnings"] = build_warnings(report)
    return report


def check_extrapolation(bundle: Mapping[str, Any], feature_vector: np.ndarray,
                        features: Sequence[str]) -> Dict[str, Any]:
    """判断最优点是否落在训练数据范围之外（外推 = 结论不可信）。"""
    ranges = bundle.get("feature_ranges") or {}
    outside: List[str] = []
    details: Dict[str, Dict[str, float]] = {}
    for name, value in zip(features, np.atleast_1d(feature_vector)):
        if name not in ranges:
            continue
        lo, hi = float(ranges[name][0]), float(ranges[name][1])
        details[name] = {"value": float(value), "train_min": lo, "train_max": hi}
        if value < lo - 1e-9 or value > hi + 1e-9:
            outside.append(name)
    return {"outside_features": outside, "details": details}


def build_warnings(report: Mapping[str, Any]) -> List[str]:
    warnings: List[str] = []
    if report["active_bounds"]:
        names = "、".join(item["feature"] for item in report["active_bounds"])
        warnings.append(
            f"最优解被边界卡住（{names}）：说明当前设计空间内无法同时满足目标，"
            "请放宽边界或调整固定参数，否则残差只是「边界效应」，不是物理结论。"
        )
    if report["abs_residual_pcm"] > 3.0 * max(report["k_pred_sigma_pcm"], 1.0):
        warnings.append(
            f"残差 {report['abs_residual_pcm']:.0f} pcm 远超代理模型不确定度 "
            f"{report['k_pred_sigma_pcm']:.0f} pcm，未达到临界。"
        )
    outside = report.get("extrapolation", {}).get("outside_features", [])
    if outside:
        warnings.append(
            f"最优点外推超出了训练数据范围（{('、'.join(outside))}），"
            "代理模型在该处没有数据支撑，必须用 OpenMC 直接计算复核。"
        )
    if report["k_pred_sigma_pcm"] > 300.0:
        warnings.append(
            f"最优点处 GP 的 1σ 达 {report['k_pred_sigma_pcm']:.0f} pcm，"
            "远大于蒙特卡罗统计误差（~100 pcm），该最优点的置信度有限。"
        )
    if not report["success"]:
        warnings.append(f"优化器未正常收敛：{report['message']}")
    return warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_report(report: Mapping[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("临界设计寻优结果（代理模型）")
    print("=" * 72)
    for name, value in report["optimum"].items():
        label = FEATURE_LABELS.get(name, name)
        lo, hi = report["bounds"][name]
        print(f"  {label:>16} = {value:12.4f}   [{lo:g}, {hi:g}]")
    print("-" * 72)
    print(f"  k_eff          = {report['k_pred']:.5f} ± {report['k_pred_sigma']:.5f}")
    print(f"  残差            = {report['residual_pcm']:+.0f} pcm "
          f"(± {report['k_pred_sigma_pcm']:.0f} pcm)")
    print(f"  优化器          = {report['method']}（{report['n_evaluations']} 次评估）")
    print(f"  达到临界        = {'是' if report['reached_target'] else '否'}")
    if report["warnings"]:
        print("-" * 72)
        for warning in report["warnings"]:
            print(f"  ⚠ {warning}")
    print("=" * 72 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用代理模型求临界富集度/硼浓度")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None,
                        help="模型文件，缺省用 results/surrogate.joblib")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", type=Path, nargs="?", const=PATHS.results_dir / "optimization.json",
                        default=None, help="结果 JSON 路径（省略路径时写到 results/optimization.json）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    config = load_config(args.config)
    paths = PATHS.ensure()
    bundle = load_model(args.model or paths.model_joblib)
    report = optimize_criticality(bundle, config, seed=args.seed)
    _print_report(report)
    out = Path(args.save) if args.save else (paths.results_dir / "optimization.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("寻优结果已写入 %s", out)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
