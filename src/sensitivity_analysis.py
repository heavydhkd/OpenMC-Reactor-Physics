"""基于代理模型的 Sobol 全局敏感性分析。

    python -m src.sensitivity_analysis

与历史版本的区别：

* **边界来自配置文件**，不再把 [1.5, 5.0] / [0, 1200] 之类的手写常量复制在代码里；
  参数一改，敏感性分析与采样、寻优自动同步。
* **特征完整**：5 个输入全部参与分解（旧版虽然建了 5 维问题，但模型只用了 3 维）。
* **给出置信区间**：输出 S1/ST 的 bootstrap 置信区间，并显式提示"当置信区间
  覆盖 0 时，该参数的影响不可与 0 区分"。
* **代价**：评估的是代理模型（毫秒级）。若要对 OpenMC 直接做 Sobol 分解，
  需要 4096 × (2×5+2) = 49152 次输运计算，不现实 —— 这也正是代理模型的
  价值所在，但前提是代理模型本身足够准（见 results/metrics.json 的 RMSE）。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .config import FEATURE_LABELS, FEATURE_COLUMNS, PATHS, Config, load_config
from .plotting import configure_matplotlib
from .surrogate import PCM, load_model, predict_k

configure_matplotlib()  # 中文字体 + Agg 后端

LOGGER = logging.getLogger(__name__)


def _require_salib():
    try:
        from SALib.analyze import sobol as sobol_analyze
        from SALib.sample import sobol as sobol_sample
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
        raise SystemExit(
            "缺少 SALib。请安装：\n    conda install -c conda-forge SALib\n"
            "或  pip install SALib"
        ) from exc
    return sobol_sample, sobol_analyze


def problem_from_config(config: Config, bundle: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """构造 Sobol 问题。边界与训练数据范围取交集，避免在无数据区做外推分解。"""
    bounds = config.bounds()
    features = config.features()
    ranges = (bundle or {}).get("feature_ranges") or {}
    problem_bounds: List[List[float]] = []
    clipped: Dict[str, List[float]] = {}
    for name in features:
        lo, hi = float(bounds[name][0]), float(bounds[name][1])
        if name in ranges:
            data_lo, data_hi = float(ranges[name][0]), float(ranges[name][1])
            new_lo, new_hi = max(lo, data_lo), min(hi, data_hi)
            if (new_lo, new_hi) != (lo, hi):
                clipped[name] = [new_lo, new_hi]
            lo, hi = new_lo, new_hi
        problem_bounds.append([lo, hi])
    if clipped:
        LOGGER.warning(
            "以下参数区间被训练数据范围裁剪（超出部分没有数据支撑）：%s", clipped
        )
    return {
        "num_vars": len(features),
        "names": list(features),
        "bounds": problem_bounds,
        "clipped": clipped,
    }


def _draw_samples(sobol_sample, problem, n_samples: int, second_order: bool, seed: int) -> np.ndarray:
    """兼容 SALib 各版本（``sobol.sample`` 取代了旧的 ``saltelli.sample``）。"""
    sampler = getattr(sobol_sample, "sample", None)
    if sampler is None:  # SALib < 1.4.6
        from SALib.sample import saltelli as sampler  # type: ignore
        sampler = sampler.sample
    try:
        return sampler(problem, n_samples, calc_second_order=second_order, seed=seed)
    except TypeError:  # 老版本没有 seed 参数
        rng = np.random.default_rng(seed)
        state = np.random.get_state()
        np.random.seed(int(rng.integers(0, 2 ** 31 - 1)))
        try:
            return sampler(problem, n_samples, calc_second_order=second_order)
        finally:
            np.random.set_state(state)


def _analyze(sobol_analyze, problem, Y: np.ndarray, second_order: bool, seed: int) -> Dict[str, Any]:
    try:
        result = sobol_analyze.analyze(problem, Y, calc_second_order=second_order,
                                       print_to_console=False, seed=seed)
    except TypeError:
        result = sobol_analyze.analyze(problem, Y, calc_second_order=second_order,
                                       print_to_console=False)
    return dict(result)


def analyze_sensitivity(bundle: Dict[str, Any], config: Config) -> Dict[str, Any]:
    sobol_sample, sobol_analyze = _require_salib()
    problem = problem_from_config(config, bundle)
    n_samples = int(config["sensitivity.n_samples"])
    second_order = bool(config["sensitivity.calc_second_order"])
    seed = int(config["sensitivity.random_state"])

    samples = _draw_samples(sobol_sample, problem, n_samples, second_order, seed)
    LOGGER.info("Sobol 样本：N = %d，总评估次数 %d（%d 维）", n_samples, len(samples), problem["num_vars"])

    k_pred, _ = predict_k(bundle, samples)
    elevation = (k_pred - 1.0) * PCM  # 用"偏离 1.0 的 pcm 量"做输出，物理含义更直观

    result = _analyze(sobol_analyze, problem, elevation, second_order, seed)
    names: List[str] = list(problem["names"])
    s1 = np.asarray(result["S1"], dtype=float)
    st = np.asarray(result["ST"], dtype=float)
    s1_conf = np.asarray(result.get("S1_conf", np.zeros_like(s1)), dtype=float)
    st_conf = np.asarray(result.get("ST_conf", np.zeros_like(st)), dtype=float)

    rows = [
        {
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "S1": float(s1[i]),
            "S1_conf": float(s1_conf[i] if i < len(s1_conf) else 0.0),
            "ST": float(st[i]),
            "ST_conf": float(st_conf[i] if i < len(st_conf) else 0.0),
            "interaction": float(st[i] - s1[i]),
            "significant": bool(abs(s1[i]) > (s1_conf[i] if i < len(s1_conf) else 0.0)),
        }
        for i, name in enumerate(names)
    ]
    rows.sort(key=lambda r: r["ST"], reverse=True)

    return {
        "n_samples": int(n_samples),
        "n_evaluations": int(len(samples)),
        "calc_second_order": second_order,
        "output": "k_eff 偏离 1.0 的量 (pcm)",
        "bounds_used": {name: list(b) for name, b in zip(names, problem["bounds"])},
        "clipped_bounds": problem.get("clipped", {}),
        "surrogate_rmse_pcm": float(bundle["metrics"]["test"]["rmse_pcm"]),
        "sum_S1": float(np.sum(s1)),
        "sum_ST": float(np.sum(st)),
        "indices": rows,
        "second_order": _second_order_table(result, names) if second_order else [],
        "mean_output_pcm": float(np.mean(elevation)),
        "std_output_pcm": float(np.std(elevation)),
    }


def _second_order_table(result: Dict[str, Any], names: Sequence[str]) -> List[Dict[str, Any]]:
    s2 = result.get("S2")
    if s2 is None:
        return []
    s2 = np.asarray(s2, dtype=float)
    table: List[Dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            table.append({"pair": f"{names[i]} × {names[j]}", "S2": float(s2[i, j])})
    table.sort(key=lambda r: abs(r["S2"]), reverse=True)
    return table


def print_table(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("Sobol 全局敏感性分析（输出：k_eff 偏离 1.0 的量，单位 pcm）")
    print("=" * 78)
    print(f"{'参数':<26}{'S1 (一阶)':>18}{'ST (总效应)':>20}{'交互项':>12}")
    print("-" * 78)
    for row in report["indices"]:
        star = "" if row["significant"] else "  ← 与 0 不可区分"
        print(f"{row['label']:<26}{row['S1']:>10.4f} ± {row['S1_conf']:<5.4f}"
              f"{row['ST']:>10.4f} ± {row['ST_conf']:<5.4f}{row['interaction']:>12.4f}{star}")
    print("-" * 78)
    print(f"Σ S1 = {report['sum_S1']:.4f}（交互作用占 {max(0.0, 1.0 - report['sum_S1']):.1%}）")
    print(f"Σ ST = {report['sum_ST']:.4f}（≈ 1 说明分解自洽）")
    print(f"输出标准差 {report['std_output_pcm']:.0f} pcm；代理模型测试 RMSE "
          f"{report['surrogate_rmse_pcm']:.0f} pcm")
    if report.get("clipped_bounds"):
        clipped = "，".join(f"{FEATURE_LABELS.get(k, k)} → [{v[0]:g}, {v[1]:g}]"
                            for k, v in report["clipped_bounds"].items())
        print(f"注：区间已按训练数据范围裁剪（{clipped}），超出部分无数据支撑。")
    if report["second_order"] and report["second_order"][0]["S2"] > 0.02:
        top = report["second_order"][0]
        print(f"最强二阶交互：{top['pair']}，S2 = {top['S2']:.4f}")
    print("=" * 78 + "\n")


def plot_indices(report: Dict[str, Any], out_path: Path) -> Path:
    rows = report["indices"]
    labels = [r["label"] for r in rows]
    s1 = np.array([r["S1"] for r in rows])
    st = np.array([r["ST"] for r in rows])
    s1_err = np.array([r["S1_conf"] for r in rows])
    st_err = np.array([r["ST_conf"] for r in rows])

    y = np.arange(len(rows))
    height = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    ax.barh(y + height / 2, st, height, xerr=st_err, label="总效应 $S_T$",
            color="tab:orange", alpha=0.9, error_kw={"elinewidth": 0.9, "capsize": 2})
    ax.barh(y - height / 2, s1, height, xerr=s1_err, label="一阶 $S_1$",
            color="tab:blue", alpha=0.9, error_kw={"elinewidth": 0.9, "capsize": 2})
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Sobol 指数")
    ax.set_title("$k_{eff}$ 对各输入参数的全局敏感性（误差棒为 bootstrap 置信区间）")
    ax.legend(loc="center right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(left=min(-0.02, float(s1.min()) - 0.02))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sobol 全局敏感性分析（基于 GP 代理模型）")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=None, help="覆盖 sensitivity.n_samples")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    config = load_config(args.config)
    if args.samples:
        config["sensitivity.n_samples"] = int(args.samples)
    paths = PATHS.ensure()
    bundle = load_model(args.model or paths.model_joblib)
    report = analyze_sensitivity(bundle, config)
    print_table(report)
    figure = plot_indices(report, paths.figures_dir / "sobol_sensitivity.png")
    LOGGER.info("图：%s", figure)
    out = paths.results_dir / "sensitivity.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("结果：%s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
