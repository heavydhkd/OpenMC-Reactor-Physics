"""Streamlit 交互界面：用真实训练的 GP 代理模型做在线预测。

历史版本的问题：界面写着"基于 OpenMC 数据与 AI 代理模型"，实际却在启动时
用 ``keff = 0.8 + 0.08*enrichment + 0.3*density + np.random.normal(...)``
现场编了 200 条假数据去训练随机森林，并且把模型对象塞进了 ``@st.cache_data``。

现在：模型和数据集都从磁盘读取，来源、训练指标、物理版本全部显示在页面上；
找不到模型时给出明确的训练命令，而不是退回假数据。

    streamlit run web_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import FEATURE_COLUMNS, FEATURE_LABELS, PATHS, load_config
from src.data import load_dataset
from src.surrogate import PCM, load_model, predict_k

CONFIG = load_config()
PATHS = PATHS.ensure()

st.set_page_config(page_title="VERA P1 栅元代理模型", page_icon="⚛️", layout="wide")


# ---------------------------------------------------------------------------
# 资源缓存
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    """模型对象用 cache_resource（不可哈希、需要复用同一个实例）。"""
    return load_model()


@st.cache_data(show_spinner=False)
def get_dataset():
    return load_dataset(PATHS.dataset_csv)


@st.cache_data(show_spinner=False)
def get_metrics():
    path = PATHS.metrics_json
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data(show_spinner=False)
def get_bounds() -> dict:
    return CONFIG.bounds()


def predict(params: dict) -> tuple[float, float]:
    bundle = get_model()
    features = bundle.get("features", list(FEATURE_COLUMNS))
    vector = np.array([[float(params[f]) for f in features]], dtype=float)
    mean, sigma = predict_k(bundle, vector)
    return float(mean[0]), float(sigma[0])


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
def sidebar() -> dict:
    bounds = get_bounds()
    st.sidebar.header("输入参数")
    st.sidebar.caption("范围与 configs/params.yaml 中的采样边界一致；越界即为外推。")

    def slider(name: str, step: float, fmt: str) -> float:
        lo, hi = bounds[name]
        return st.sidebar.slider(
            FEATURE_LABELS.get(name, name), float(lo), float(hi),
            float(np.clip(CONFIG.section("optimization").get("fixed", {}).get(name, 0.5 * (lo + hi)), lo, hi)),
            step=step, format=fmt,
        )

    params = {
        "enrichment": slider("enrichment", 0.01, "%.2f"),
        "boron_ppm": slider("boron_ppm", 10.0, "%.0f"),
        "clad_r": slider("clad_r", 0.001, "%.3f"),
        "fuel_temp": slider("fuel_temp", 5.0, "%.0f"),
        "mod_temp": slider("mod_temp", 5.0, "%.0f"),
    }
    st.sidebar.divider()
    st.sidebar.caption(
        f"物理模型版本：{CONFIG['meta.physics_version']}｜"
        f"训练特征数：{len(CONFIG.features())}"
    )
    return params


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
def render() -> None:
    st.title("⚛️ VERA P1 栅元：中子学代理模型")
    st.caption(
        "高斯过程（GPR）代理模型在线预测 $k_{eff}$。"
        "模型、数据集与训练指标均从 `results/` 目录读取，页面本身不生成任何数据。"
    )

    metrics = get_metrics()
    physics_version = metrics.get("physics_version", "unknown")
    if physics_version not in ("2.0", "2"):
        st.warning(
            "⚠️ 当前模型训练所用的数据集是**旧物理模型**（包壳内半径用 "
            "`clad_outer_r - 0.057` 反推、硼按原子分数加入）产生的历史数据。"
            "数值仅用于演示流程；要得到可信的物理结果，请安装 OpenMC 后运行 "
            "`python -m src.run_batch` 重新生成数据集，再运行 "
            "`python -m src.train_surrogate`。",
            icon="⚠️",
        )

    try:
        model_bundle = get_model()
    except Exception as exc:  # 模型缺失
        st.error(f"未找到已训练的模型：{exc}")
        st.code("python -m src.train_surrogate", language="bash")
        st.stop()

    params = sidebar()
    k_pred, k_sigma = predict(params)
    elevation = (k_pred - 1.0) * PCM
    sigma_pcm = k_sigma * PCM

    left, mid, right = st.columns(3)
    left.metric("$k_{eff}$ 预测", f"{k_pred:.5f}", f"{elevation:+.0f} pcm")
    mid.metric("预测 1σ", f"±{k_sigma:.5f}", f"±{sigma_pcm:.0f} pcm")
    state = "临界" if abs(elevation) <= 2.0 * sigma_pcm else ("超临界" if elevation > 0 else "次临界")
    right.metric("状态", state)

    if abs(elevation) > 2.0 * sigma_pcm:
        st.info(
            f"偏离 1.0 约 {elevation:+.0f} pcm，超出预测不确定度（±{sigma_pcm:.0f} pcm）"
            f"的 2 倍，可判定为{state}。",
            icon="ℹ️",
        )
    else:
        st.success(f"与临界状态不可区分（|{elevation:+.0f}| pcm ≤ 2σ = {2 * sigma_pcm:.0f} pcm）。", icon="✅")

    tab_predict, tab_data, tab_sens, tab_about = st.tabs(["预测", "训练数据", "敏感性", "关于模型"])

    with tab_predict:
        _plot_slice(params, k_pred)

    with tab_data:
        try:
            df = get_dataset()
        except Exception as exc:
            st.error(f"读取数据集失败：{exc}")
        else:
            st.write(f"共 {len(df)} 条样本，来源 `{PATHS.dataset_csv}`。")
            st.dataframe(df, use_container_width=True, height=340)
            st.caption(
                f"物理版本：{df['physics_version'].iloc[0] if 'physics_version' in df else 'unknown'}｜"
                f"$k_{{eff}}$ 范围 {df['k_eff'].min():.5f} – {df['k_eff'].max():.5f}"
            )

    with tab_sens:
        sens_path = PATHS.results_dir / "sensitivity.json"
        if sens_path.exists():
            report = json.loads(sens_path.read_text(encoding="utf-8"))
            st.write("Sobol 全局敏感性指数（来自 `src/sensitivity_analysis.py`）")
            st.dataframe(pd.DataFrame(report["indices"]), use_container_width=True)
            st.caption(
                f"ΣS₁ = {report['sum_S1']:.3f}，ΣS_T = {report['sum_ST']:.3f}；"
                f"代理模型测试 RMSE {report['surrogate_rmse_pcm']:.0f} pcm"
            )
        else:
            st.info("尚无敏感性结果，请先运行 `python -m src.sensitivity_analysis`。")

    with tab_about:
        st.markdown("#### 模型信息")
        info = {
            "核函数": model_bundle.get("kernel", "—"),
            "训练/测试样本数": f"{model_bundle.get('n_train', '—')} / {model_bundle.get('n_test', '—')}",
            "训练时间": model_bundle.get("trained_at", "—"),
            "数据集": model_bundle.get("source_dataset", "—"),
            "物理版本": model_bundle.get("physics_version", "—"),
        }
        st.table(pd.DataFrame({"项目": list(info.keys()), "值": list(info.values())}))
        test = (model_bundle.get("metrics") or {}).get("test", {})
        if test:
            st.markdown("#### 测试集指标")
            st.table(pd.DataFrame({
                "指标": ["RMSE (pcm)", "MAE (pcm)", "最大误差 (pcm)", "R²", "MC 1σ 平均 (pcm)"],
                "值": [
                    f"{test.get('rmse_pcm', float('nan')):.1f}",
                    f"{test.get('mae_pcm', float('nan')):.1f}",
                    f"{test.get('max_abs_err_pcm', float('nan')):.1f}",
                    f"{test.get('r2', float('nan')):.4f}",
                    f"{test.get('mean_mc_sigma_pcm', float('nan')):.1f}",
                ],
            }))
        st.markdown(
            "#### 局限\n"
            "- 代理模型只在训练数据覆盖的范围内可信，页面上的滑条已被限制在该范围内；\n"
            "- $k_{eff}$ 的蒙特卡罗统计误差本身约 100 pcm，模型误差与之相当即已「到噪声地板」；\n"
            "- 敏感性分析是对代理模型做的分解，不是对 OpenMC 直接做的（后者需要数万次输运计算）。"
        )


def _plot_slice(params: dict, k_pred: float) -> None:
    """固定其余参数，扫描硼浓度，展示 k_eff 曲线与不确定度带。"""
    bundle = get_model()
    features = bundle.get("features", list(FEATURE_COLUMNS))
    bounds = get_bounds()
    lo, hi = bounds["boron_ppm"]
    grid = np.linspace(lo, hi, 160)

    rows = []
    for boron in grid:
        row = dict(params)
        row["boron_ppm"] = boron
        rows.append([float(row[f]) for f in features])
    mean, sigma = predict_k(bundle, np.asarray(rows))
    elevation = (mean - 1.0) * PCM

    fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
    ax.plot(grid, elevation, color="tab:blue", lw=2.0, label="GP 预测")
    ax.fill_between(grid, elevation - 2 * sigma * PCM, elevation + 2 * sigma * PCM,
                    color="tab:blue", alpha=0.18, lw=0, label="±2σ")
    ax.axhline(0.0, color="k", ls="--", lw=1.0, label="$k_{eff}=1$")
    ax.axvline(params["boron_ppm"], color="tab:red", ls=":", lw=1.2, label="当前硼浓度")
    ax.plot([params["boron_ppm"]], [(k_pred - 1.0) * PCM], "o", color="tab:red", ms=7)
    ax.set_xlabel("硼浓度 (ppm)")
    ax.set_ylabel("$k_{eff}$ 偏离 1.0 (pcm)")
    ax.set_title("硼浓度扫描（其余参数固定为侧边栏取值）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    crossing = np.where(np.diff(np.sign(elevation)) != 0)[0]
    if crossing.size:
        b_crit = float(np.interp(0.0, elevation[crossing[0]:crossing[0] + 2][::-1],
                                 grid[crossing[0]:crossing[0] + 2][::-1]))
        st.caption(f"在该参数组合下，临界硼浓度约为 **{b_crit:.0f} ppm**（线性插值）。")
    else:
        st.caption("当前参数组合下，扫描区间内没有 $k_{eff}=1$ 的交点（需要更大的硼浓度范围）。")


if __name__ == "__main__":
    render()
