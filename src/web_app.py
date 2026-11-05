"""反应堆物理-热工耦合仪表盘 —— Streamlit + Plotly 暗色科幻主题（单文件版）。

相比 V1.0 的改动：
  1. 去掉 matplotlib / st.line_chart，全部换成 Plotly 辉光图；
  2. 「开始预测」按钮取消，滑块一动卡片立即重算重绘；
  3. 耦合演示不再用 `T_new = 600 + 1e8*3e-6` 这种假公式，
     而是先算 k_eff，再由 k_eff 决定功率，再解一维导热，做真正的 Picard 迭代。

数据来源：
  * 在 OpenMC-Reactor-Physics 仓库内运行时，自动用 src.surrogate 里训练好的 GP 代理模型；
  * 脱离仓库（例如拷到别处单独跑）时退回内置占位模型，界面右下角会标明用的是哪个。

运行：streamlit run web_app.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# 重构后脚本位于 src/：把脚本目录和项目根目录都加入 sys.path，
# 这样无论从项目根目录、src/ 还是其他 cwd 启动，都能导入
# 同目录的 anderson_accelerator 以及项目根的 src 包。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for _path in (SRC_DIR, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from anderson_accelerator import AndersonAccelerator

# ============================================================ 1. 主题

NEON = "#00FF9C"   # 荧光绿：Anderson 主数据曲线
PICARD_GRAY = "#9AA6B8"  # 灰色：Picard 基线与散点
CYAN = "#22D3EE"   # 科幻蓝：辅助线 / 标注
VIOLET = "#A78BFA"
PINK = "#FF3D9A"   # 告警
TRANSIENT_POWER = "#FF5A1F"   # 弹棒事故：荧光橙红，归一化功率
TRANSIENT_FUEL = "#4DA3FF"    # 弹棒事故：蓝色，燃料温度
TRANSIENT_COOLANT = "#00E5FF" # 弹棒事故：青色，冷却剂温度
INK = "#C9D6E3"
DIM = "#6B7A8F"
MONO = '"JetBrains Mono", Consolas, "Microsoft YaHei", monospace'

CSS = """
<style>
  .stApp {
    background:
      radial-gradient(1200px 620px at 12% -12%, rgba(34,211,238,.12), transparent 62%),
      radial-gradient(900px 520px at 92% -6%, rgba(0,255,156,.09), transparent 58%),
      linear-gradient(180deg, #05070E 0%, #070B16 100%);
  }
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070B14 0%, #050810 100%);
    border-right: 1px solid rgba(34,211,238,.14);
  }
  .kpi {
    position: relative; padding: 16px 18px 14px; border-radius: 16px; height: 100%;
    background: linear-gradient(160deg, rgba(12,20,34,.94), rgba(8,12,22,.94));
    border: 1px solid rgba(34,211,238,.20);
    box-shadow: 0 14px 38px -22px rgba(34,211,238,.65),
                inset 0 0 48px -26px rgba(0,255,156,.55);
    overflow: hidden;
  }
  .kpi::before {
    content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent, #22D3EE), transparent);
  }
  .kpi-label { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: #6B7A8F; }
  .kpi-value {
    font: 600 1.95rem/1.15 "JetBrains Mono", Consolas, monospace;
    color: #EAF6FF; text-shadow: 0 0 20px rgba(34,211,238,.55); margin-top: 4px;
  }
  .kpi-sub { font-size: .76rem; color: #8FA3B8; margin-top: 6px; }
  .chip {
    display: inline-block; margin: 6px 8px 0 0; padding: 5px 11px; border-radius: 999px;
    font-size: .76rem; color: #BFD3E6;
    background: rgba(34,211,238,.07); border: 1px solid rgba(34,211,238,.18);
  }
  .chip b { color: #EAF6FF; font-family: "JetBrains Mono", Consolas, monospace; }
  .hint { font-size: .76rem; color: #6B7A8F; line-height: 1.5; }
  /* st.metric 与上方自定义霓虹卡片保持同一套视觉语言 */
  div[data-testid="stMetric"] {
    position: relative; overflow: hidden;
    padding: 14px 16px 12px; border-radius: 16px;
    background: linear-gradient(160deg, rgba(12,20,34,.94), rgba(8,12,22,.94));
    border: 1px solid rgba(34,211,238,.20);
    box-shadow: 0 14px 38px -24px rgba(34,211,238,.55),
                inset 0 0 48px -30px rgba(0,255,156,.50);
  }
  div[data-testid="stMetric"]::before {
    content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
    background: linear-gradient(90deg, transparent, #22D3EE, transparent);
    opacity: .7;
  }
  div[data-testid="stMetricLabel"] p {
    font-size: .74rem; letter-spacing: .10em; text-transform: uppercase; color: #8FA3B8;
  }
  div[data-testid="stMetricValue"] {
    color: #EAF6FF; font-family: "JetBrains Mono", Consolas, monospace;
    text-shadow: 0 0 18px rgba(34,211,238,.35);
  }
</style>
"""

# 离线结果统一放在项目根的 data/；用 __file__ 绝对定位，不依赖当前工作目录。
DATA_DIR = PROJECT_ROOT / "data"
HISTORY_PATH = DATA_DIR / "iteration_history.json"
TRANSIENT_PATH = DATA_DIR / "transient_precise_results.json"

# ============================================================ 2. 一维导热（自包含）

PITCH_CM = 1.26
R_PELLET = 0.4096
R_CLAD_IN = 0.4180
R_CLAD_OUT = 0.4750
H_GAP = 0.7          # W/(cm²·K)，气隙换热系数
H_FILM = 3.0         # W/(cm²·K)，包壳-冷却剂对流换热系数
K_CLAD = 0.165       # W/(cm·K)，Zr-4 @600 K
T_MELT_UO2 = 3120.0  # K，UO₂ 熔点近似；超过后 k(T) 外推会趋近 0，必须截断


def get_uo2_conductivity(T: float) -> float:
    """UO₂ 燃料热导率 k(T)，单位 W/(m·K)。

    经典经验关联式（Fink 形式、仅取声子项）：

        k(T) = 1 / (0.0375 + 2.165e-4 T + 4.715e-7 T²)

    其中 T 为绝对温度（K）。该式适用于固态 UO₂；温度越高 k 越小，
    因此在热工求解中必须用当前燃料温度实时更新，不能再用常数。
    """
    T = float(T)
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError(f"UO₂ 温度必须是正有限值，收到 {T!r}")
    return 1.0 / (0.0375 + 2.165e-4 * T + 4.715e-7 * T * T)


@dataclass(frozen=True)
class Field:
    power: float
    t_center: float
    t_surface: float
    t_avg: float
    t_clad_in: float
    t_clad_out: float
    d_fuel: float
    d_gap: float
    d_clad: float
    d_film: float
    k_fuel: float   # 本次温度场使用的 UO₂ 热导率，W/(m·K)
    melted: bool    # 芯块中心是否已按熔点截断


def temperature_field(power_w_cm: float, mod_temp: float,
                      fuel_temp: float | None = None) -> Field:
    """稳态一维导热：冷却剂 → 液膜 → 包壳 → 气隙 → 芯块，串联热阻依次叠加。

    fuel_temp 是当前 Picard 迭代的燃料温度估计（K）；k(T) 用它实时计算，
    不再使用常数热导率。未传入时用冷却剂温度 + 40 K 作为冷态起步值。
    """
    if fuel_temp is None:
        fuel_temp = mod_temp + 40.0
    # k(T) 只在固态 UO₂ 的物理区间内取值；上限截断可避免外推时
    # 分母溢出、k 变成 0 导致的 ZeroDivisionError。
    fuel_temp_eval = min(max(float(fuel_temp), 300.0), T_MELT_UO2)
    k_fuel = get_uo2_conductivity(fuel_temp_eval)        # W/(m·K)
    k_fuel_cm = k_fuel / 100.0                           # W/(cm·K)

    d_film = power_w_cm / (2.0 * math.pi * R_CLAD_OUT * H_FILM)
    d_clad = power_w_cm / (2.0 * math.pi * K_CLAD) * math.log(R_CLAD_OUT / R_CLAD_IN)
    k_gap = H_GAP * (R_CLAD_IN - R_PELLET)              # 薄气隙等效导热系数
    d_gap = power_w_cm / (2.0 * math.pi * k_gap) * math.log(R_CLAD_IN / R_PELLET)

    t_clad_out = mod_temp + d_film
    t_clad_in = t_clad_out + d_clad
    t_surface = t_clad_in + d_gap

    # 用当前燃料温度求 k(T)，再算均匀体积释热下的芯块温降。
    d_fuel_raw = power_w_cm / (4.0 * math.pi * k_fuel_cm)
    t_center_raw = t_surface + d_fuel_raw
    melted = t_center_raw > T_MELT_UO2
    if melted:
        # k(T) 在高温端按 1/T² 衰减，外推到熔点以上会数值爆炸；
        # 这里把芯块中心截断在熔点，并明确标记，便于 UI 提示模型失效。
        t_center = T_MELT_UO2
        d_fuel = max(T_MELT_UO2 - t_surface, 0.0)
    else:
        t_center = t_center_raw
        d_fuel = d_fuel_raw

    return Field(
        power=power_w_cm, t_center=t_center, t_surface=t_surface,
        t_avg=t_surface + 2.0 / 3.0 * d_fuel,
        t_clad_in=t_clad_in, t_clad_out=t_clad_out,
        d_fuel=d_fuel, d_gap=d_gap, d_clad=d_clad, d_film=d_film,
        k_fuel=k_fuel, melted=melted)


# ============================================================ 3. 代理模型（有真模型就用真的）

# 15.5 MPa 液态水密度表（密度必须升序，np.interp 才能正确插值）
_DENSITY_TABLE = np.array([0.6282, 0.6611, 0.6883, 0.7119,
                           0.7329, 0.7697, 0.8428, 0.8784])
_TEMPERATURE_TABLE = np.array([610.0, 600.0, 590.0, 580.0, 570.0, 550.0, 500.0, 470.0])


def temperature_from_density(density: float) -> float:
    return float(np.interp(density, _DENSITY_TABLE, _TEMPERATURE_TABLE))


class RepoSurrogate:
    """包装仓库里的 GP：5 维特征 (enrichment, boron_ppm, clad_r, fuel_temp, mod_temp)。"""

    def __init__(self, bundle: dict[str, Any], label: str) -> None:
        from src.surrogate import predict_k  # 局部导入，避免脱离仓库时报错

        self._bundle = bundle
        self._predict_k = predict_k
        self.features = list(bundle.get("features", []))
        self.label = label
        self.has_sigma = True

    def k_eff(self, enrichment: float, boron_ppm: float, density: float,
              fuel_temp: float, ) -> tuple[float, float | None]:
        values = {
            "enrichment": enrichment,
            "boron_ppm": boron_ppm,
            "clad_r": R_CLAD_OUT,
            "fuel_temp": fuel_temp,
            "mod_temp": temperature_from_density(density),
        }
        vector = np.array([[values[f] for f in self.features]], dtype=float)
        mean, sigma = self._predict_k(self._bundle, vector)
        return float(mean[0]), float(sigma[0])

    def fuel_temp_coefficient(self, enrichment: float, boron_ppm: float,
                              density: float, fuel_temp: float) -> float:
        hot, _ = self.k_eff(enrichment, boron_ppm, density, fuel_temp + 50.0)
        cold, _ = self.k_eff(enrichment, boron_ppm, density, fuel_temp - 50.0)
        return (hot - cold) / 100.0 * 1e5  # pcm/K


class PlaceholderSurrogate:
    """脱离仓库时的占位模型：和 V1.0 一样的随机森林 + 合成数据，仅用于调界面。"""

    label = "占位模型 · 随机森林 + 合成数据（仅界面演示，不可用于结论）"
    has_sigma = False

    def __init__(self) -> None:
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(42)
        n = 300
        enrichment = rng.uniform(1.5, 5.0, n)
        density = rng.uniform(0.62, 0.84, n)
        k = 0.93 + 0.045 * enrichment + 0.30 * (density - 0.60) + rng.normal(0, 0.01, n)
        self._model = RandomForestRegressor(n_estimators=200, random_state=42)
        self._model.fit(np.column_stack([enrichment, density]), k)

    def k_eff(self, enrichment: float, boron_ppm: float, density: float,
              fuel_temp: float) -> tuple[float, float | None]:
        return float(self._model.predict([[enrichment, density]])[0]), None

    def fuel_temp_coefficient(self, *_: Any) -> float:
        return -3.0  # 占位：没有温度特征，直接给个典型值


@st.cache_resource(show_spinner=False)
def get_surrogate():
    try:
        from src.config import PATHS
        from src.surrogate import load_model, train
    except Exception:  # noqa: BLE001 - 不在仓库里，退回占位模型
        return PlaceholderSurrogate()

    try:
        if PATHS.model_joblib.exists():
            return RepoSurrogate(load_model(PATHS.model_joblib), "GP 代理模型 · results/surrogate.joblib")
        return RepoSurrogate(train(), "GP 代理模型 · 现场训练 data/processed")
    except Exception:  # noqa: BLE001
        return PlaceholderSurrogate()


# ============================================================ 4. Picard + Anderson 耦合迭代

ANDERSON_M = 3     # Anderson 记忆深度：最近 m+1 组 (T_in, T_out) 构造残差矩阵
TOLERANCE = 0.05   # K，固定点残差 |T_out - T_in| 的收敛判据
MAX_ITER = 60


@dataclass
class Coupled:
    history: list[float]
    k_eff: float
    sigma: float | None
    field: Field
    iterations: int
    converged: bool
    fuel_temp_coeff: float


def coupled_iteration(enrichment: float, boron_ppm: float, density: float,
                      q_rated: float, surrogate: Any) -> Coupled:
    """k_eff ↔ 功率 ↔ 温度 的不动点迭代，用 Anderson 加速替代固定松弛。

    把每一步看作不动点映射 G(T_in) = T_out：
      * T_in  = 当前 Picard 输入温度；
      * T_out = temperature_field(power(T_in), mod_temp, T_in).t_avg。
    Anderson 加速器在前 m 步仍返回 G(T_in)（普通 Picard），
    第 m+1 步起用最近 m 个残差差分为基做最小二乘外推，
    因此不需要 RELAX 因子，也能在强反馈工况下减少迭代步数。
    """
    mod_temp = temperature_from_density(density)
    fuel_temp = mod_temp + 40.0     # 冷态起步，曲线才有可看的收敛过程
    history: list[float] = []
    field = temperature_field(q_rated, mod_temp, fuel_temp)
    aae = AndersonAccelerator(m=ANDERSON_M)
    converged = False

    for step in range(1, MAX_ITER + 1):
        k_eff, sigma = surrogate.k_eff(enrichment, boron_ppm, density, fuel_temp)
        # 反应性越大，实际功率越偏离额定值：1.0 表示 k=1 时正好额定功率
        power = q_rated * min(max(1.0 + (k_eff - 1.0), 0.25), 2.4)
        # 用当前燃料温度估计实时更新 UO₂ 的 k(T)，再解温度场。
        field = temperature_field(power, mod_temp, fuel_temp)

        t_calc = field.t_avg
        # 关键：x_k = 当前输入温度，f_k = 本次温度场输出；返回的是下一步输入。
        t_next = float(aae.accelerate(
            np.array([fuel_temp]), np.array([t_calc]))[0])
        # Anderson 在极端工况可能外推出物理区间外的温度；先夹到
        # [300 K, UO₂ 熔点] 再进入下一轮，避免 k(T) 公式在 0 K 附近除零。
        t_next = min(max(t_next, 300.0), T_MELT_UO2)
        history.append(t_next)

        # 用真正的固定点残差判断收敛，而不是“相邻两次迭代步长”。
        # 这样 Anderson 的大步外推不会因为步长看起来大而被误判为未收敛。
        if step > 1 and abs(t_calc - fuel_temp) < TOLERANCE:
            converged = True
            break

        fuel_temp = t_next

    # 循环结束后用最终输入温度重算一次，保证返回的 field/k_eff 与
    # history[-1] 对应（Anderson 的最后一步输入不一定等于上一次的 T_out）。
    final_temp = history[-1] if history else fuel_temp
    k_eff, sigma = surrogate.k_eff(enrichment, boron_ppm, density, final_temp)
    power = q_rated * min(max(1.0 + (k_eff - 1.0), 0.25), 2.4)
    field = temperature_field(power, mod_temp, final_temp)

    return Coupled(
        history=history, k_eff=k_eff, sigma=sigma, field=field,
        iterations=len(history), converged=converged,
        fuel_temp_coeff=surrogate.fuel_temp_coefficient(
            enrichment, boron_ppm, density, final_temp))


# ============================================================ 5. 辉光图元


def _glow(fig: go.Figure, x: Any, y: Any, color: str, core: float = 2.4,
          layers: tuple[tuple[float, float], ...] = ((18.0, 0.045), (11.0, 0.075),
                                                    (6.5, 0.13), (3.4, 0.22)),
          spline: bool = True, dash: str | None = None) -> None:
    """几条半透明粗线叠出外发光，再画一条细实线当灯丝。"""
    line: dict[str, Any] = dict(color=color, shape="spline", smoothing=0.55) if spline \
        else dict(color=color)
    if dash:
        line["dash"] = dash
    for width, opacity in layers:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", hoverinfo="skip",
                                 showlegend=False, line={**line, "width": width},
                                 opacity=opacity))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", showlegend=False,
                             line={**line, "width": core}))


def _dark_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(6,11,20,0.55)",
        font=dict(family=MONO, color=INK, size=12),
        title=dict(text=title, x=0.008, y=0.97, font=dict(size=13, color="#9FB4C9")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(5,10,18,.96)", bordercolor="rgba(0,255,156,.45)",
                        font=dict(color="#EAF6FF", family=MONO)),
        showlegend=False,
        xaxis=dict(gridcolor="rgba(34,211,238,0.07)", zeroline=False,
                   showline=True, linecolor="rgba(34,211,238,0.20)",
                   tickfont=dict(color=DIM)),
        yaxis=dict(gridcolor="rgba(34,211,238,0.07)", zeroline=False,
                   tickfont=dict(color=DIM)),
    )
    return fig


def convergence_figure(history: list[float], tolerance: float, converged: bool,
                       mod_temp: float, y_bounds: tuple[float, float] | None = None,
                       partial: bool = False) -> go.Figure:
    x = list(range(1, len(history) + 1))
    y = [float(v) for v in history]
    final = y[-1]

    fig = go.Figure()
    _glow(fig, x, y, NEON)
    fig.add_trace(go.Scatter(x=[x[-1]], y=[final], mode="markers", hoverinfo="skip",
                             showlegend=False, marker=dict(size=22, color=NEON, opacity=0.18)))
    fig.add_trace(go.Scatter(
        x=[x[-1]], y=[final], mode="markers", showlegend=False,
        marker=dict(size=11, color=NEON, line=dict(color="#04120C", width=2.5)),
        hovertemplate=f"最新点 {final:.2f} K<extra></extra>"))

    lo, hi = y_bounds or (min(y) - 30.0, max(y) + 30.0)
    fig.add_hline(y=final, line=dict(color=CYAN, width=1, dash="dot"), opacity=0.5)

    if partial:
        subtitle = "迭代进行中…"
    else:
        state = "已收敛" if converged else "未达判据"
        fig.add_annotation(
            x=x[-1], y=final, text=f"  {final:.2f} K · {len(y)} 次迭代  ",
            showarrow=False, xanchor="right", yanchor="bottom",
            font=dict(color=CYAN, size=12, family=MONO),
            bgcolor="rgba(5,7,14,.78)", bordercolor="rgba(34,211,238,.35)",
            borderwidth=1, borderpad=6)
        subtitle = f"慢化剂 {mod_temp:.0f} K · 固定点残差 |T_out−T_in| &lt; {tolerance:g} K · {state}"

    _dark_layout(fig, "ANDERSON 加速 · 燃料平均温度收敛轨迹　"
                      f"<span style='color:#6B7A8F'>{subtitle}</span>")
    fig.update_xaxes(title="迭代次数", dtick=1 if len(x) <= 14 else None,
                     range=[0.4, len(x) + 0.6])
    fig.update_yaxes(title="燃料平均温度 / K", range=[lo, hi])
    return fig


def scan_figure(surrogate: Any, enrichment: float, boron_ppm: float, density: float,
                fuel_temp: float, k_eff: float) -> go.Figure:
    grid = np.linspace(1.5, 5.0, 41)
    curve = [surrogate.k_eff(float(e), boron_ppm, density, fuel_temp)[0] for e in grid]

    fig = go.Figure()
    _glow(fig, list(grid), curve, NEON, core=2.2)
    fig.add_hline(y=1.0, line=dict(color=PINK, width=1.4, dash="dash"), opacity=0.7)
    fig.add_annotation(x=1.5, y=1.0, text=" 临界 k_eff = 1.0 ", yshift=12, xanchor="left",
                       showarrow=False, font=dict(color=PINK, size=11, family=MONO))
    fig.add_trace(go.Scatter(
        x=[enrichment], y=[k_eff], mode="markers", showlegend=False,
        marker=dict(size=16, color=CYAN, opacity=0.25), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[enrichment], y=[k_eff], mode="markers", showlegend=False,
        marker=dict(size=9, color="#EAF6FF", line=dict(color=CYAN, width=2)),
        hovertemplate=f"当前设计 ({enrichment:.2f} %, {k_eff:.5f})<extra></extra>"))

    _dark_layout(fig, "富集度扫描　<span style='color:#6B7A8F'>"
                      f"硼 {boron_ppm:.0f} ppm · 密度 {density:.2f} g/cm³</span>", height=360)
    fig.update_xaxes(title="U-235 富集度 / %")
    fig.update_yaxes(title="k_eff")
    return fig


# ============================================================ 6. 组件


def kpi(label: str, value: str, sub: str, accent: str) -> None:
    st.markdown(
        f'<div class="kpi" style="--accent:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True)


def plot(fig: go.Figure, target: Any = None) -> None:
    """Streamlit 1.49+ 用 width="stretch"，老版本只认 use_container_width。"""
    import inspect

    kwargs = ({"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters
              else {"use_container_width": True})
    (target or st).plotly_chart(fig, config={"displayModeBar": False, **kwargs})


# ============================================================ 7. 实测迭代对比（离线 JSON）
#
# 这里只读 coupled_pin.py 生成的 iteration_history.json，
# 绝不在 Streamlit 里调用 OpenMC。


def _load_iteration_history(path: Path = HISTORY_PATH) -> dict[str, Any]:
    """读取 JSON；文件不存在 / 编码错误 / JSON 语法错误都交给调用方提示。"""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("JSON 顶层必须是对象")
    return payload


def _extract_series(block: Any, name: str) -> tuple[list[int], list[float]]:
    """取出 iterations / temperatures 两条等长数组，并转成 Python 数值。"""
    if not isinstance(block, dict):
        raise ValueError(f"{name} 字段缺失或不是 JSON 对象")
    xs, ys = block.get("iterations"), block.get("temperatures")
    if not isinstance(xs, list) or not isinstance(ys, list):
        raise ValueError(f"{name} 缺少 iterations / temperatures 数组")
    if not xs or len(xs) != len(ys):
        raise ValueError(f"{name} 的 iterations / temperatures 为空或长度不一致")
    try:
        xi = [int(x) for x in xs]
        yf = [float(y) for y in ys]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 含非数值元素：{exc}") from exc
    if not all(math.isfinite(v) for v in yf):
        raise ValueError(f"{name}.temperatures 含 NaN / Infinity")
    return xi, yf


def _iteration_count(block: dict[str, Any], xs: list[int], name: str) -> int:
    """优先用 JSON 里的 n_iterations；缺失时退回数组长度减一。"""
    value = block.get("n_iterations", len(xs) - 1)
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}.n_iterations 不是整数") from exc
    if count < 0:
        raise ValueError(f"{name}.n_iterations 不能为负数")
    if count != len(xs) - 1:
        raise ValueError(
            f"{name}.n_iterations={count} 与 iterations 数组长度 {len(xs)} 不一致"
        )
    return count


def history_comparison_figure(
    xa: list[int], ya: list[float],
    xp: list[int] | None = None, yp: list[float] | None = None,
) -> go.Figure:
    """Anderson 实线 + Picard 虚线，两条曲线都在每个数据点上画散点。"""
    fig = go.Figure()

    # Anderson 的霓虹外发光：两层半透明粗线，最后再叠主线和散点。
    fig.add_trace(go.Scatter(
        x=xa, y=ya, mode="lines", hoverinfo="skip", showlegend=False,
        line=dict(color=NEON, width=10), opacity=0.07))
    fig.add_trace(go.Scatter(
        x=xa, y=ya, mode="lines", hoverinfo="skip", showlegend=False,
        line=dict(color=NEON, width=5), opacity=0.13))

    if xp is not None and yp is not None:
        fig.add_trace(go.Scatter(
            x=xp, y=yp, mode="lines+markers", name="Picard 基线",
            line=dict(color=PICARD_GRAY, width=2.0, dash="dash"),
            marker=dict(size=8, color=PICARD_GRAY,
                        line=dict(color="#0B0F17", width=1)),
            cliponaxis=False,
            hovertemplate="Picard · 第 %{x} 步<br>燃料温度 %{y:.2f} K<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=xa, y=ya, mode="lines+markers", name="Anderson 加速",
        line=dict(color=NEON, width=3.0),
        marker=dict(size=9, color=NEON,
                    line=dict(color="#04120C", width=1.5)),
        cliponaxis=False,
        hovertemplate="Anderson · 第 %{x} 步<br>燃料温度 %{y:.2f} K<extra></extra>"))

    all_x = xa + (xp or [])
    all_y = ya + (yp or [])
    pad = max(4.0, (max(all_y) - min(all_y)) * 0.08)

    _dark_layout(
        fig,
        "燃料温度 · 迭代收敛轨迹　<span style='color:#6B7A8F'>"
        "Anderson 荧光绿实线 / Picard 灰色虚线</span>",
        height=440,
    )
    fig.update_layout(
        showlegend=True,
        margin=dict(l=10, r=10, t=54, b=42),
        legend=dict(
            orientation="h", x=0.99, y=0.02, xanchor="right", yanchor="bottom",
            bgcolor="rgba(5,10,18,.55)", bordercolor="rgba(34,211,238,.18)",
            borderwidth=1, font=dict(color=INK, size=11)),
    )
    fig.update_xaxes(
        title="迭代次数",
        dtick=1 if max(all_x) <= 12 else 2,
        range=[min(all_x) - 0.35, max(all_x) + 0.35],
    )
    fig.update_yaxes(
        title="燃料温度 / K",
        range=[min(all_y) - pad, max(all_y) + pad],
    )
    return fig


def render_history_comparison() -> None:
    """页面顶部的离线对比区：3 个 st.metric + 1 张 Plotly 对比图。"""
    st.markdown("### ⚡ Anderson vs Picard · 实测迭代对比")
    st.caption("读取 `iteration_history.json` 的离线结果；本页不会实时调用 OpenMC。")

    try:
        payload = _load_iteration_history()
    except FileNotFoundError:
        st.warning(f"未找到 `{HISTORY_PATH}`。请先运行下面的命令生成它：")
        st.code(f"cd {PROJECT_ROOT}\npython src/coupled_pin.py", language="bash")
        return
    except (OSError, ValueError) as exc:
        st.error(f"读取 `{HISTORY_PATH}` 失败：{exc}")
        return

    try:
        anderson_block = payload.get("anderson")
        xa, ya = _extract_series(anderson_block, "anderson")
        n_a = _iteration_count(anderson_block, xa, "anderson")

        picard_block = payload.get("picard")
        if picard_block:
            xp, yp = _extract_series(picard_block, "picard")
            n_p: int | None = _iteration_count(picard_block, xp, "picard")
        else:
            xp, yp, n_p = None, None, None
    except ValueError as exc:
        st.error(f"`{HISTORY_PATH.name}` 结构不完整：{exc}")
        return

    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        comparison = {}
    speedup = comparison.get("speedup_ratio")
    if n_p is None or not isinstance(speedup, (int, float)) \
            or not math.isfinite(float(speedup)) or speedup <= 0:
        speedup = (n_p / n_a) if (n_p is not None and n_a > 0) else None

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.metric("Anderson 迭代步数", f"{n_a}",
                  help="iteration_history.json → anderson.n_iterations")
    with c2:
        st.metric("Picard 迭代步数", f"{n_p}" if n_p is not None else "—",
                  help="iteration_history.json → picard.n_iterations")
    with c3:
        st.metric("计算加速比", f"{speedup:.2f}x" if speedup else "—",
                  help="Picard 步数 ÷ Anderson 步数，越大越好")

    if n_p is None:
        st.info("这份 JSON 只包含 Anderson 结果（Picard 基线未运行）。")

    plot(history_comparison_figure(xa, ya, xp, yp))

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    st.caption(
        f"数据源：`{HISTORY_PATH}` · 生成于 {meta.get('generated_at', '未知')}"
        " · 第 0 点为初始温度"
    )


# ============================================================ 8. 瞬态事故分析（离线 JSON）
#
# 只读 transient_precise.py 生成的 JSON，绝不在 Streamlit 里调用 OpenMC。


@st.cache_data(show_spinner=False)
def _read_transient_json(path_str: str, mtime_ns: int) -> dict[str, Any]:
    """带 mtime 的缓存读取：JSON 重新生成后 mtime 变化，缓存自动失效。"""
    del mtime_ns  # 只参与缓存键，函数体不需要
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("JSON 顶层必须是对象")

    series: dict[str, list[float]] = {}
    for key in ("t", "n", "Tf", "Tm"):
        values = payload.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"缺少非空数组 `{key}`")
        try:
            series[key] = [float(v) for v in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"`{key}` 含非数值元素：{exc}") from exc
        if not all(math.isfinite(v) for v in series[key]):
            raise ValueError(f"`{key}` 含 NaN / Infinity")

    lengths = {len(values) for values in series.values()}
    if len(lengths) != 1:
        raise ValueError(f"t / n / Tf / Tm 长度不一致：{sorted(lengths)}")

    payload.update(series)
    return payload


def load_transient_results(path: Path = TRANSIENT_PATH) -> dict[str, Any]:
    """读取 JSON；把文件 mtime 一起交给缓存函数，重新生成后不会读到旧结果。"""
    stat = path.stat()
    return _read_transient_json(str(path), stat.st_mtime_ns)


def transient_figure(t: Any, n: Any, tf: Any, tm: Any) -> go.Figure:
    """双 Y 轴：左轴对数功率，右轴燃料/冷却剂温度。"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 功率主曲线：橙红荧光；再叠一层半透明粗线做外发光。
    fig.add_trace(go.Scatter(
        x=t, y=n, mode="lines", hoverinfo="skip", showlegend=False,
        line=dict(color=TRANSIENT_POWER, width=8), opacity=0.10,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=t, y=n, mode="lines", name="归一化功率 n",
        line=dict(color=TRANSIENT_POWER, width=3.0),
        hovertemplate="t=%{x:.3f} s<br>n=%{y:.3f}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=t, y=tf, mode="lines", name="燃料温度 Tf",
        line=dict(color=TRANSIENT_FUEL, width=2.2),
        hovertemplate="t=%{x:.3f} s<br>Tf=%{y:.2f} K<extra></extra>",
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=t, y=tm, mode="lines", name="冷却剂温度 Tm",
        line=dict(color=TRANSIENT_COOLANT, width=2.2),
        hovertemplate="t=%{x:.3f} s<br>Tm=%{y:.2f} K<extra></extra>",
    ), secondary_y=True)

    t_lo = min(float(np.min(tf)), float(np.min(tm)))
    t_hi = max(float(np.max(tf)), float(np.max(tm)))
    pad = max(4.0, (t_hi - t_lo) * 0.08)

    _dark_layout(
        fig,
        "弹棒事故瞬态　<span style='color:#6B7A8F'>"
        "0–10 s · 左轴对数功率 / 右轴温度</span>",
        height=520,
    )
    fig.update_xaxes(title_text="时间 / s", range=[0.0, 10.0], dtick=1.0)
    fig.update_yaxes(
        title_text="归一化功率 n（对数）",
        type="log",
        exponentformat="power",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="温度 / K",
        type="linear",
        range=[t_lo - pad, t_hi + pad],
        secondary_y=True,
    )
    fig.update_layout(
        showlegend=True,
        margin=dict(l=10, r=10, t=58, b=52),
        legend=dict(
            orientation="h", x=0.99, y=0.98, xanchor="right", yanchor="top",
            bgcolor="rgba(5,10,18,.55)", bordercolor="rgba(34,211,238,.18)",
            borderwidth=1, font=dict(color=INK, size=11)),
    )
    return fig


def render_transient_tab() -> None:
    st.markdown("### ⚠️ 瞬态安全分析：弹棒事故")
    st.caption("读取 `transient_precise_results.json` 的离线结果；本页不会实时调用 OpenMC。")

    try:
        payload = load_transient_results()
    except FileNotFoundError:
        st.warning(f"未找到 `{TRANSIENT_PATH}`。请先运行：")
        st.code(f"cd {PROJECT_ROOT}\npython src/transient_precise.py", language="bash")
        return
    except (OSError, ValueError) as exc:
        st.error(f"读取 `{TRANSIENT_PATH}` 失败：{exc}")
        return

    t = np.asarray(payload["t"], dtype=float)
    n = np.asarray(payload["n"], dtype=float)
    tf = np.asarray(payload["Tf"], dtype=float)
    tm = np.asarray(payload["Tm"], dtype=float)

    window = t <= 10.0 + 1e-12
    if not bool(np.any(window)):
        st.error("JSON 中没有 0–10 s 范围内的数据点。")
        return
    t_win, n_win = t[window], n[window]
    tf_win, tm_win = tf[window], tm[window]

    i_peak = int(np.argmax(n_win))
    peak_n = float(n_win[i_peak])
    peak_tf = float(np.max(tf_win))

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    neutronics = meta.get("neutronics")
    if not isinstance(neutronics, dict):
        neutronics = {}
    initial = meta.get("initial_conditions")
    if not isinstance(initial, dict):
        initial = {}
    alpha_f = float(neutronics.get("alpha_f_per_K", -3e-5))
    tf0 = float(initial.get("Tf0_K", 600.0))
    doppler_pcm = alpha_f * (peak_tf - tf0) * 1e5

    st.warning(
        "警告：当前模拟为 +1500 pcm 弹棒事故（瞬发超临界）。"
        "多普勒效应在 0.11s 内介入，成功抑制了功率尖峰。"
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.metric("峰值功率 n_max", f"{peak_n:.2f} ×",
                  help="0–10 s 内归一化功率的最大值")
    with c2:
        st.metric("燃料峰值温度", f"{peak_tf:.2f} K",
                  help="0–10 s 内燃料温度 Tf 的最大值")
    with c3:
        st.metric("反应性反馈", f"{doppler_pcm:.0f} pcm",
                  help="峰值时刻多普勒反馈：alpha_f × (Tf_peak − Tf0)")

    plot(transient_figure(t_win, n_win, tf_win, tm_win))
    st.caption(
        f"数据源：`{TRANSIENT_PATH.name}` · 0–10 s 共 {t_win.size} 点 · "
        f"功率峰值 {peak_n:.2f}× @ {t_win[i_peak]:.3f} s"
    )


# ============================================================ 9. 主界面


def render_steady_state_dashboard() -> None:
    st.markdown("### 🧪 交互式设计空间 · 实时代理模型")

    surrogate = get_surrogate()

    with st.sidebar:
        st.markdown("### ⚙️ 设计参数")
        enrichment = st.slider("U-235 富集度 / wt%", 1.5, 5.0, 3.10, 0.05)
        boron = st.slider("硼浓度 / ppm", 0.0, 2500.0, 1300.0, 10.0)
        density = st.slider("慢化剂密度 / g·cm⁻³", 0.62, 0.84, 0.74, 0.005)
        st.caption(f"≈ {temperature_from_density(density):.0f} K（15.5 MPa 液态水）")

        st.markdown("### 🔥 热工边界")
        power = st.slider("额定线功率 q′ / W·cm⁻¹", 100.0, 400.0, 220.0, 5.0)

        st.markdown("### ✨ 显示")
        animate = st.checkbox("播放收敛动画", value=False)

        st.markdown(f'<div class="hint">模型：{surrogate.label}</div>',
                    unsafe_allow_html=True)

    started = time.perf_counter()
    result = coupled_iteration(enrichment, boron, density, power, surrogate)
    elapsed = (time.perf_counter() - started) * 1e3
    field = result.field
    mod_temp = temperature_from_density(density)

    if abs(result.k_eff - 1.0) * 1e5 < 500:
        state, color = "临界", NEON
    elif result.k_eff > 1.0:
        state, color = "超临界", CYAN if abs(result.k_eff - 1.0) * 1e5 < 5000 else PINK
    else:
        state, color = "次临界", PINK

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        sub = f"{state} · " + (f"1σ ±{result.sigma * 1e5:.0f} pcm"
                               if result.sigma is not None else "无不确定度")
        kpi("k_eff", f"{result.k_eff:.5f}", sub, color)
    with c2:
        kpi("收敛温度 · 燃料平均", f"{result.history[-1]:.1f} K",
            f"中心 {field.t_center:.1f} K · 表面 {field.t_surface:.1f} K", NEON)
    with c3:
        kpi("本次交互 Anderson 迭代", f"{result.iterations}",
            "已收敛（固定点残差）" if result.converged else "达到迭代上限（可能振荡）",
            CYAN if result.converged else PINK)
    with c4:
        kpi("燃料温度系数", f"{result.fuel_temp_coeff:+.2f}", "pcm / K（±50 K 差分）", VIOLET)

    if field.melted:
        st.warning(
            f"⚠️ 芯块中心温度已超过 UO₂ 熔点 {T_MELT_UO2:.0f} K，"
            "温度场按熔点截断；当前线功率已超出该简化一维导热模型的适用范围。")

    st.write("")

    y_bounds = (min(result.history) - 30.0, max(result.history) + 30.0)
    hero = st.empty()
    if animate:
        for i in range(1, len(result.history) + 1):
            plot(convergence_figure(result.history[:i], TOLERANCE, result.converged,
                                    mod_temp, y_bounds, partial=True), target=hero)
            time.sleep(0.15)
    plot(convergence_figure(result.history, TOLERANCE, result.converged, mod_temp, y_bounds),
         target=hero)

    total = field.d_fuel + field.d_gap + field.d_clad + field.d_film
    chips = [("芯块温降", field.d_fuel), ("气隙温降", field.d_gap),
             ("包壳温降", field.d_clad), ("液膜温降", field.d_film)]
    st.markdown(
        "".join(f'<span class="chip">{name} <b>{value:.1f} K</b>'
                f'（{value / total * 100:.1f}%）</span>' for name, value in chips)
        + f'<span class="chip">实际线功率 <b>{field.power:.1f} W/cm</b></span>'
        + f'<span class="chip">UO₂ k(T_f) <b>{field.k_fuel:.3f} W/(m·K)</b></span>',
        unsafe_allow_html=True)

    st.write("")
    plot(scan_figure(surrogate, enrichment, boron, density, result.history[-1], result.k_eff))

    st.caption(f"单次耦合求解 {elapsed:.0f} ms · 滑块变动即重算 · "
               f"UO₂ k(T_f) 随燃料温度动态更新 · 温度场由串联热阻模型给出")


def main() -> None:
    st.set_page_config(page_title="反应堆物理-热工耦合仪表盘", page_icon="⚛", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    tab_transient, tab_steady = st.tabs([
        "⚠️ 瞬态安全分析：弹棒事故",
        "🧪 稳态设计空间",
    ])
    with tab_transient:
        render_transient_tab()
    with tab_steady:
        render_history_comparison()
        st.divider()
        render_steady_state_dashboard()


if __name__ == "__main__":
    main()