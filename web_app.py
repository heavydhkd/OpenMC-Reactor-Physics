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

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ============================================================ 1. 主题

NEON = "#00FF9C"   # 荧光绿：主数据曲线
CYAN = "#22D3EE"   # 科幻蓝：辅助线 / 标注
VIOLET = "#A78BFA"
PINK = "#FF3D9A"   # 告警
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
</style>
"""

# ============================================================ 2. 一维导热（自包含）

PITCH_CM = 1.26
R_PELLET = 0.4096
R_CLAD_IN = 0.4180
R_CLAD_OUT = 0.4750
H_GAP = 0.7          # W/(cm²·K)，气隙换热系数
H_FILM = 3.0         # W/(cm²·K)，包壳-冷却剂对流换热系数
K_CLAD = 0.165       # W/(cm·K)，Zr-4 @600 K
K_FUEL = 0.035       # W/(cm·K)，UO₂ @~1000 K 常值简化


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


def temperature_field(power_w_cm: float, mod_temp: float) -> Field:
    """稳态一维导热：冷却剂 → 液膜 → 包壳 → 气隙 → 芯块，串联热阻依次叠加。"""
    d_film = power_w_cm / (2.0 * math.pi * R_CLAD_OUT * H_FILM)
    d_clad = power_w_cm / (2.0 * math.pi * K_CLAD) * math.log(R_CLAD_OUT / R_CLAD_IN)
    k_gap = H_GAP * (R_CLAD_IN - R_PELLET)              # 薄气隙等效导热系数
    d_gap = power_w_cm / (2.0 * math.pi * k_gap) * math.log(R_CLAD_IN / R_PELLET)
    d_fuel = power_w_cm / (4.0 * math.pi * K_FUEL)      # 均匀体积释热

    t_clad_out = mod_temp + d_film
    t_clad_in = t_clad_out + d_clad
    t_surface = t_clad_in + d_gap
    return Field(
        power=power_w_cm, t_center=t_surface + d_fuel, t_surface=t_surface,
        t_avg=t_surface + 2.0 / 3.0 * d_fuel,
        t_clad_in=t_clad_in, t_clad_out=t_clad_out,
        d_fuel=d_fuel, d_gap=d_gap, d_clad=d_clad, d_film=d_film)


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


# ============================================================ 4. Picard 耦合迭代

RELAX = 0.5        # 松弛因子，太小慢、太大会振荡
TOLERANCE = 0.05   # K，收敛判据
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
    """真的不动点迭代：k_eff 决定功率，功率决定温度，温度再回去影响 k_eff。"""
    mod_temp = temperature_from_density(density)
    fuel_temp = mod_temp + 40.0     # 冷态起步，曲线才有可看的收敛过程
    history: list[float] = []
    field = temperature_field(q_rated, mod_temp)

    for step in range(1, MAX_ITER + 1):
        k_eff, sigma = surrogate.k_eff(enrichment, boron_ppm, density, fuel_temp)
        # 反应性越大，实际功率越偏离额定值：1.0 表示 k=1 时正好额定功率
        power = q_rated * min(max(1.0 + (k_eff - 1.0), 0.25), 2.4)
        field = temperature_field(power, mod_temp)
        history.append(fuel_temp + RELAX * (field.t_avg - fuel_temp))
        if step > 1 and abs(history[-1] - history[-2]) < TOLERANCE:
            break
        fuel_temp = history[-1]

    k_eff, sigma = surrogate.k_eff(enrichment, boron_ppm, density, history[-1])
    return Coupled(
        history=history, k_eff=k_eff, sigma=sigma, field=field,
        iterations=len(history), converged=abs(history[-1] - history[-2]) < TOLERANCE
        if len(history) > 1 else True,
        fuel_temp_coeff=surrogate.fuel_temp_coefficient(
            enrichment, boron_ppm, density, history[-1]))


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
        subtitle = f"慢化剂 {mod_temp:.0f} K · 判据 |ΔT| &lt; {tolerance:g} K · {state}"

    _dark_layout(fig, "PICARD 迭代 · 燃料平均温度收敛轨迹　"
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


# ============================================================ 7. 主界面


def main() -> None:
    st.set_page_config(page_title="反应堆物理-热工耦合仪表盘", page_icon="⚛", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

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
        kpi("Picard 迭代", f"{result.iterations}",
            "已收敛" if result.converged else "达到迭代上限（可能振荡）",
            CYAN if result.converged else PINK)
    with c4:
        kpi("燃料温度系数", f"{result.fuel_temp_coeff:+.2f}", "pcm / K（±50 K 差分）", VIOLET)

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
        + f'<span class="chip">实际线功率 <b>{field.power:.1f} W/cm</b></span>',
        unsafe_allow_html=True)

    st.write("")
    plot(scan_figure(surrogate, enrichment, boron, density, result.history[-1], result.k_eff))

    st.caption(f"单次耦合求解 {elapsed:.0f} ms · 滑块变动即重算 · 温度场由串联热阻模型给出")


if __name__ == "__main__":
    main()