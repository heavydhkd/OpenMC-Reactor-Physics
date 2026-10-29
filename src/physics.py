"""纯 Python 物性关系与几何校验（不依赖 OpenMC，可被单元测试直接调用）。

这里刻意不引入 ``iapws`` 等第三方库：水密度表直接以 IAPWS-IF97 在
15.5 MPa 下的计算结果内插，既保证可复现，也让 ``pytest`` 不需要额外依赖。

.. warning::
    VERA P1 基准题规定的慢化剂密度 0.743 g/cm³ 与 15.5 MPa/600 K 下的
    热力学值 0.661 g/cm³ 并不自洽（0.743 对应约 567 K 的饱和水）。
    因此本模块提供两种模型：
    * ``"iapws_table"``：物理模型，随温度变化，用于参数化研究；
    * ``"constant"``：常数模型，用于复现基准题规定的参考态。
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Iterable, List, Tuple

#: 物理模型版本号，写入数据集用于区分"旧几何/旧硼浓度"的历史数据
PHYSICS_VERSION = "2.0"

#: 最小二乘意义上的"工程有效"包壳/芯块间隙下限（cm）
MIN_GAP_CM = 1.0e-3

# --------------------------------------------------------------------------
# 水的饱和温度（IAPWS-95 饱和压力辅助方程，Wagner & Pruss 2002）
# --------------------------------------------------------------------------
_T_C = 647.096          # 临界温度 [K]
_P_C = 22.064           # 临界压力 [MPa]
_PSAT_COEFFS = (
    -7.85951783,
    1.84408259,
    -11.7866497,
    22.6807411,
    -15.9618719,
    1.80122502,
)


def saturation_pressure(temperature_k: float) -> float:
    """给定温度返回饱和压力 [MPa]（Wagner-Pruss 辅助方程）。"""
    if not 273.16 <= temperature_k < _T_C:
        raise ValueError(f"饱和温度超出适用范围 (273.16–{_T_C} K)：{temperature_k}")
    theta = 1.0 - temperature_k / _T_C
    # theta^1.5 无法用整数幂，逐项写出以免可读性下降
    terms = (
        _PSAT_COEFFS[0] * theta,
        _PSAT_COEFFS[1] * theta ** 1.5,
        _PSAT_COEFFS[2] * theta ** 3.0,
        _PSAT_COEFFS[3] * theta ** 3.5,
        _PSAT_COEFFS[4] * theta ** 4.0,
        _PSAT_COEFFS[5] * theta ** 7.5,
    )
    return _P_C * math.exp((_T_C / temperature_k) * sum(terms))


def saturation_temperature(pressure_mpa: float) -> float:
    """给定压力返回饱和温度 [K]，在 (273.16, 647.095) 区间内二分求解。"""
    if not (saturation_pressure(273.17) < pressure_mpa < _P_C):
        raise ValueError(f"饱和压力超出适用范围 (0–{_P_C} MPa)：{pressure_mpa}")
    lo, hi = 273.17, _T_C - 1e-3
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if saturation_pressure(mid) < pressure_mpa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# 慢化剂密度
# --------------------------------------------------------------------------
#: 15.5 MPa 下的水密度表（IAPWS-IF97，T[K] -> rho[g/cm³]），步长 10 K
COOLANT_DENSITY_TABLE: Tuple[Tuple[float, float], ...] = (
    (470.0, 0.8784),
    (480.0, 0.8671),
    (490.0, 0.8552),
    (500.0, 0.8428),
    (510.0, 0.8298),
    (520.0, 0.8160),
    (530.0, 0.8015),
    (540.0, 0.7861),
    (550.0, 0.7697),
    (560.0, 0.7521),
    (570.0, 0.7329),
    (580.0, 0.7119),
    (590.0, 0.6883),
    (600.0, 0.6611),
    (610.0, 0.6282),
)

#: 基准题（VERA P1）规定的慢化剂密度，用于 ``constant`` 模型
BENCHMARK_COOLANT_DENSITY = 0.743  # g/cm³

_PRESSURE_MPA = 15.5


def iapws_water_density(temperature_k: float) -> float:
    """15.5 MPa 下的水密度 [g/cm³]，表外线性外推并在 ±10 K 外给出警告。"""
    table = COOLANT_DENSITY_TABLE
    t_lo, rho_lo = table[0]
    t_hi, rho_hi = table[-1]
    if temperature_k < t_lo - 10.0 or temperature_k > t_hi + 10.0:
        warnings.warn(
            f"慢化剂温度 {temperature_k:.1f} K 超出密度表范围 "
            f"[{t_lo:.0f}, {t_hi:.0f}] K，已线性外推，结果可能不物理",
            RuntimeWarning,
            stacklevel=2,
        )
    if temperature_k <= t_lo:
        slope = (table[1][1] - table[0][1]) / (table[1][0] - table[0][0])
        return rho_lo + slope * (temperature_k - t_lo)
    if temperature_k >= t_hi:
        slope = (table[-1][1] - table[-2][1]) / (table[-1][0] - table[-2][0])
        return rho_hi + slope * (temperature_k - t_hi)
    for (t0, r0), (t1, r1) in zip(table, table[1:]):
        if t0 <= temperature_k <= t1:
            frac = (temperature_k - t0) / (t1 - t0)
            return r0 + frac * (r1 - r0)
    raise AssertionError("unreachable")


def water_density(
    temperature_k: float,
    model: str = "iapws_table",
    constant: float | None = None,
) -> float:
    """返回慢化剂密度 [g/cm³]。

    Parameters
    ----------
    temperature_k : 慢化剂温度 [K]
    model : ``"iapws_table"``（物理模型）或 ``"constant"``（基准题常数模型）
    constant : ``constant`` 模型下的密度，缺省为 :data:`BENCHMARK_COOLANT_DENSITY`
    """
    model = model.lower()
    if model in ("iapws", "iapws_table", "table"):
        return iapws_water_density(temperature_k)
    if model in ("constant", "fixed", "benchmark"):
        return BENCHMARK_COOLANT_DENSITY if constant is None else float(constant)
    raise ValueError(f"未知的冷却剂密度模型：{model!r}（可选 iapws_table / constant）")


# --------------------------------------------------------------------------
# 燃料与包壳导热系数（lumped 一维热工模型使用，单位 W/(cm·K)）
# --------------------------------------------------------------------------
def k_uo2(temperature_k: float) -> float:
    """UO2 导热系数 [W/(cm·K)]，NFI/MATPRO 关联式（95 % 理论密度、未辐照）。"""
    t = max(float(temperature_k), 300.0)
    k_w_m_k = 1.0 / (0.0375 + 2.165e-4 * t) + 4.715e9 / t ** 2 * math.exp(-16361.0 / t)
    return k_w_m_k / 100.0


def k_zircaloy(temperature_k: float) -> float:
    """Zircaloy-4 导热系数 [W/(cm·K)]，MATPRO 三次多项式拟合。"""
    t = float(temperature_k)
    k_w_m_k = 7.51 + 2.09e-2 * t - 1.45e-5 * t ** 2 + 7.67e-9 * t ** 3
    return k_w_m_k / 100.0


# --------------------------------------------------------------------------
# 硼浓度换算
# --------------------------------------------------------------------------
def boron_mass_fraction(boron_ppm: float) -> float:
    """ppm（溶液质量分数 ×1e6）-> 质量分数，用于 ``percent_type='wo'``。"""
    if boron_ppm < 0:
        raise ValueError(f"硼浓度不能为负：{boron_ppm}")
    return float(boron_ppm) * 1e-6


def expected_b_h2o_ratio(boron_ppm: float, density_g_cm3: float) -> float:
    """给定 ppm 与溶液密度，返回 B 与 H2O 的摩尔比（用于验证材料定义是否正确）。

    这是把"硼按原子分数加入"这个历史 bug 量化的判据：把 B 当作原子分数处理
    会得到明显偏低的 B/H2O 比，对应反应性效应约弱 1.6 倍。
    """
    m_h2o = 18.01528
    m_b = 10.811
    w_b = boron_mass_fraction(boron_ppm)
    n_h2o = density_g_cm3 * (1.0 - w_b) / m_h2o
    n_b = density_g_cm3 * w_b / m_b
    return n_b / n_h2o


# --------------------------------------------------------------------------
# 几何校验
# --------------------------------------------------------------------------
def validate_pin_geometry(
    pellet_radius_cm: float,
    clad_inner_cm: float,
    clad_outer_cm: float,
    pitch_cm: float,
) -> List[str]:
    """检查栅元几何是否闭合，返回问题列表（空列表表示几何合法）。

    历史版本的 ``clad_inner_r = clad_outer_r - 0.057`` 会让 0.4096 cm 的芯块
    与 0.4180 cm 的包壳内壁之间 84 μm 的气隙被燃料填满；更糟的是当
    ``clad_outer_r < 0.4666`` 时包壳内半径会小于芯块半径，几何直接失效。
    """
    problems: List[str] = []
    if clad_outer_cm <= clad_inner_cm:
        problems.append(f"包壳外半径 {clad_outer_cm:.4f} 必须大于内半径 {clad_inner_cm:.4f}")
    if clad_inner_cm <= pellet_radius_cm:
        problems.append(
            f"包壳内半径 {clad_inner_cm:.4f} cm 必须大于芯块半径 "
            f"{pellet_radius_cm:.4f} cm（否则芯块无法装入包壳）"
        )
    elif clad_inner_cm - pellet_radius_cm < MIN_GAP_CM:
        problems.append(
            f"芯块-包壳间隙 {clad_inner_cm - pellet_radius_cm:.4f} cm "
            f"小于下限 {MIN_GAP_CM} cm"
        )
    if pitch_cm <= 2.0 * clad_outer_cm:
        problems.append(
            f"栅距 {pitch_cm:.3f} cm 过小，包壳之间重叠（需 > {2.0 * clad_outer_cm:.3f} cm）"
        )
    return problems
