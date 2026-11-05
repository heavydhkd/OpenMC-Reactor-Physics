"""高保真点堆动力学（PRK）与双节点热工耦合的瞬态求解器。

状态向量
--------
    y = [n, C1, C2, C3, C4, C5, C6, Tf, Tm]

其中：
    n       归一化中子密度（初始为 1.0，功率正比于 n）
    C_i     第 i 组缓发中子先驱核浓度（按题给公式归一化）
    Tf      燃料温度，K
    Tm      冷却剂温度，K

方程组
------
    rho(t) = rho_ext(t) + alpha_f * (Tf - Tf0) + alpha_m * (Tm - Tm0)

    dn/dt   = ((rho - beta_eff) / Lambda) * n + sum_i(lambda_i * C_i)
    dCi/dt  = (beta_i / Lambda) * n - lambda_i * C_i
    Cf dTf/dt = P0 * n - K_fm * (Tf - Tm)
    Cm dTm/dt = K_fm * (Tf - Tm) - 2 * W * cp * (Tm - 565)

    0 <= t <= 0.1 s:  rho_ext 从 0 线性升到 0.015（1500 pcm）
    t > 0.1 s:        rho_ext 保持 0.015

数值方法
--------
    使用 scipy.integrate.solve_ivp，强制 BDF 或 Radau。
    PRK 的时间尺度从 Lambda=2e-5 s 到缓发中子/热工秒级，方程是刚性的；
    显式方法（RK45 等）会需要极小步长，甚至数值发散。
    默认使用 BDF，rtol=1e-10、atol=1e-12、max_step=5e-3 s，
    输出 6001 个等距时间点（0.005 s 分辨率）。

热工参数说明（题面未给出，以下为单通道 PWR 集总参数示例）
------------------------------------------------------------
    P0       = 6.0e4 W        单棒/单通道额定热功率
    Cf       = 500 J/K        燃料集总热容
    Cm       = 2.0e4 J/K      冷却剂集总热容
    2*W*cp   = 2.0e3 W/K      冷却剂流量热容（线性排热系数）
    K_fm     = P0 / (Tf0 - Tm0) = 1714.29 W/K
              取这个值是为了让 n=1 时 dTf/dt = 0，即燃料初始热平衡。

重要物理说明
------------
1. 1500 pcm 大于 beta_eff=650 pcm，因此这是 prompt supercritical
   （瞬发超临界）插入。功率会先出现一个尖峰，然后被负温度反馈压回；
   这不是数值爆炸，而是方程本身描述的物理过程。
2. 题给 Tm0=565 K，而冷却剂排热项是 (Tm-565)。在 n=1、Tf0>Tm0 时
   dTm/dt = K_fm*(Tf0-Tm0)/Cm = 3 K/s，并不严格等于 0。
   也就是说，题给初值只能保证中子学初始临界和燃料初始热平衡，
   不能同时保证冷却剂严格稳态。本程序忠实保留题给方程与初值，
   并在 JSON 的 meta 中记录这一点。

运行
----
    python transient_precise.py
    python transient_precise.py --method Radau
    python transient_precise.py --output /tmp/transient.json --points 12001

输出
----
    默认写入脚本同目录的 transient_precise_results.json（UTF-8，原子替换）。
    JSON 顶层包含 meta、t、n、Tf、Tm、rho 六个字段，数组长度一致。
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

# =============================================================================
# 1. 物理常量（题给值，不要随意修改）
# =============================================================================

# 6 组缓发中子份额 beta_i 与衰变常数 lambda_i（单位 1/s）
BETA = np.array(
    [0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273],
    dtype=np.float64,
)
LAMBDA_D = np.array(
    [0.0124, 0.0305, 0.1110, 0.3010, 1.1400, 3.0100],
    dtype=np.float64,
)

# 总有效缓发中子份额与中子代时间
BETA_EFF = 0.0065
NEUTRON_LIFETIME = 2.0e-5  # s

# 反应性反馈系数（单位 1/K；乘 1e5 即 pcm/K）
ALPHA_F = -3.0e-5  # -3 pcm/K
ALPHA_M = -3.0e-4  # -30 pcm/K

# =============================================================================
# 2. 初始条件与外部反应性
# =============================================================================

N0 = 1.0
TF0 = 600.0
TM0 = 565.0

RHO_EXT_MAX = 0.015  # 1500 pcm
RAMP_TIME = 0.1  # s
T_END_DEFAULT = 30.0  # s

# =============================================================================
# 3. 热工参数（题面未给，采用单通道 PWR 集总参数示例）
# =============================================================================

P0_W = 6.0e4  # W
C_FUEL = 500.0  # J/K
C_COOLANT = 2.0e4  # J/K
TWO_W_CP = 2.0e3  # W/K，即 2*W*cp

# 让 n=1 时燃料方程右端为 0：P0 - K_fm*(Tf0-Tm0) = 0。
K_FM = P0_W / (TF0 - TM0)  # W/K

# 数值参数
DEFAULT_METHOD = "BDF"
DEFAULT_RTOL = 1.0e-10
DEFAULT_ATOL = 1.0e-12
DEFAULT_MAX_STEP = 5.0e-3  # s
DEFAULT_POINTS = 6001

# 输出文件固定在脚本目录，避免依赖当前工作目录（WSL 下同样正确）。
OUTPUT_PATH = Path(__file__).resolve().parent / "transient_precise_results.json"

# 状态向量下标，避免魔法数字
IDX_N = 0
IDX_C = slice(1, 7)
IDX_TF = 7
IDX_TM = 8

# =============================================================================
# 4. 方程右端
# =============================================================================


def external_reactivity(t: float) -> float:
    """外部反应性 rho_ext(t)，单位 Δk/k。

    0 ~ 0.1 s 线性从 0 升到 0.015，之后保持 0.015。
    对 t<0 返回 0，避免积分器试探负时间时出现非物理值。
    """
    if t <= 0.0:
        return 0.0
    if t >= RAMP_TIME:
        return RHO_EXT_MAX
    return RHO_EXT_MAX * (t / RAMP_TIME)


def total_reactivity(t: float, tf: float, tm: float) -> float:
    """总反应性 = 外部反应性 + 燃料反馈 + 冷却剂反馈。"""
    return (
        external_reactivity(t)
        + ALPHA_F * (tf - TF0)
        + ALPHA_M * (tm - TM0)
    )


def initial_state() -> np.ndarray:
    """按题给公式构造 y(0)。

    Ci0 = beta_i / (Lambda * lambda_i) * n0
    该式使 t=0、rho=0 时 dn/dt 近似为 0：题给 beta_eff=0.0065，
    而 sum(beta_i)=0.006502，二者相差约 0.2 pcm，因此严格来说
    dn/dt=0.1 s^-1；相对 1500 pcm 的插入可忽略，但这里如实说明。
    """
    ci0 = BETA / (NEUTRON_LIFETIME * LAMBDA_D) * N0
    return np.concatenate(
        (
            np.array([N0], dtype=np.float64),
            ci0,
            np.array([TF0, TM0], dtype=np.float64),
        )
    )


def rhs(t: float, y: np.ndarray) -> np.ndarray:
    """PRK + 双节点热工方程的右端项。

    这里不做 max(n, 0) 之类的截断：截断会改变方程本身，并可能掩盖
    数值问题。BDF/Radau 在正初值下会保持 n>0；求解完成后仍会检查。
    """
    n = y[IDX_N]
    c = y[IDX_C]
    tf = y[IDX_TF]
    tm = y[IDX_TM]

    rho = total_reactivity(t, tf, tm)

    # 中子密度：瞬发项 + 6 组缓发中子源项
    dn_dt = ((rho - BETA_EFF) / NEUTRON_LIFETIME) * n + float(
        np.dot(LAMBDA_D, c)
    )

    # 每组先驱核：产生项 - 衰变项
    dc_dt = (BETA / NEUTRON_LIFETIME) * n - LAMBDA_D * c

    # 燃料节点：裂变热 - 向冷却剂传热
    dtf_dt = (P0_W * n - K_FM * (tf - tm)) / C_FUEL

    # 冷却剂节点：从燃料吸热 - 流量带走的热量
    dtm_dt = (K_FM * (tf - tm) - TWO_W_CP * (tm - TM0)) / C_COOLANT

    dydt = np.empty_like(y)
    dydt[IDX_N] = dn_dt
    dydt[IDX_C] = dc_dt
    dydt[IDX_TF] = dtf_dt
    dydt[IDX_TM] = dtm_dt
    return dydt


# =============================================================================
# 5. 求解
# =============================================================================


def solve_transient(
    *,
    method: str = DEFAULT_METHOD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_step: float = DEFAULT_MAX_STEP,
    t_end: float = T_END_DEFAULT,
    n_points: int = DEFAULT_POINTS,
) -> Any:
    """用 solve_ivp 求解瞬态，返回 scipy 的 OdeResult。

    参数
    ----
    method : {"BDF", "Radau"}
        刚性 ODE 求解器。题面要求强制指定，二者均可用。
    rtol, atol : float
        相对/绝对误差控制。默认 1e-10 / 1e-12。
    max_step : float
        最大步长。默认 5 ms，保证 0.1 s 爬升段和功率尖峰不被跨过。
    t_end : float
        终止时间，默认 30 s。
    n_points : int
        输出时间点数量，默认 6001（0.005 s 间隔）。
    """
    if method not in ("BDF", "Radau"):
        raise ValueError("method 必须是 'BDF' 或 'Radau'")
    if t_end <= 0.0:
        raise ValueError("t_end 必须为正数")
    if n_points < 2:
        raise ValueError("n_points 必须 >= 2")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError("rtol 和 atol 必须为正数")
    if max_step <= 0.0:
        raise ValueError("max_step 必须为正数")

    t_eval = np.linspace(0.0, t_end, n_points)
    y0 = initial_state()

    sol = solve_ivp(
        fun=rhs,
        t_span=(0.0, t_end),
        y0=y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp 求解失败：{sol.message}")
    if sol.y.shape != (y0.size, n_points):
        raise RuntimeError(
            f"输出形状异常：期望 {(y0.size, n_points)}，实际 {sol.y.shape}"
        )
    if not np.isfinite(sol.y).all():
        raise RuntimeError("解中出现 NaN 或 Inf，拒绝写入 JSON")
    if np.any(sol.y[IDX_N] < -1e-12):
        raise RuntimeError("中子密度出现负值，拒绝写入 JSON")

    return sol


# =============================================================================
# 6. 组装并保存 JSON
# =============================================================================


def build_payload(
    sol: Any,
    *,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
) -> dict[str, Any]:
    """把 OdeResult 转成 Web UI 友好的 JSON 字典。"""
    t = sol.t
    n = sol.y[IDX_N]
    tf = sol.y[IDX_TF]
    tm = sol.y[IDX_TM]

    # 输出时刻的总反应性，便于 UI 画 rho(t) 或做诊断。
    rho = np.array(
        [
            total_reactivity(float(ti), float(tfi), float(tmi))
            for ti, tfi, tmi in zip(t, tf, tm)
        ],
        dtype=np.float64,
    )

    i_pmax = int(np.argmax(n))
    i_tfmax = int(np.argmax(tf))
    i_tmmax = int(np.argmax(tm))

    meta: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "solver": {
            "library": "scipy.integrate.solve_ivp",
            "method": method,
            "rtol": float(rtol),
            "atol": float(atol),
            "max_step_s": float(max_step),
            "t_span_s": [0.0, float(t[-1])],
            "n_points": int(t.size),
            "success": bool(sol.success),
            "message": str(sol.message),
            "nfev": int(sol.nfev),
            "njev": int(getattr(sol, "njev", 0) or 0),
        },
        "units": {
            "t": "s",
            "n": "dimensionless (normalized to 1.0 at t=0)",
            "Tf": "K",
            "Tm": "K",
            "rho": "Delta k / k (absolute)",
        },
        "neutronics": {
            "beta_i": BETA.tolist(),
            "lambda_i_per_s": LAMBDA_D.tolist(),
            "beta_eff": float(BETA_EFF),
            "Lambda_s": float(NEUTRON_LIFETIME),
            "alpha_f_per_K": float(ALPHA_F),
            "alpha_m_per_K": float(ALPHA_M),
            "rho_ext_max": float(RHO_EXT_MAX),
            "rho_ext_max_pcm": float(RHO_EXT_MAX * 1.0e5),
            "ramp_time_s": float(RAMP_TIME),
            "prompt_critical": bool(RHO_EXT_MAX > BETA_EFF),
            "prompt_critical_excess_pcm": float(
                (RHO_EXT_MAX - BETA_EFF) * 1.0e5
            ),
        },
        "thermal_parameters": {
            "P0_W": float(P0_W),
            "C_fuel_J_per_K": float(C_FUEL),
            "C_coolant_J_per_K": float(C_COOLANT),
            "K_fm_W_per_K": float(K_FM),
            "two_W_cp_W_per_K": float(TWO_W_CP),
            "note": (
                "题面未给热工参数；以上为单通道 PWR 集总参数示例。"
                "K_fm 由 P0=K_fm*(Tf0-Tm0) 反推，使 n=1 时 dTf/dt=0。"
            ),
        },
        "initial_conditions": {
            "n0": float(N0),
            "Tf0_K": float(TF0),
            "Tm0_K": float(TM0),
            "Ci0": [float(x) for x in (BETA / (NEUTRON_LIFETIME * LAMBDA_D))],
            "rho0": 0.0,
        },
        "summary": {
            "max_power_n": float(n[i_pmax]),
            "time_of_max_power_s": float(t[i_pmax]),
            "max_fuel_temperature_K": float(tf[i_tfmax]),
            "time_of_max_fuel_temperature_s": float(t[i_tfmax]),
            "max_coolant_temperature_K": float(tm[i_tmmax]),
            "time_of_max_coolant_temperature_s": float(t[i_tmmax]),
            "final_power_n": float(n[-1]),
            "final_fuel_temperature_K": float(tf[-1]),
            "final_coolant_temperature_K": float(tm[-1]),
            "final_reactivity": float(rho[-1]),
        },
        "notes": [
            "rho_ext=1500 pcm > beta_eff=650 pcm，瞬态为 prompt supercritical；"
            "功率尖峰由瞬发中子引起，随后被负温度反馈压回。",
            "题给 Tm0=565 K 且冷却项为 (Tm-565)，在 n=1 时 "
            "dTm/dt=K_fm*(Tf0-Tm0)/Cm=3 K/s，因此冷却剂并非严格热工稳态；"
            "程序忠实保留题给方程与初值。",
        ],
    }

    return {
        "meta": meta,
        "t": t.tolist(),
        "n": n.tolist(),
        "Tf": tf.tolist(),
        "Tm": tm.tolist(),
        "rho": rho.tolist(),
    }


def save_results(payload: dict[str, Any], path: Path | str = OUTPUT_PATH) -> Path:
    """UTF-8 写入 JSON，先写临时文件再原子替换。

    这样即使写入过程中被打断，也不会留下半个 JSON 供 Web UI 读取。
    使用 os.replace，Linux/Windows 都可用；不依赖任何 Windows 专用库。
    """
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(
                payload,
                fh,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            fh.write("\n")
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return out_path


# =============================================================================
# 7. 命令行
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "高保真点堆动力学 + 双节点热工耦合瞬态求解器 "
            "(scipy solve_ivp, BDF/Radau)"
        )
    )
    parser.add_argument(
        "--method",
        choices=("BDF", "Radau"),
        default=DEFAULT_METHOD,
        help="刚性 ODE 方法，默认 BDF",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="输出 JSON 路径，默认脚本同目录 transient_precise_results.json",
    )
    parser.add_argument(
        "--rtol", type=float, default=DEFAULT_RTOL, help="相对误差，默认 1e-10"
    )
    parser.add_argument(
        "--atol", type=float, default=DEFAULT_ATOL, help="绝对误差，默认 1e-12"
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=DEFAULT_MAX_STEP,
        help="最大步长（秒），默认 5e-3",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=DEFAULT_POINTS,
        help="输出时间点数量，默认 6001",
    )
    parser.add_argument(
        "--t-end", type=float, default=T_END_DEFAULT, help="终止时间（秒），默认 30"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.points < 2:
        parser.error("--points 必须 >= 2")
    if args.t_end <= 0.0:
        parser.error("--t-end 必须为正数")
    if args.max_step <= 0.0:
        parser.error("--max-step 必须为正数")
    if args.rtol <= 0.0 or args.atol <= 0.0:
        parser.error("--rtol/--atol 必须为正数")

    sol = solve_transient(
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        max_step=args.max_step,
        t_end=args.t_end,
        n_points=args.points,
    )
    payload = build_payload(
        sol,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        max_step=args.max_step,
    )
    out_path = save_results(payload, args.output)

    summary = payload["meta"]["summary"]
    print(
        "[transient] method={method} success=True nfev={nfev} points={points}".format(
            method=args.method,
            nfev=payload["meta"]["solver"]["nfev"],
            points=len(payload["t"]),
        )
    )
    print(
        "[transient] peak n={n:.6g} at t={t:.4f} s; "
        "max Tf={tf:.3f} K at t={tt:.4f} s".format(
            n=summary["max_power_n"],
            t=summary["time_of_max_power_s"],
            tf=summary["max_fuel_temperature_K"],
            tt=summary["time_of_max_fuel_temperature_s"],
        )
    )
    print(
        "[transient] final n={n:.6g}, Tf={tf:.3f} K, Tm={tm:.3f} K".format(
            n=summary["final_power_n"],
            tf=summary["final_fuel_temperature_K"],
            tm=summary["final_coolant_temperature_K"],
        )
    )
    print(f"[transient] saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
