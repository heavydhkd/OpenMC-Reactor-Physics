import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import openmc

from anderson_accelerator import AndersonAccelerator

# ==========================================
# 0. 文件清理函数（防止读取上一次的旧数据）
# ==========================================
def clean_files():
    files = ['materials.xml', 'geometry.xml', 'settings.xml', 'tallies.xml', 
             'summary.h5', 'statepoint.50.h5']
    for f in files:
        if os.path.exists(f):
            os.remove(f)

# ==========================================
# 1. 极简一维热工模型 (纯物理公式计算)
# ==========================================
def calculate_fuel_temp(power_density):
    """
    根据体积释热率计算燃料中心温度。
    T_fuel = T_coolant + q''' * R_total
    """
    T_coolant = 600.0  # 冷却剂温度固定为 600K
    # 假设一个简化的综合热阻 (包含气隙、包壳导热、对流换热)
    # 这个数值是人为设定的，目的是让温度变化明显，便于观察收敛
    R_total = 3.0e-6
    
    T_fuel_new = T_coolant + (power_density * R_total)
    return T_fuel_new

# ==========================================
# 2. 单次 OpenMC 物理计算模型
# ==========================================
def run_openmc(fuel_temp_K):
    clean_files()
    
    # --- 材料定义 ---
    fuel = openmc.Material(name='UO2')
    fuel.add_element('U', 1, enrichment=4.95)
    fuel.add_element('O', 2)
    fuel.set_density('g/cm3', 10.257)
    fuel.temperature = fuel_temp_K  # 核心：输入热工反馈的温度

    clad = openmc.Material(name='Zircaloy') # 包壳材料（修复报错的关键）
    clad.add_element('Zr', 1)
    clad.set_density('g/cm3', 6.56)

    water = openmc.Material(name='Water')
    water.add_element('H', 2)
    water.add_element('O', 1)
    water.set_density('g/cm3', 0.743)
    water.add_s_alpha_beta('c_H_in_H2O')
    water.temperature = 600.0

    mats = openmc.Materials([fuel, clad, water])
    mats.export_to_xml()

    # --- 几何定义 ---
    fuel_cyl = openmc.ZCylinder(r=0.4096)
    clad_cyl = openmc.ZCylinder(r=0.475)
    
    # 三个区域：燃料内部、包壳层、外部慢化剂
    fuel_cell = openmc.Cell(fill=fuel, region=-fuel_cyl)
    clad_cell = openmc.Cell(fill=clad, region=+fuel_cyl & -clad_cyl) # 修复报错的关键
    mod_cell = openmc.Cell(fill=water, region=+clad_cyl)
    
    # 外部边界：全反射模拟无限阵列
    x_min = openmc.XPlane(-0.63, boundary_type='reflective')
    x_max = openmc.XPlane(0.63, boundary_type='reflective')
    y_min = openmc.YPlane(-0.63, boundary_type='reflective')
    y_max = openmc.YPlane(0.63, boundary_type='reflective')

    root = openmc.Cell(fill=openmc.Universe(cells=[fuel_cell, clad_cell, mod_cell]), 
                       region=+x_min & -x_max & +y_min & -y_max)
    geometry = openmc.Geometry([root])
    geometry.export_to_xml()

    # --- Tally 设置 (提取燃料区的能量沉积) ---
    heating_tally = openmc.Tally(name='heating')
    heating_tally.filters = [openmc.CellFilter(fuel_cell)]
    heating_tally.scores = ['heating']
    tallies = openmc.Tallies([heating_tally])
    tallies.export_to_xml()

    # --- 计算设置 ---
    settings = openmc.Settings()
    settings.temperature = {'method': 'interpolation'}
    settings.run_mode = 'eigenvalue'
    settings.batches = 50
    settings.inactive = 10
    settings.particles = 5000
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((0,0,0))
    settings.source = source
    settings.export_to_xml()

    # --- 运行并提取数据 ---
    openmc.run(output=False)
    
    sp = openmc.StatePoint(f'statepoint.{settings.batches}.h5')
    keff = sp.keff.nominal_value
    
    # 提取 heating tally 的值
    tally_res = sp.get_tally(name='heating')
    power_density = tally_res.mean.flatten()[0] 
    
    return keff, power_density

# ==========================================
# 3. 耦合迭代：Anderson 加速 vs 普通 Picard
# ==========================================
# 迭代参数集中放在这里，方便调参，也会写进 JSON 的 meta 字段。
T_INITIAL = 600.0           # 初始猜测燃料温度 (K)
RELAX_FACTOR = 0.5          # 普通 Picard 基线的固定松弛因子
TOLERANCE = 1.0             # 收敛判据：|T_calc - T_current| < 1 K
MAX_ITER = 30               # 每种方法的最大迭代次数
ANDERSON_M = 3              # Anderson 记忆深度 m

# 是否额外跑一遍普通 Picard 作为对照。
# 注意：Picard 轨迹必须自己独立调用 OpenMC 才能得到真实结果，
# 所以打开后总耗时约为只跑 Anderson 的 2 倍；只想看 Anderson 时改成 False。
RUN_PICARD_BASELINE = True

# JSON 固定写到项目根的 data/，避免依赖当前工作目录。
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_PATH = DATA_DIR / "iteration_history.json"


def coupled_step(T_current):
    """执行一次完整的物理-热工耦合步。

    参数
    ----
    T_current : float
        本轮传给 OpenMC 的燃料温度 (K)。

    返回
    ----
    (keff, power, T_calc) : tuple[float, float, float]
        keff   : 当前温度下的有效增殖因子；
        power  : OpenMC heating tally 给出的相对功率；
        T_calc : 热工模型算出的新温度，尚未做松弛或 Anderson 加速。
    """
    keff, power = run_openmc(T_current)
    T_calc = calculate_fuel_temp(power)
    return float(keff), float(power), float(T_calc)


def _record(history, iteration, T_next, T_calc, keff, power, residual):
    """把一轮迭代写进历史列表（字段都转成 Python float，保证 JSON 可序列化）。"""
    history.append(
        {
            "iteration": int(iteration),
            "temperature": float(T_next),
            "t_calc": float(T_calc),
            "keff": float(keff),
            "power": float(power),
            "residual": float(residual),
        }
    )


def run_picard(T0=T_INITIAL, relax_factor=RELAX_FACTOR,
               tol=TOLERANCE, max_iter=MAX_ITER):
    """普通 Picard + 固定松弛迭代，作为 Anderson 的对照基线。

    更新公式：
        T_next = T_current + relax_factor * (T_calc - T_current)

    收敛判据使用物理上的不动点残差 |T_calc - T_current|，
    而不是松弛后的步长，这样两种方法比较的是同一个物理标准。
    """
    T_current = float(T0)
    history = [{"iteration": 0, "temperature": T_current}]
    converged = False
    final_keff = None

    for iteration in range(1, max_iter + 1):
        keff, power, T_calc = coupled_step(T_current)
        T_next = T_current + relax_factor * (T_calc - T_current)
        residual = abs(T_calc - T_current)
        final_keff = keff

        _record(history, iteration, T_next, T_calc, keff, power, residual)
        print(
            f"  [Picard {iteration:02d}] "
            f"T_in={T_current:.2f} K, T_calc={T_calc:.2f} K, "
            f"T_next={T_next:.2f} K, |res|={residual:.3f} K, "
            f"k_eff={keff:.5f}"
        )

        T_current = T_next
        if residual < tol:
            converged = True
            break

    return {
        "method": "Picard + relaxation",
        "relax_factor": float(relax_factor),
        "converged": converged,
        "n_iterations": len(history) - 1,
        "final_temperature": float(T_current),
        "final_keff": float(final_keff) if final_keff is not None else None,
        "history": history,
    }


def run_anderson(T0=T_INITIAL, tol=TOLERANCE, max_iter=MAX_ITER):
    """Anderson 加速迭代：用 aae.accelerate 替代固定松弛。

    注意：AndersonAccelerator 的前 m 次调用会原样返回 T_calc
    （即不带松弛的普通 Picard），从第 m+1 次开始才真正加速。
    """
    # 加速器必须建在循环外，才能跨迭代保留历史。
    aae = AndersonAccelerator(m=ANDERSON_M)

    T_current = float(T0)
    history = [{"iteration": 0, "temperature": T_current}]
    converged = False
    final_keff = None

    for iteration in range(1, max_iter + 1):
        keff, power, T_calc = coupled_step(T_current)

        # 关键接口：把当前温度与热工算出的新温度都包成 (1,) 数组；
        # accelerate 返回同形状数组，取 [0] 得到标量。
        T_next = aae.accelerate(
            np.array([T_current]),
            np.array([T_calc]),
        )[0]
        T_next = float(T_next)
        residual = abs(T_calc - T_current)
        final_keff = keff

        _record(history, iteration, T_next, T_calc, keff, power, residual)
        print(
            f"  [Anderson {iteration:02d}] "
            f"T_in={T_current:.2f} K, T_calc={T_calc:.2f} K, "
            f"T_next={T_next:.2f} K, |res|={residual:.3f} K, "
            f"k_eff={keff:.5f}"
        )

        T_current = T_next
        if residual < tol:
            converged = True
            break

    return {
        "method": "Anderson Acceleration",
        "m": int(aae.m),
        "n_accelerated": int(aae.n_accelerated),
        "converged": converged,
        "n_iterations": len(history) - 1,
        "final_temperature": float(T_current),
        "final_keff": float(final_keff) if final_keff is not None else None,
        "history": history,
    }


def _method_summary(result):
    """把 history 展开成 Web UI 画图更方便的扁平数组。"""
    history = result["history"]
    summary = {
        "method": result["method"],
        "converged": result["converged"],
        "n_iterations": result["n_iterations"],
        "final_temperature": result["final_temperature"],
        "final_keff": result["final_keff"],
        "iterations": [h["iteration"] for h in history],
        "temperatures": [h["temperature"] for h in history],
        "t_calc": [h.get("t_calc") for h in history],
        "keff": [h.get("keff") for h in history],
        "residuals": [h.get("residual") for h in history],
        "history": history,
    }
    if "m" in result:
        summary["m"] = result["m"]
        summary["n_accelerated"] = result["n_accelerated"]
    return summary


def build_payload(anderson_result, picard_result):
    """组装最终 JSON。picard_result 为 None 时表示没有跑基线。"""
    payload = {
        "meta": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "initial_temperature": float(T_INITIAL),
            "relax_factor": float(RELAX_FACTOR),
            "tolerance": float(TOLERANCE),
            "max_iterations": int(MAX_ITER),
            "anderson_m": int(ANDERSON_M),
            "convergence_criterion": "abs(T_calc - T_current) < tolerance",
            "picard_baseline_enabled": bool(RUN_PICARD_BASELINE),
        },
        "anderson": _method_summary(anderson_result),
        "picard": _method_summary(picard_result) if picard_result else None,
        "comparison": None,
    }

    if picard_result:
        n_a = anderson_result["n_iterations"]
        n_p = picard_result["n_iterations"]
        payload["comparison"] = {
            "anderson_iterations": n_a,
            "picard_iterations": n_p,
            "iteration_saving": n_p - n_a,
            "speedup_ratio": (n_p / n_a) if n_a > 0 else None,
        }

    return payload


def save_iteration_history(payload, path=HISTORY_PATH):
    """把结果写成 UTF-8 JSON；返回实际写入路径，方便日志打印。"""
    # 同时兼容 str 和 pathlib.Path，避免调用方传 Path 时字符串拼接失败。
    path = os.fspath(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # 先写临时文件再原子替换，防止写到一半被 Ctrl+C 打断留下半个 JSON。
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


if __name__ == "__main__":
    print("=== 开始物理-热工耦合迭代 ===")
    print(
        f"初始温度: {T_INITIAL:.2f} K | "
        f"收敛判据: |T_calc - T_current| < {TOLERANCE} K | "
        f"最大迭代: {MAX_ITER}"
    )
    print(
        "Picard 基线: "
        + ("开启（总耗时约为只跑 Anderson 的 2 倍）"
           if RUN_PICARD_BASELINE else "关闭")
    )

    print("\n=== [1/2] Anderson 加速迭代 ===")
    anderson_result = run_anderson()

    # 先落盘一份只含 Anderson 的结果：Picard 基线较慢，
    # 万一中途按 Ctrl+C，至少已经拿到加速器的轨迹。
    partial_path = save_iteration_history(build_payload(anderson_result, None))
    print(f"\n[临时结果] 已写入: {partial_path}")

    picard_result = None
    if RUN_PICARD_BASELINE:
        print("\n=== [2/2] 普通 Picard 基线（用于对比）===")
        picard_result = run_picard()
    else:
        print("\n[跳过] RUN_PICARD_BASELINE=False，未运行 Picard 基线。")

    payload = build_payload(anderson_result, picard_result)
    path = save_iteration_history(payload)

    print("\n=== 对比结果 ===")
    print(
        f"Anderson : {anderson_result['n_iterations']:2d} 次, "
        f"收敛={anderson_result['converged']}, "
        f"最终 T={anderson_result['final_temperature']:.2f} K, "
        f"k_eff={anderson_result['final_keff']:.5f}"
    )

    if picard_result:
        print(
            f"Picard   : {picard_result['n_iterations']:2d} 次, "
            f"收敛={picard_result['converged']}, "
            f"最终 T={picard_result['final_temperature']:.2f} K, "
            f"k_eff={picard_result['final_keff']:.5f}"
        )
        n_a = anderson_result["n_iterations"]
        n_p = picard_result["n_iterations"]
        if n_a < n_p:
            print(
                f"节省迭代 : {n_p - n_a} 次 "
                f"({n_p / n_a:.2f}× Picard / Anderson)"
            )
        elif n_a == n_p:
            print("两种方法迭代次数相同")
        else:
            print(f"Anderson 多用了 {n_a - n_p} 次迭代（本例未加速）")
    else:
        print("Picard   : 未运行（RUN_PICARD_BASELINE=False）")

    print(f"\n💾 迭代历史已保存: {path}")
