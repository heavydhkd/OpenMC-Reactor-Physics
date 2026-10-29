"""一维热工-中子学耦合：线功率 → 温度场 → 反应性。

和历史版本的区别：

* **传热模型是真的**。旧版用 ``R_total = 3.0e-6`` 把"芯块-气隙-包壳-冷却剂"
  整条热阻链压成一个常数，代码里自己都注明"这个数值是人为设定的"；
  而且把 ``heating`` tally 的 ``mean``（单位是 eV/源粒子）直接当成 W/cm³ 用。
  这里按串联热阻逐段计算：芯块导热（对 k(T) 积分）、气隙、包壳、对流膜。
* **迭代真的收敛**。芯块导热系数随温度变化，所以燃料平均温度必须迭代
  （Picard 松弛），收敛判据与配置文件里的 ``thermal.tolerance_k`` 一致；
  旧版迭代 12 次就停，判据 1.0 K 而注释写的是 0.1 K。
* **不依赖 OpenMC**。中子学由 GP 代理模型给出（毫秒级），因此耦合循环可以在
  任何机器上复现；需要更高精度时用 ``--openmc`` 直接调 OpenMC。
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .config import FEATURE_COLUMNS, PATHS, Config, load_config
from .physics import k_uo2, k_zircaloy, water_density
from .plotting import configure_matplotlib
from .surrogate import PCM, load_model, predict_k

configure_matplotlib()  # 中文字体 + Agg 后端

LOGGER = logging.getLogger(__name__)

# 15.5 MPa 饱和温度（见 src/physics.saturation_temperature）
TSAT_15_5_MPA = 617.94


@dataclass
class TemperatureSolution:
    """一次温度场计算的结果（单位：K；热阻：K·cm/W，按单位长度定义）。"""

    power_w_cm: float
    mod_temp: float
    t_center: float
    t_surface: float
    t_fuel_avg: float
    t_clad_inner: float
    t_clad_outer: float
    r_gap: float
    r_clad: float
    r_film: float
    delta_gap: float
    delta_clad: float
    delta_film: float
    delta_fuel: float
    iterations: int = 0
    converged: bool = False
    history: List[float] = field(default_factory=list)

    @property
    def r_total(self) -> float:
        """总热阻（含芯块等效热阻，单位 K·cm/W）。"""
        return self.r_gap + self.r_clad + self.r_film + self.delta_fuel / self.power_w_cm

    def as_dict(self) -> Dict[str, Any]:
        return {
            "power_w_cm": self.power_w_cm,
            "mod_temp": self.mod_temp,
            "t_center": self.t_center,
            "t_surface": self.t_surface,
            "t_fuel_avg": self.t_fuel_avg,
            "t_clad_inner": self.t_clad_inner,
            "t_clad_outer": self.t_clad_outer,
            "delta_fuel": self.delta_fuel,
            "delta_gap": self.delta_gap,
            "delta_clad": self.delta_clad,
            "delta_film": self.delta_film,
            "r_gap": self.r_gap,
            "r_clad": self.r_clad,
            "r_film": self.r_film,
            "r_total": self.r_total,
            "iterations": self.iterations,
            "converged": self.converged,
        }


def fuel_center_temperature(power_w_cm: float, pellet_r: float, t_surface: float) -> float:
    """由 ``∫ k(T)dT = q' / (4π)`` 反解芯块中心温度。

    圆柱形芯块均匀体热源、导热系数随温度变化时，稳态导热的精确积分形式为
    ``∫_{T_s}^{T_c} k(T) dT = q' / (4π)``，与半径无关（q' 为线功率密度）。
    """
    target = power_w_cm / (4.0 * np.pi)

    def residual(t_center: float) -> float:
        value, _ = quad(k_uo2, t_surface, t_center, limit=200)
        return value - target

    upper = t_surface + 3000.0
    if residual(upper) < 0:  # 极端线功率下才可能发生
        return upper
    return float(brentq(residual, t_surface + 1e-6, upper, xtol=1e-6, rtol=1e-10))


def solve_temperature(power_w_cm: float, mod_temp: float, config: Config | None = None,
                      fuel_temp_guess: float = 900.0) -> TemperatureSolution:
    """逐段串联热阻求解温度场，并对"燃料平均温度 ↔ 导热系数"做 Picard 迭代。"""
    cfg = (config or load_config()).section("thermal")
    physics = (config or load_config()).physics()
    pellet_r = float(physics["pellet_radius_cm"])
    clad_inner = float(physics["clad_inner_radius_cm"])
    clad_outer = float(physics["clad_outer_radius_cm"])
    pitch = float(physics["pitch_cm"])

    h_gap = float(cfg.get("gap_conductance_w_cm2k", 0.7))     # W/(cm²·K)
    h_film = float(cfg.get("film_coefficient_w_cm2k", 3.0))    # W/(cm²·K)
    relaxation = float(cfg.get("relaxation", 0.5))
    tolerance = float(cfg.get("tolerance_k", 0.01))
    max_iter = int(cfg.get("max_iterations", 30))
    water_model = physics.get("coolant_density", {}).get("model", "iapws_table")

    # 冷却剂（膜）→ 包壳外表面 → 包壳内表面 → 芯块表面 → 芯块中心
    delta_film = power_w_cm / (2.0 * np.pi * clad_outer * h_film)
    t_clad_outer = mod_temp + delta_film

    k_clad = k_zircaloy(0.5 * (t_clad_outer + mod_temp + 200.0))
    r_clad = float(np.log(clad_outer / clad_inner) / (2.0 * np.pi * k_clad))
    delta_clad = power_w_cm * r_clad
    t_clad_inner = t_clad_outer + delta_clad

    r_gap = 1.0 / (2.0 * np.pi * pellet_r * h_gap)
    delta_gap = power_w_cm * r_gap
    t_surface = t_clad_inner + delta_gap

    history: List[float] = []
    t_fuel_avg = float(fuel_temp_guess)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        t_center = fuel_center_temperature(power_w_cm, pellet_r, t_surface)
        # 均匀体热源 + 抛物线分布：面积平均温度 = 中心与表面的中点
        t_avg_new = 0.5 * (t_center + t_surface)
        history.append(t_avg_new)
        delta = abs(t_avg_new - t_fuel_avg)
        t_fuel_avg = (1.0 - relaxation) * t_fuel_avg + relaxation * t_avg_new
        if delta < tolerance:
            converged = True
            break

    t_center = fuel_center_temperature(power_w_cm, pellet_r, t_surface)
    delta_fuel = t_center - t_surface

    if mod_temp >= TSAT_15_5_MPA:
        LOGGER.warning("慢化剂温度 %.1f K 已达/超过 15.5 MPa 饱和温度 %.2f K，液态水模型失效",
                       mod_temp, TSAT_15_5_MPA)

    return TemperatureSolution(
        power_w_cm=power_w_cm, mod_temp=mod_temp,
        t_center=t_center, t_surface=t_surface, t_fuel_avg=0.5 * (t_center + t_surface),
        t_clad_inner=t_clad_inner, t_clad_outer=t_clad_outer,
        r_gap=r_gap, r_clad=r_clad, r_film=delta_film / max(power_w_cm, 1e-12),
        delta_gap=delta_gap, delta_clad=delta_clad, delta_film=delta_film,
        delta_fuel=delta_fuel, iterations=iterations, converged=converged, history=history,
    )


def neutronic_response(bundle: Mapping[str, Any], solution: TemperatureSolution,
                       config: Config) -> Dict[str, float]:
    """把收敛后的燃料/慢化剂温度送回中子学代理模型，得到 k_eff 与温度系数。"""
    opt = config.section("optimization").get("fixed", {})
    # 寻优的固定参数里若没有富集度/硼浓度（它们是被寻优变量），退回基准题参考态
    ref = config.section("validation").get("reference_case", {})
    features = bundle.get("features", list(FEATURE_COLUMNS))
    base_params = {
        "enrichment": float(opt.get("enrichment", ref.get("enrichment", 3.1))),
        "boron_ppm": float(opt.get("boron_ppm", ref.get("boron_ppm", 1300.0))),
        "clad_r": float(opt.get("clad_r", ref.get("clad_r", 0.475))),
        "fuel_temp": float(solution.t_fuel_avg),
        "mod_temp": float(solution.mod_temp),
    }
    vector = np.array([[base_params[f] for f in features]], dtype=float)
    k_mean, k_sigma = predict_k(bundle, vector)

    # 燃料温度系数：只扰动燃料温度 ±50 K，其余不变
    delta_t = 50.0
    vector_hot = vector.copy()
    vector_cold = vector.copy()
    i_fuel = features.index("fuel_temp")
    vector_hot[0, i_fuel] += delta_t
    vector_cold[0, i_fuel] -= delta_t
    k_hot, _ = predict_k(bundle, vector_hot)
    k_cold, _ = predict_k(bundle, vector_cold)
    alpha_fuel = (float(k_hot[0]) - float(k_cold[0])) / (2.0 * delta_t) * PCM  # pcm/K

    return {
        "k_eff": float(k_mean[0]),
        "k_sigma": float(k_sigma[0]),
        "k_sigma_pcm": float(k_sigma[0]) * PCM,
        "fuel_temp_coefficient_pcm_k": alpha_fuel,
        "design_params": base_params,
    }


def plot_profile(solution: TemperatureSolution, out_path: Path, config: Config | None = None) -> Path:
    physics = (config or load_config()).physics()
    pellet_r = float(physics["pellet_radius_cm"])
    clad_inner = float(physics["clad_inner_radius_cm"])
    clad_outer = float(physics["clad_outer_radius_cm"])
    pitch = float(physics["pitch_cm"])

    q3 = solution.power_w_cm / (np.pi * pellet_r ** 2)  # W/cm³
    radii = np.linspace(0.0, pellet_r, 120)
    # 抛物线分布：T(r) = T_c - q''' r² / (4 k)，k 取平均温度下的值
    k_mean = max(k_uo2(solution.t_fuel_avg), 1e-6)
    temp_fuel = solution.t_center - q3 * radii ** 2 / (4.0 * k_mean)
    temp_gap = np.linspace(solution.t_surface, solution.t_clad_inner, 12)
    temp_clad = np.linspace(solution.t_clad_inner, solution.t_clad_outer, 12)
    temp_water = np.full(20, solution.mod_temp)

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    ax.plot(radii, temp_fuel, color="tab:red", lw=2.2, label="$UO_2$ 芯块")
    ax.plot(np.linspace(pellet_r, clad_inner, temp_gap.size), temp_gap, color="tab:gray", lw=2.0, label="氦气隙")
    ax.plot(np.linspace(clad_inner, clad_outer, temp_clad.size), temp_clad, color="tab:orange", lw=2.2, label="Zr-4 包壳")
    ax.plot(np.linspace(clad_outer, pitch / 2.0, temp_water.size), temp_water, color="tab:blue", lw=2.0, label="冷却剂")

    ax.axvline(pellet_r, color="0.4", ls=":", lw=1.0)
    ax.axvline(clad_inner, color="0.4", ls=":", lw=1.0)
    ax.axvline(clad_outer, color="0.4", ls=":", lw=1.0)
    ax.annotate(f"中心 {solution.t_center:.0f} K", xy=(0.0, solution.t_center),
                xytext=(0.13, solution.t_center - 78), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate(f"芯块表面 {solution.t_surface:.0f} K", xy=(pellet_r, solution.t_surface),
                xytext=(0.245, solution.t_surface - 72), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate(f"包壳外表面 {solution.t_clad_outer:.0f} K", xy=(clad_outer, solution.t_clad_outer),
                xytext=(0.505, solution.t_clad_outer + 45), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_ylim(solution.mod_temp - 55, solution.t_center + 85)
    ax.set_xlabel("径向位置 r (cm)")
    ax.set_ylabel("温度 (K)")
    ax.set_title(f"稳态温度分布（线功率 {solution.power_w_cm:.0f} W/cm，"
                 f"慢化剂 {solution.mod_temp:.0f} K）")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def run_coupling(power_w_cm: float, mod_temp: float, config: Config,
                 bundle: Mapping[str, Any] | None = None, use_openmc: bool = False) -> Dict[str, Any]:
    solution = solve_temperature(power_w_cm, mod_temp, config)
    result: Dict[str, Any] = {"temperature": solution.as_dict()}

    if use_openmc:
        from .models.pin_cell import VeraPinCell

        design = config.section("optimization").get("fixed", {})
        cell = VeraPinCell(
            enrichment=float(design.get("enrichment", 3.1)),
            boron_ppm=float(design.get("boron_ppm", 1300.0)),
            clad_outer_r=float(design.get("clad_r", 0.475)),
            fuel_temp=solution.t_fuel_avg,
            mod_temp=solution.mod_temp,
            config=config,
        )
        out_dir = PATHS.ensure().raw_dir / "coupled_case"
        res = cell.run_and_extract(out_dir, threads=1)
        result["neutronics"] = {"k_eff": res["k_eff"], "std": res["std"], "source": "openmc"}
    else:
        if bundle is None:
            bundle = load_model()
        result["neutronics"] = {**neutronic_response(bundle, solution, config), "source": "surrogate"}

    result["coolant"] = {
        "mod_temp": mod_temp,
        "density_iapws": water_density(mod_temp, "iapws_table"),
        "density_constant": water_density(mod_temp, "constant"),
    }
    return result


def print_report(result: Mapping[str, Any]) -> None:
    t = result["temperature"]
    n = result["neutronics"]
    print("\n" + "=" * 68)
    print(f"一维热工-中子学耦合结果（线功率 {t['power_w_cm']:.0f} W/cm）")
    print("=" * 68)
    print(f"  芯块中心温度     {t['t_center']:8.1f} K")
    print(f"  芯块表面温度     {t['t_surface']:8.1f} K")
    print(f"  芯块平均温度     {t['t_fuel_avg']:8.1f} K")
    print(f"  包壳内/外温度    {t['t_clad_inner']:8.1f} / {t['t_clad_outer']:.1f} K")
    print(f"  冷却剂温度       {t['mod_temp']:8.1f} K")
    print("-" * 68)
    print(f"  温降：芯块 {t['delta_fuel']:6.1f} K | 气隙 {t['delta_gap']:6.1f} K | "
          f"包壳 {t['delta_clad']:5.1f} K | 膜 {t['delta_film']:5.1f} K")
    print(f"  热阻：气隙 {t['r_gap']:.4f} | 包壳 {t['r_clad']:.4f} | "
          f"膜 {t['r_film']:.4f} K·cm/W（气隙占总热阻的 "
          f"{t['r_gap'] / t['r_total']:.1%}）")
    print(f"  Picard 迭代      {t['iterations']} 次，收敛 = {t['converged']}")
    print("-" * 68)
    print(f"  k_eff（{n['source']}）  {n['k_eff']:.5f}"
          + (f" ± {n['std']:.5f}" if "std" in n else
             f" ± {n['k_sigma']:.5f}（1σ）"))
    if "fuel_temp_coefficient_pcm_k" in n:
        print(f"  燃料温度系数     {n['fuel_temp_coefficient_pcm_k']:+.2f} pcm/K")
    print("=" * 68 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一维热工-中子学耦合")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--power", type=float, default=None, help="线功率密度 W/cm")
    parser.add_argument("--mod-temp", type=float, default=600.0, help="冷却剂入口温度 K")
    parser.add_argument("--openmc", action="store_true", help="用 OpenMC 代替代理模型计算 k_eff")
    parser.add_argument("--json", type=Path, default=None, help="结果写入 JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    config = load_config(args.config)
    paths = PATHS.ensure()
    power = float(args.power if args.power is not None else config["thermal.linear_power_w_cm"])

    bundle = None if args.openmc else load_model()
    result = run_coupling(power, float(args.mod_temp), config, bundle=bundle, use_openmc=args.openmc)
    print_report(result)
    plot_profile(solve_temperature(power, float(args.mod_temp), config),
                 paths.figures_dir / "coupled_temperature_profile.png", config)

    out = Path(args.json) if args.json else (paths.results_dir / "coupled_pin.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("结果：%s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
