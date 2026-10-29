"""把代理模型的结论拿回 OpenMC 复核。

这个脚本存在的意义：代理模型的预测不能自己给自己打分。凡是"寻优给出一个
临界设计"这类结论，都必须用 OpenMC 原样算一遍，并和基准题参考值对照。

    python -m src.validate --check          # 不需要 OpenMC：只校验几何/理化关系是否自洽
    python -m src.validate                  # 跑基准态（3.1 wt%、1300 ppm、600 K）并与参考解对照
    python -m src.validate --design results/optimization.json   # 复核寻优得到的设计点

需要 OpenMC（conda install -c conda-forge openmc，并下载 HDF5 截面库）。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .config import PATHS, Config, REPO_ROOT, load_config
from .physics import (
    BENCHMARK_COOLANT_DENSITY,
    expected_b_h2o_ratio,
    iapws_water_density,
    saturation_temperature,
    validate_pin_geometry,
    water_density,
)

LOGGER = logging.getLogger(__name__)
OPERATING_PRESSURE_MPA = 15.5


def check_physics(config: Config) -> Dict[str, Any]:
    """不需要 OpenMC 的一致性检查：几何、饱和温度、硼浓度换算。"""
    physics = config.physics()
    pellet_r = float(physics["pellet_radius_cm"])
    clad_inner = float(physics["clad_inner_radius_cm"])
    clad_outer = float(physics["clad_outer_radius_cm"])
    pitch = float(physics["pitch_cm"])

    problems: List[str] = validate_pin_geometry(pellet_r, clad_inner, clad_outer, pitch)

    # 采样边界里的每个角点都必须几何可行
    bounds = config.bounds()
    for clad_r in (bounds["clad_r"][0], bounds["clad_r"][1]):
        inner = float(clad_r) - (clad_outer - clad_inner)
        corner_problems = validate_pin_geometry(pellet_r, inner, float(clad_r), pitch)
        problems.extend(f"clad_r={clad_r}: {p}" for p in corner_problems)

    # 慢化剂温度上限必须低于饱和温度，否则液态水模型失效
    mod_max = float(bounds["mod_temp"][1])
    tsat = saturation_temperature(OPERATING_PRESSURE_MPA)
    if mod_max >= tsat:
        problems.append(f"mod_temp 上限 {mod_max:.1f} K ≥ 15.5 MPa 饱和温度 {tsat:.2f} K")

    # 硼浓度换算：质量分数必须与"每 10⁶ 质量份水含硼 X 份"一致
    boron_ppm = float(bounds["boron_ppm"][1])
    rho_bench = BENCHMARK_COOLANT_DENSITY
    ratio = expected_b_h2o_ratio(boron_ppm, rho_bench)

    return {
        "ok": not problems,
        "problems": problems,
        "physics_version": config.get("meta.physics_version", "?"),
        "saturation_temperature_k": tsat,
        "mod_temp_max_k": mod_max,
        "gap_cm": clad_inner - pellet_r,
        "clad_thickness_cm": clad_outer - clad_inner,
        "water_density_iapws_600k": iapws_water_density(600.0),
        "water_density_benchmark": water_density(600.0, "constant"),
        "boron_max_ppm": boron_ppm,
        "b_h2o_ratio_at_max": ratio,
    }


def verify_design(params: Mapping[str, float], config: Config,
                  paths: Any = None) -> Dict[str, Any]:
    """用 OpenMC 复核一个设计点，并与代理模型预测、参考解对照。"""
    from .models.pin_cell import VeraPinCell  # 延迟导入，避免无 OpenMC 时也报错

    paths = paths or PATHS.ensure()
    design = {
        "enrichment": float(params["enrichment"]),
        "boron_ppm": float(params["boron_ppm"]),
        "clad_outer_r": float(params.get("clad_r", config["optimization.fixed.clad_r"])),
        "fuel_temp": float(params.get("fuel_temp", config["optimization.fixed.fuel_temp"])),
        "mod_temp": float(params.get("mod_temp", config["optimization.fixed.mod_temp"])),
    }
    cell = VeraPinCell(
        enrichment=design["enrichment"],
        boron_ppm=design["boron_ppm"],
        clad_outer_r=design["clad_outer_r"],
        fuel_temp=design["fuel_temp"],
        mod_temp=design["mod_temp"],
        config=config,
    )
    out_dir = paths.raw_dir / "verify_design"
    result = cell.run_and_extract(out_dir, threads=1)

    report: Dict[str, Any] = {"design": design, **result}
    reference = config["validation.reference_k_inf"]
    if reference is not None:
        report["reference_k_inf"] = float(reference)
        report["delta_vs_reference_pcm"] = (result["k_eff"] - float(reference)) * 1e5
        report["delta_over_sigma"] = abs(report["delta_vs_reference_pcm"]) / max(result["std"] * 1e5, 1e-9)
    LOGGER.info("OpenMC 复核：k_eff = %.5f ± %.5f（%s）", result["k_eff"], result["std"], out_dir)
    return report


def run_reference_case(config: Config, paths: Any = None) -> Dict[str, Any]:
    """跑基准态并与文献参考值对照。"""
    ref = config.section("validation").get("reference_case", {})
    design = {
        "enrichment": float(ref.get("enrichment", 3.1)),
        "boron_ppm": float(ref.get("boron_ppm", 1300.0)),
        "clad_r": float(ref.get("clad_r", 0.475)),
        "fuel_temp": float(ref.get("fuel_temp", 900.0)),
        "mod_temp": float(ref.get("mod_temp", 600.0)),
    }
    report = verify_design(design, config, paths=paths)
    reference = float(config["validation.reference_k_inf"])
    diff = (report["k_eff"] - reference) * 1e5
    verdict = "一致" if abs(diff) <= 3.0 * report["std"] * 1e5 else "不一致（需排查）"
    print("\n" + "=" * 72)
    print("基准题复核：VERA P1 栅元")
    print("=" * 72)
    for key, value in design.items():
        print(f"  {key:<14} = {value:.4f}")
    print("-" * 72)
    print(f"  OpenMC 计算     = {report['k_eff']:.5f} ± {report['std']:.5f}")
    print(f"  文献参考值       = {reference:.5f}")
    print(f"  偏差            = {diff:+.0f} pcm（{abs(diff) / max(report['std'] * 1e5, 1e-9):.2f}σ）→ {verdict}")
    print("=" * 72 + "\n")
    report["verdict"] = verdict
    return report


def print_check_report(report: Mapping[str, Any], config) -> None:
    """把自检结果打印成人类可读的几行（机械可读的 JSON 用 --json）。"""
    physics = config.physics()
    ok = bool(report["ok"])
    mark = "✓" if ok else "✗"
    print(f"[配置自检] physics_version = {report.get('physics_version', '?')}")
    print(f"  几何：芯块 {physics['pellet_radius_cm']:.4f} / 包壳内 {physics['clad_inner_radius_cm']:.4f}"
          f" / 包壳外 {physics['clad_outer_radius_cm']:.4f} / 栅距 {physics['pitch_cm']:.2f} cm  {mark}")
    print(f"  气隙：{report['gap_cm']:.4f} cm = {report['gap_cm'] * 1e4:.1f} μm  {mark}")
    print(f"  慢化剂：上限 {report['mod_temp_max_k']:.1f} K < 饱和温度 "
          f"{report['saturation_temperature_k']:.2f} K @15.5 MPa  {mark}")
    print(f"  冷却剂密度：IAPWS 600 K = {report['water_density_iapws_600k']:.4f} g/cm³；"
          f"基准题 {report['water_density_benchmark']:.3f} g/cm³（两者不可混用）")
    print(f"  硼浓度：上限 {report['boron_max_ppm']:.0f} ppm ↔ B/H2O = "
          f"{report['b_h2o_ratio_at_max']:.4e}（质量分数换算）  {mark}")
    if report["problems"]:
        print("  问题：")
        for item in report["problems"]:
            print(f"    - {item}")
    print(f"结论：{'通过' if ok else '未通过'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用 OpenMC 复核代理模型的结论")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="只做几何/理化一致性检查（不需要 OpenMC）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（便于 CI 解析）")
    parser.add_argument("--design", type=Path, default=None, help="optimization.json 或含 optimum 的 JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    config = load_config(args.config)
    paths = PATHS.ensure()

    if args.check:
        report = check_physics(config)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print_check_report(report, config)
        if not report["ok"]:
            LOGGER.error("配置存在问题：%s", report["problems"])
            return 1
        return 0

    if args.design:
        payload = json.loads(Path(args.design).read_text(encoding="utf-8"))
        design = payload.get("optimum", payload)
        report = verify_design(design, config, paths=paths)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    report = run_reference_case(config, paths=paths)
    out = paths.results_dir / "validation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("结果：%s", out)
    return 0 if report.get("verdict") == "一致" else 2


if __name__ == "__main__":
    raise SystemExit(main())
