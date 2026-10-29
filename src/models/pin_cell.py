"""VERA P1 单栅元（PWR pin cell）物理模型。

相对历史版本修复的物理问题（每一项都会改变 k_eff，量级见 README）：

1. **芯块-包壳气隙**：燃料半径取芯块半径 0.4096 cm，包壳内半径 0.4180 cm，
   两者之间 84 μm 的环隙填充氦气。旧版把燃料圆柱半径写成包壳内半径，
   等于让燃料"长进"气隙里，燃料体积偏大 4.2 %。
2. **硼浓度**：以质量分数（``percent_type='wo'``）加入。旧版用默认的原子分数，
   实际硼含量只有标称值的 ~1/1.66，与"硼微分价值只有物理值的一半"的观测一致。
3. **Zircaloy-4 组分**：Sn 1.45 / Fe 0.21 / Cr 0.10 / O 0.125 wt%，Zr 余量。
   旧版 Zr 0.98 + Sn 0.015 用原子分数、总和 0.995，成分和归一化都不对。
4. **慢化剂密度**：改为 IAPWS-IF97 表内插（15.5 MPa），并保留常数模型以复现
   基准题规定的 0.743 g/cm³ 参考态。
5. **可复现性**：每个算例写入确定的随机种子；``batches/inactive/particles``
   来自 ``configs/params.yaml``，不再硬编码。
6. **结果读取**：用 :class:`openmc.StatePoint` 读 ``k_eff``，不再直接读
   ``k_combined`` 这个内部数组；statepoint 文件名由 ``batches`` 决定，
   不再用"最新修改时间"猜文件。

OpenMC 是 C++/Python 混合程序，不能用 pip 安装；本模块在函数内部惰性导入，
因此训练、敏感性分析与单元测试可以在没有 OpenMC 的机器上运行。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..config import Config, load_config
from ..physics import (
    PHYSICS_VERSION,
    boron_mass_fraction,
    validate_pin_geometry,
    water_density,
)

#: VERA P1 的标称几何与材料（cm、g/cm³）
DEFAULT_PHYSICS: Dict[str, Any] = {
    "pellet_radius_cm": 0.4096,
    "clad_inner_radius_cm": 0.4180,
    "clad_outer_radius_cm": 0.4750,
    "pitch_cm": 1.26,
    "fuel_density_g_cm3": 10.257,
    "clad_density_g_cm3": 6.56,
    "gap_density_g_cm3": 8.13e-5,  # 1 atm He @ 600 K
    "coolant_density": {"model": "iapws_table", "value": 0.743},
    "zircaloy_wt_pct": {"Sn": 1.45, "Fe": 0.21, "Cr": 0.10, "O": 0.125},
}

#: 原子质量（g/mol），用于把富集度从 wt% 换算成显式的核素质量分数
_M_U235 = 235.043930
_M_U238 = 238.050788
_M_O16 = 15.999405


def _openmc():
    """惰性导入 OpenMC，缺失时给出可操作的安装提示。"""
    try:
        import openmc  # noqa: F401  (局部导入，避免无 OpenMC 环境下 import 失败)
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise RuntimeError(
            "未找到 openmc 模块。OpenMC 是 C++/Python 混合程序，不能用 pip 安装：\n"
            "  conda install -c conda-forge openmc\n"
            "  或参考 https://docs.openmc.org/en/stable/quickinstall.html 源码编译\n"
            "训练代理模型、Sobol 敏感性分析和 Web 演示都不需要 OpenMC。"
        ) from exc
    return openmc


def uo2_weight_fractions(enrichment_wt_pct: float) -> Dict[str, float]:
    """把 U-235 质量富集度 [wt%] 换算成 UO2 中各核素的质量分数。

    显式给出核素质量分数，避免 ``add_element(enrichment=...)`` 在
    "原子百分比 / 质量百分比"上的歧义 —— 基准题给的是 wt%。
    """
    if not 0.0 < enrichment_wt_pct < 100.0:
        raise ValueError(f"富集度必须在 (0, 100) wt% 之间：{enrichment_wt_pct}")
    e = enrichment_wt_pct / 100.0
    m_u = e * _M_U235 + (1.0 - e) * _M_U238
    m_uo2 = m_u + 2.0 * _M_O16
    return {
        "U235": e * m_u / m_uo2,
        "U238": (1.0 - e) * m_u / m_uo2,
        "O16": 2.0 * _M_O16 / m_uo2,
    }


class VeraPinCell:
    """参数化 VERA P1 栅元。

    Parameters
    ----------
    enrichment : U-235 富集度 [wt%]
    boron_ppm : 冷却剂硼浓度 [ppm，按溶液质量]
    clad_outer_r : 包壳外半径 [cm]；包壳厚度保持标称值，内半径随之平移
    fuel_temp : 燃料温度 [K]（用于截面温度插值）
    mod_temp : 慢化剂温度 [K]
    config : 配置对象，缺省读取 ``configs/params.yaml``
    case_index : 算例编号，用于派生随机种子
    """

    def __init__(
        self,
        enrichment: float = 3.1,
        boron_ppm: float = 1300.0,
        clad_outer_r: Optional[float] = None,
        fuel_temp: float = 900.0,
        mod_temp: float = 600.0,
        *,
        config: Optional[Config] = None,
        case_index: int = 0,
    ) -> None:
        self.config = config or load_config()
        physics = {**DEFAULT_PHYSICS, **self.config.physics()}
        self.physics: Dict[str, Any] = physics

        self.enrichment = float(enrichment)
        self.boron_ppm = float(boron_ppm)
        self.fuel_temp = float(fuel_temp)
        self.mod_temp = float(mod_temp)
        self.case_index = int(case_index)

        self.pellet_r = float(physics["pellet_radius_cm"])
        self.clad_outer_r = float(
            physics["clad_outer_radius_cm"] if clad_outer_r is None else clad_outer_r
        )
        # 包壳厚度保持标称值，内半径随外半径平移（这样 clad_outer_r 才是真正的设计变量）
        self.clad_thickness = float(physics["clad_outer_radius_cm"]) - float(
            physics["clad_inner_radius_cm"]
        )
        self.clad_inner_r = self.clad_outer_r - self.clad_thickness
        self.pitch = float(physics["pitch_cm"])

        self.problems = validate_pin_geometry(
            self.pellet_r, self.clad_inner_r, self.clad_outer_r, self.pitch
        )
        if self.problems:
            raise ValueError(
                "栅元几何不合法（case %d）：%s" % (self.case_index, "；".join(self.problems))
            )

        self.materials: Any = None
        self.geometry: Any = None
        self.settings: Any = None

    # ------------------------------------------------------------------ 材料
    @property
    def coolant_density(self) -> float:
        cool = self.physics.get("coolant_density", {}) or {}
        return water_density(
            self.mod_temp,
            model=cool.get("model", "iapws_table"),
            constant=cool.get("value"),
        )

    def _build_materials(self) -> Any:
        openmc = _openmc()
        p = self.physics

        # 1. 燃料 UO2（显式质量分数，避免富集度的百分比歧义）
        fuel = openmc.Material(name="UO2", temperature=self.fuel_temp)
        for name, frac in uo2_weight_fractions(self.enrichment).items():
            fuel.add_nuclide(name, frac, percent_type="wo")
        fuel.set_density("g/cm3", float(p["fuel_density_g_cm3"]))

        # 2. 气隙（1 atm 氦气，对中子学影响可忽略，但保持几何闭合）
        gap = openmc.Material(name="Helium gap", temperature=self.fuel_temp)
        gap.add_element("He", 1.0)
        gap.set_density("g/cm3", float(p["gap_density_g_cm3"]))

        # 3. 包壳 Zircaloy-4（真实 wt%，Zr 取余量）
        zirc = dict(p.get("zircaloy_wt_pct", {}))
        clad = openmc.Material(name="Zircaloy-4", temperature=self.mod_temp)
        zr_wt = 100.0 - sum(zirc.values())
        if zr_wt <= 0:
            raise ValueError(f"Zircaloy 合金元素质量分数之和超过 100 %：{zirc}")
        clad.add_element("Zr", zr_wt, percent_type="wo")
        for element, wt in zirc.items():
            clad.add_element(element, float(wt), percent_type="wo")
        clad.set_density("g/cm3", float(p["clad_density_g_cm3"]))

        # 4. 含硼水：H2O 按原子比，硼按**质量分数**加入
        water = openmc.Material(name="Borated water", temperature=self.mod_temp)
        water.add_element("H", 2.0)
        water.add_element("O", 1.0)
        water.set_density("g/cm3", self.coolant_density)
        water.add_s_alpha_beta("c_H_in_H2O")
        if self.boron_ppm > 0.0:
            water.add_element("B", boron_mass_fraction(self.boron_ppm), percent_type="wo")

        self.materials = openmc.Materials([fuel, gap, clad, water])
        return self.materials

    # ------------------------------------------------------------------ 几何
    def _build_geometry(self) -> Any:
        openmc = _openmc()
        pellet_cyl = openmc.ZCylinder(r=self.pellet_r, name="pellet OD")
        clad_inner_cyl = openmc.ZCylinder(r=self.clad_inner_r, name="clad ID")
        clad_outer_cyl = openmc.ZCylinder(r=self.clad_outer_r, name="clad OD")

        fuel_cell = openmc.Cell(name="Fuel", fill=self.materials[0], region=-pellet_cyl)
        gap_cell = openmc.Cell(
            name="Gap", fill=self.materials[1], region=+pellet_cyl & -clad_inner_cyl
        )
        clad_cell = openmc.Cell(
            name="Clad", fill=self.materials[2], region=+clad_inner_cyl & -clad_outer_cyl
        )
        mod_cell = openmc.Cell(name="Moderator", fill=self.materials[3], region=+clad_outer_cyl)

        half = self.pitch / 2.0
        x_min = openmc.XPlane(-half, boundary_type="reflective")
        x_max = openmc.XPlane(half, boundary_type="reflective")
        y_min = openmc.YPlane(-half, boundary_type="reflective")
        y_max = openmc.YPlane(half, boundary_type="reflective")

        universe = openmc.Universe(cells=[fuel_cell, gap_cell, clad_cell, mod_cell])
        root_cell = openmc.Cell(
            name="Root", fill=universe, region=+x_min & -x_max & +y_min & -y_max
        )
        self.geometry = openmc.Geometry([root_cell])
        return self.geometry

    # ------------------------------------------------------------------ 设置
    def _build_settings(self) -> Any:
        openmc = _openmc()
        run = self.config.openmc_run()
        settings = openmc.Settings()
        settings.run_mode = "eigenvalue"
        settings.batches = int(run.get("batches", 100))
        settings.inactive = int(run.get("inactive", 20))
        settings.particles = int(run.get("particles", 10000))
        settings.seed = int(run.get("seed", 1)) + self.case_index
        # 温度相关截面用插值，避免每个温度都要生成专用的 HDF5 截面库
        settings.temperature = {"method": "interpolation"}
        settings.output = {"tallies": False}

        source = openmc.IndependentSource()
        source.space = openmc.stats.Point((0.0, 0.0, 0.0))
        settings.source = source

        self.settings = settings
        return settings

    # -------------------------------------------------------------- 对外接口
    def build_model(self) -> Any:
        """按当前参数构建 :class:`openmc.model.Model`（不运行）。"""
        openmc = _openmc()
        self._build_materials()
        self._build_geometry()
        self._build_settings()
        return openmc.model.Model(
            geometry=self.geometry,
            materials=self.materials,
            settings=self.settings,
        )

    def describe(self) -> Dict[str, Any]:
        """返回本次计算的关键参数，用于 dry-run 打印与日志。"""
        return {
            "case_index": self.case_index,
            "enrichment": self.enrichment,
            "boron_ppm": self.boron_ppm,
            "clad_inner_r": round(self.clad_inner_r, 4),
            "clad_outer_r": round(self.clad_outer_r, 4),
            "pellet_r": self.pellet_r,
            "gap_cm": round(self.clad_inner_r - self.pellet_r, 5),
            "pitch_cm": self.pitch,
            "fuel_temp": self.fuel_temp,
            "mod_temp": self.mod_temp,
            "coolant_density": round(self.coolant_density, 4),
            "batches": self.config["openmc_run.batches"],
            "particles": self.config["openmc_run.particles"],
            "physics_version": PHYSICS_VERSION,
        }

    def run_and_extract(
        self,
        output_dir: Path | str,
        threads: int = 1,
        clean: bool = True,
    ) -> Dict[str, Any]:
        """运行 OpenMC 并返回 ``k_eff`` 及其统计误差。

        ``clean=True`` 会先清掉目标目录里旧的 ``statepoint.*.h5``，
        避免像旧版那样在 glob 结果里误取上一次运行留下的文件。
        """
        openmc = _openmc()
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if clean:
            for stale in out_dir.glob("statepoint.*.h5"):
                stale.unlink()

        model = self.build_model()
        model.run(cwd=str(out_dir), threads=int(threads), output=False)

        sp_path = out_dir / f"statepoint.{self.settings.batches}.h5"
        if not sp_path.is_file():
            found = sorted(out_dir.glob("statepoint.*.h5"))
            raise FileNotFoundError(
                f"运行结束但找不到 {sp_path.name}（目录内现有：{[p.name for p in found]}）"
            )

        with openmc.StatePoint(str(sp_path)) as sp:
            k = sp.keff
            return {
                "k_eff": float(k.nominal_value),
                "std": float(k.std_dev),
                "statepoint": str(sp_path),
                "batches": int(self.settings.batches),
                "particles": int(self.settings.particles),
                "seed": int(self.settings.seed),
            }


def describe_case(params: Mapping[str, float], config: Optional[Config] = None) -> Dict[str, Any]:
    """不依赖 OpenMC 地构造一个 :class:`VeraPinCell` 并返回其参数摘要（dry-run 用）。"""
    cell = VeraPinCell(
        enrichment=params["enrichment"],
        boron_ppm=params["boron_ppm"],
        clad_outer_r=params["clad_r"],
        fuel_temp=params["fuel_temp"],
        mod_temp=params["mod_temp"],
        config=config,
    )
    return cell.describe()
