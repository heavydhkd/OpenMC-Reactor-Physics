"""集中式配置与路径管理。

设计原则：**任何产物都不依赖当前工作目录（CWD）**。
仓库根目录由本文件位置反推（``src/config.py`` 的上一级），因此从仓库根目录、
``src/`` 目录或任意其它目录调用脚本，输出都会落在仓库内固定的位置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
    raise SystemExit("缺少 PyYAML，请先执行 `pip install pyyaml`") from exc

# src/config.py -> 仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "params.yaml"

#: 代理模型的输入特征（顺序即模型输入顺序，也是 CSV 列顺序）
FEATURE_COLUMNS: Tuple[str, ...] = (
    "enrichment",
    "boron_ppm",
    "clad_r",
    "fuel_temp",
    "mod_temp",
)

#: 特征的中文名与单位，用于绘图与前端展示
FEATURE_LABELS: Dict[str, str] = {
    "enrichment": "U-235 富集度 (%)",
    "boron_ppm": "硼浓度 (ppm)",
    "clad_r": "包壳外半径 (cm)",
    "fuel_temp": "燃料温度 (K)",
    "mod_temp": "慢化剂温度 (K)",
}

#: 数据集的完整列（顺序固定，失败算例也保留全部列）
DATASET_COLUMNS: Tuple[str, ...] = (
    "case_id",
    "enrichment",
    "boron_ppm",
    "clad_r",
    "fuel_temp",
    "mod_temp",
    "k_eff",
    "std",
    "status",
    "physics_version",
)


@dataclass(frozen=True)
class Paths:
    """仓库内所有产物的固定位置。"""

    root: Path = REPO_ROOT
    raw_dir: Path = REPO_ROOT / "data" / "raw"
    processed_dir: Path = REPO_ROOT / "data" / "processed"
    results_dir: Path = REPO_ROOT / "results"
    figures_dir: Path = REPO_ROOT / "results" / "figures"
    logs_dir: Path = REPO_ROOT / "results" / "logs"

    @property
    def dataset_csv(self) -> Path:
        return self.processed_dir / "vera_p1_dataset.csv"

    @property
    def cases_jsonl(self) -> Path:
        """逐算例增量落盘文件，用于断点续跑。"""
        return self.processed_dir / "cases.jsonl"

    @property
    def metrics_json(self) -> Path:
        return self.results_dir / "metrics.json"

    @property
    def model_joblib(self) -> Path:
        return self.results_dir / "surrogate.joblib"

    def ensure(self) -> "Paths":
        for d in (self.raw_dir, self.processed_dir, self.results_dir,
                  self.figures_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths()


class Config:
    """对 ``configs/params.yaml`` 的只读包装，支持点号路径访问。

    >>> cfg = Config.load()
    >>> cfg["openmc_run.batches"]
    100
    >>> cfg.bounds()["boron_ppm"]
    (0.0, 3000.0)
    """

    def __init__(self, data: Mapping[str, Any], path: Path | str | None = None):
        self._data: Dict[str, Any] = dict(data)
        self.path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    # ------------------------------------------------------------------ 构造
    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not cfg_path.is_file():
            raise FileNotFoundError(f"找不到配置文件：{cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"配置文件格式错误（顶层应为映射）：{cfg_path}")
        return cls(data, cfg_path)

    def as_dict(self) -> Dict[str, Any]:
        return self._data

    def __getitem__(self, dotted: str) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise KeyError(f"配置项不存在：{dotted!r}（在 {self.path}）")
            node = node[part]
        return node

    def get(self, dotted: str, default: Any = None) -> Any:
        try:
            return self[dotted]
        except KeyError:
            return default

    def __contains__(self, dotted: str) -> bool:
        try:
            self[dotted]
        except KeyError:
            return False
        return True

    def section(self, name: str) -> Dict[str, Any]:
        value = self.get(name, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"配置节 {name!r} 应为映射，实际是 {type(value).__name__}")
        return dict(value)

    # ------------------------------------------------------------ 常用视图
    def features(self) -> List[str]:
        """模型使用的特征列，缺省为全部 5 维。"""
        feats = list(self.get("training.features", FEATURE_COLUMNS))
        unknown = [f for f in feats if f not in FEATURE_COLUMNS]
        if unknown:
            raise ValueError(f"未知特征 {unknown}，可选值：{list(FEATURE_COLUMNS)}")
        return feats

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """采样/设计域边界，键为特征列名。"""
        raw = self["sampling.bounds"]
        out: Dict[str, Tuple[float, float]] = {}
        for name in FEATURE_COLUMNS:
            if name not in raw:
                raise KeyError(f"sampling.bounds 缺少 {name!r}")
            lo, hi = raw[name]
            lo, hi = float(lo), float(hi)
            if not lo < hi:
                raise ValueError(f"sampling.bounds.{name} 上界必须大于下界：{lo} !< {hi}")
            out[name] = (lo, hi)
        return out

    def design_bounds(self) -> Dict[str, Tuple[float, float]]:
        """逆向寻优的变量边界（通常只有富集度与硼浓度）。"""
        raw = self["optimization.bounds"]
        return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}

    def physics(self) -> Dict[str, Any]:
        return self.section("physics")

    def openmc_run(self) -> Dict[str, Any]:
        return self.section("openmc_run")

    # ---------------------------------------------------------------- 工具
    def resolve(self, p: str | os.PathLike[str]) -> Path:
        """把可能的相对路径按仓库根目录解析，绝对路径原样返回。"""
        path = Path(p)
        return path if path.is_absolute() else (REPO_ROOT / path)


def load_config(path: Path | str | None = None) -> Config:
    """读取配置；``path`` 为空时使用 ``configs/params.yaml``。"""
    return Config.load(path)
