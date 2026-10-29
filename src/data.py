"""数据集读写：固定列结构、失败算例保留全部字段、JSONL 增量落盘。

历史版本的失败分支只返回 ``{'case_id': ..., 'status': ...}``，``pandas`` 会因此
得到一行"缺列"的记录，后续 ``df[['enrichment', ...]]`` 直接报 KeyError。
这里统一用 :data:`src.config.DATASET_COLUMNS` 生成记录，失败行也带全字段
（数值列填 NaN），从而保证任何时刻的 CSV 都是规整的矩形表。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from .config import DATASET_COLUMNS, Paths

NUMERIC_COLUMNS = ("enrichment", "boron_ppm", "clad_r", "fuel_temp", "mod_temp", "k_eff", "std")


def case_record(
    case_id: int,
    params: Mapping[str, float],
    k_eff: Optional[float] = None,
    std: Optional[float] = None,
    status: str = "success",
    physics_version: str = "unknown",
) -> Dict[str, Any]:
    """构造一条完整的数据集记录（缺省数值为 NaN 而非缺字段）。"""
    record: Dict[str, Any] = {
        "case_id": int(case_id),
        "physics_version": physics_version,
        "status": str(status),
        "k_eff": float(k_eff) if k_eff is not None else math.nan,
        "std": float(std) if std is not None else math.nan,
    }
    for col in ("enrichment", "boron_ppm", "clad_r", "fuel_temp", "mod_temp"):
        value = params.get(col)
        record[col] = float(value) if value is not None else math.nan
    return {col: record[col] for col in DATASET_COLUMNS}


def load_dataset(
    path: Path | str,
    drop_failed: bool = True,
    require_min_rows: int = 1,
) -> pd.DataFrame:
    """读取数据集 CSV，规整列名与类型。

    缺少 ``physics_version`` 列的旧文件会被标记为 ``"legacy"``，
    便于在 README/日志里区分"旧几何生成的数据"与当前物理模型生成的数据。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到数据集 {path}\n"
            f"请先运行 `python -m src.run_batch`（需要 OpenMC），"
            f"或使用仓库自带的 data/processed/vera_p1_dataset.csv"
        )
    df = pd.read_csv(path)
    if "status" not in df.columns:
        raise ValueError(f"{path} 缺少 status 列，无法判断算例是否有效")

    if "physics_version" not in df.columns:
        df["physics_version"] = "legacy"
    for col in DATASET_COLUMNS:
        if col not in df.columns:
            df[col] = math.nan

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["status"] = df["status"].astype(str)
    df = df[list(DATASET_COLUMNS)]

    if drop_failed:
        df = df[df["status"].str.startswith("success")].copy()
        df = df.dropna(subset=list(NUMERIC_COLUMNS))

    df = df.sort_values("case_id").reset_index(drop=True)
    if len(df) < require_min_rows:
        raise ValueError(
            f"{path} 中有效算例只有 {len(df)} 条（要求至少 {require_min_rows} 条）"
        )
    return df


def save_dataset(df: pd.DataFrame, path: Path | str) -> Path:
    """按固定列顺序写出 CSV，并返回绝对路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [c for c in DATASET_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    df[ordered + extra].to_csv(path, index=False, float_format="%.8g")
    return path.resolve()


def append_case(record: Mapping[str, Any], path: Path | str) -> None:
    """向 JSONL 追加一条算例记录（逐条 flush，进程被杀也不丢已完成的结果）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        fh.flush()


def load_cases(path: Path | str) -> Dict[int, Dict[str, Any]]:
    """读取 JSONL，返回 ``case_id -> record``（后写入者覆盖先写入者）。"""
    path = Path(path)
    if not path.is_file():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # 进程中断可能留下半行，直接跳过
            if "case_id" in rec:
                out[int(rec["case_id"])] = rec
    return out


def records_to_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """把记录列表整理成规整 DataFrame（按 case_id 排序）。"""
    rows = [dict(rec) for rec in records]
    if not rows:
        return pd.DataFrame(columns=list(DATASET_COLUMNS))
    df = pd.DataFrame(rows)
    for col in DATASET_COLUMNS:
        if col not in df.columns:
            df[col] = math.nan
    df = df[list(DATASET_COLUMNS)].sort_values("case_id").reset_index(drop=True)
    return df


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (math.nan, float)) and math.isnan(float(obj)):
        return None
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"无法序列化 {type(obj).__name__}")
