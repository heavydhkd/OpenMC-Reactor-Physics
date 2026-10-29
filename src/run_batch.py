"""LHS 采样 + 多进程批量驱动 OpenMC 计算。

相对历史版本的修复：

* **路径**：所有路径由 :mod:`src.config` 基于仓库根目录生成，不再出现
  ``"../data/raw/case_003"`` 这种随当前工作目录漂移的相对路径
  （旧版从仓库根目录运行时，statepoint 会被写到仓库外面）。
* **并发**：进程数 = ``CPU 核数 // threads_per_case``，每个 OpenMC 算例显式
  指定线程数，避免默认 ``cpu-1`` 个进程、每个进程再各开满线程的超订。
* **数据表结构**：失败算例也带全部列（数值为 NaN），不会产生"缺列"的行。
* **断点续跑**：每个算例完成后立刻追加写入 ``data/processed/cases.jsonl``，
  ``--resume`` 时跳过已成功的算例。
* **随机种子**：``seed + case_id``，同一份配置重跑得到同一批随机数。

用法::

    python -m src.run_batch --dry-run      # 只打印采样计划，不需要 OpenMC
    python -m src.run_batch                # 正式跑批
    python -m src.run_batch --resume       # 中断后续跑
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.stats import qmc

from .config import FEATURE_COLUMNS, PATHS, Config, Paths, load_config
from .data import append_case, case_record, load_cases, records_to_frame, save_dataset
from .models.pin_cell import VeraPinCell
from .physics import PHYSICS_VERSION

LOGGER = logging.getLogger("run_batch")


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def latin_hypercube(
    n_samples: int,
    bounds: Mapping[str, Sequence[float]],
    seed: int,
    features: Sequence[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    """在给定边界内做拉丁超立方采样，返回 ``(n_samples, len(features))`` 数组。"""
    if n_samples < 1:
        raise ValueError(f"样本数必须为正：{n_samples}")
    lower = np.array([float(bounds[f][0]) for f in features])
    upper = np.array([float(bounds[f][1]) for f in features])
    sampler = qmc.LatinHypercube(d=len(features), seed=int(seed))
    return qmc.scale(sampler.random(n=int(n_samples)), lower, upper)


def _params(row: Sequence[float], features: Sequence[str] = FEATURE_COLUMNS) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(features, row)}


# ---------------------------------------------------------------------------
# 单个算例（必须定义在模块顶层，才能被 ProcessPoolExecutor pickle）
# ---------------------------------------------------------------------------
def run_case(
    case_id: int,
    params: Mapping[str, float],
    config: Config,
    raw_dir: str,
    threads: int,
) -> Dict[str, Any]:
    """跑一个栅元算例，返回一条完整的数据集记录（失败时 succuss 标记为 failed）。"""
    cell = None
    try:
        cell = VeraPinCell(
            enrichment=params["enrichment"],
            boron_ppm=params["boron_ppm"],
            clad_outer_r=params["clad_r"],
            fuel_temp=params["fuel_temp"],
            mod_temp=params["mod_temp"],
            config=config,
            case_index=case_id,
        )
        out_dir = Path(raw_dir) / f"case_{case_id:03d}"
        result = cell.run_and_extract(out_dir, threads=threads)
        return case_record(
            case_id=case_id,
            params=params,
            k_eff=result["k_eff"],
            std=result["std"],
            status="success",
            physics_version=PHYSICS_VERSION,
        )
    except Exception as exc:  # 单算例失败不应中断整个批次
        LOGGER.warning("case %03d 失败：%s: %s", case_id, type(exc).__name__, exc)
        record = case_record(
            case_id=case_id,
            params=params,
            status=f"failed: {type(exc).__name__}: {exc}",
            physics_version=PHYSICS_VERSION,
        )
        record["_traceback"] = True  # 仅用于日志提示，不写入 CSV
        return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VERA P1 栅元批量计算（LHS + 多进程）")
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径")
    parser.add_argument("--cases", type=int, default=None, help="覆盖 sampling.n_samples")
    parser.add_argument("--workers", type=int, default=None, help="并行进程数")
    parser.add_argument("--threads", type=int, default=None, help="每个算例的 OpenMC 线程数")
    parser.add_argument("--resume", action="store_true", help="跳过 cases.jsonl 中已成功的算例")
    parser.add_argument("--dry-run", action="store_true", help="只打印采样计划，不运行 OpenMC")
    parser.add_argument("--log-file", type=Path, default=None, help="日志文件（缺省写入 results/logs/）")
    return parser


def _setup_logging(log_file: Path | None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _fmt_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    paths: Paths = PATHS.ensure()
    _setup_logging(args.log_file or (paths.logs_dir / "run_batch.log"))

    features = config.features()
    bounds = config.bounds()
    n_samples = int(args.cases if args.cases is not None else config["sampling.n_samples"])
    seed = int(config["sampling.random_state"])
    run_cfg = config.openmc_run()

    threads_per_case = int(args.threads or run_cfg.get("threads_per_case", 1))
    cpu = os.cpu_count() or 1
    max_workers = int(args.workers or max(1, cpu // max(1, threads_per_case)))
    max_workers = min(max_workers, n_samples)

    samples = latin_hypercube(n_samples, bounds, seed, features)

    LOGGER.info("配置：%s", config.path)
    LOGGER.info(
        "算例数 %d | 特征 %s | 每算例 %d 线程 | 并行 %d 进程（共 %d 核）",
        n_samples, features, threads_per_case, max_workers, cpu,
    )
    LOGGER.info(
        "OpenMC：batches=%s inactive=%s particles=%s seed=%s",
        run_cfg.get("batches"), run_cfg.get("inactive"),
        run_cfg.get("particles"), run_cfg.get("seed"),
    )

    if args.dry_run:
        LOGGER.info("---- dry-run：仅打印采样计划（不运行 OpenMC）----")
        header = f"{'case':>4} " + " ".join(f"{f:>12}" for f in features) + f" {'gap[cm]':>8} {'rho[g/cc]':>9}"
        LOGGER.info(header)
        for i, row in enumerate(samples):
            params = _params(row, features)
            cell = VeraPinCell(
                enrichment=params["enrichment"], boron_ppm=params["boron_ppm"],
                clad_outer_r=params["clad_r"], fuel_temp=params["fuel_temp"],
                mod_temp=params["mod_temp"], config=config, case_index=i,
            )
            info = cell.describe()
            LOGGER.info(
                "%4d " + " ".join(f"{params[f]:>12.4f}" for f in features)
                + f" {info['gap_cm']:>8.5f} {info['coolant_density']:>9.4f}",
                i,
            )
        LOGGER.info("dry-run 结束：几何与密度均通过校验。")
        return 0

    existing = load_cases(paths.cases_jsonl) if args.resume else {}
    if not args.resume:
        paths.cases_jsonl.unlink(missing_ok=True)

    pending = [
        (i, _params(row, features))
        for i, row in enumerate(samples)
        if existing.get(i, {}).get("status") != "success"
    ]
    if args.resume:
        LOGGER.info("断点续跑：%d 个算例已完成，待算 %d 个", len(existing), len(pending))

    start = time.time()
    completed = 0
    failures: List[int] = []

    def _handle(record: Dict[str, Any]) -> None:
        nonlocal completed
        record.pop("_traceback", None)
        append_case(record, paths.cases_jsonl)
        completed += 1
        elapsed = time.time() - start
        eta = elapsed / max(completed, 1) * (len(pending) - completed)
        if record["status"] == "success":
            LOGGER.info(
                "[%d/%d] case %03d | k_eff = %.5f ± %.5f | 已用 %s，预计剩余 %s",
                completed, len(pending), record["case_id"],
                record["k_eff"], record["std"],
                _fmt_seconds(elapsed), _fmt_seconds(eta),
            )
        else:
            failures.append(int(record["case_id"]))
            LOGGER.error(
                "[%d/%d] case %03d 失败：%s",
                completed, len(pending), record["case_id"], record["status"],
            )

    if pending and max_workers == 1:
        for case_id, params in pending:
            _handle(run_case(case_id, params, config, str(paths.raw_dir), threads_per_case))
    elif pending:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(run_case, case_id, params, config, str(paths.raw_dir), threads_per_case): case_id
                for case_id, params in pending
            }
            for future in as_completed(futures):
                try:
                    _handle(future.result())
                except Exception as exc:  # 子进程自身崩溃（例如 OOM）
                    case_id = futures[future]
                    LOGGER.error("case %03d 子进程异常：%s: %s", case_id, type(exc).__name__, exc)
                    failures.append(case_id)

    records = load_cases(paths.cases_jsonl)
    df = records_to_frame(records.values())
    if df.empty:
        LOGGER.error("没有任何算例结果，未写出数据集。")
        return 1
    csv_path = save_dataset(df, paths.dataset_csv)

    ok = int((df["status"] == "success").sum())
    LOGGER.info("=" * 72)
    LOGGER.info("完成：%d/%d 成功，%d 失败，总耗时 %s", ok, len(df), len(df) - ok, _fmt_seconds(time.time() - start))
    LOGGER.info("数据集：%s", csv_path)
    LOGGER.info("k_eff 范围：%.5f – %.5f", df["k_eff"].min(), df["k_eff"].max())
    if failures:
        LOGGER.warning("失败算例：%s（可用 --resume 重试）", failures[:20])
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
