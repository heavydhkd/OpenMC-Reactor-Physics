"""pytest 公共配置：把仓库根目录放进 sys.path，避免测试依赖 pip install -e ."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def dataset(config):
    """仓库自带的 60 个算例（旧物理模型生成），仅用于代理模型链路测试。"""
    from src.data import load_dataset

    path = config.resolve("data/processed/vera_p1_dataset.csv")
    if not path.exists():
        pytest.skip("缺少 data/processed/vera_p1_dataset.csv")
    return load_dataset(path)
