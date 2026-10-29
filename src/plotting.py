"""matplotlib 统一配置：中文字体、负号、无界面后端。

仓库里的图注都是中文，而 matplotlib 默认的 DejaVu Sans 没有 CJK 字形，
不显式指定字体会画出一堆方框（"豆腐块"）。这里按优先级探测系统里可用的
中文字体，找不到就退回 DejaVu Sans 并给出提示。
"""

from __future__ import annotations

import logging
import warnings
from typing import Iterable

LOGGER = logging.getLogger(__name__)

CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "Noto Sans SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Hiragino Sans GB",
    "SimHei",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
)


def configure_matplotlib(candidates: Iterable[str] = CJK_FONT_CANDIDATES) -> str | None:
    """设置中文字体与负号显示，返回实际选用的字体名（找不到则为 ``None``）。"""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            rcParams["font.family"] = "sans-serif"
            rcParams["axes.unicode_minus"] = False
            rcParams["figure.dpi"] = 120
            LOGGER.debug("matplotlib 使用中文字体：%s", name)
            return name

    warnings.warn(
        "未找到中文字体，图中的中文可能显示为方框。"
        "可安装 SimHei / Noto Sans SC / Microsoft YaHei 后重试。",
        stacklevel=2,
    )
    rcParams["axes.unicode_minus"] = False
    return None
