"""趋势拼图：一个起报时次的同产品多时效图 → 单张带标注的证据板。

对应业务里"把多张图拼到一页PPT看趋势"的做法：VLM 只看一张拼图即可
给出跨时效的演变判断，上下文成本从 N 张图降到 1 张。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover
    raise ImportError("imaging 模块需要 Pillow：pip install 'sitian[imaging]'") from exc

# 兼容 surface_2024010200_006_china_east.png 与 rain2025101400_006_jingjinji.png 两种命名
_NAME_RE = re.compile(
    r"^(?P<prod>.+?)_?(?P<init>\d{10})_(?P<lead>\d{3})_(?P<domain>china_east|china|jingjinji)\.png$")

DOMAINS = ("jingjinji", "china_east", "china")


def index_cycle_images(cycle_dir: str | Path) -> dict[str, dict[str, dict[int, Path]]]:
    """扫描时次目录 → {product: {domain: {lead_hour: path}}}。"""
    out: dict[str, dict[str, dict[int, Path]]] = {}
    root = Path(cycle_dir)
    if not root.is_dir():
        return out
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sub.iterdir():
            m = _NAME_RE.match(f.name)
            if m:
                out.setdefault(sub.name, {}).setdefault(
                    m["domain"], {})[int(m["lead"])] = f
    return out


def available_summary(cycle_dir: str | Path) -> dict:
    idx = index_cycle_images(cycle_dir)
    return {
        prod: {dom: {"leads": sorted(leadmap), "count": len(leadmap)}
               for dom, leadmap in doms.items()}
        for prod, doms in idx.items()
    }


def build_mosaic(cycle_dir: str | Path, product: str, lead_hours: list[int],
                 domain: str = "jingjinji", tile_width: int = 560,
                 cols: int = 3, max_tiles: int = 8) -> Image.Image:
    idx = index_cycle_images(cycle_dir)
    doms = idx.get(product)
    if not doms:
        raise FileNotFoundError(f"cycle 无产品 {product!r}，可用: {sorted(idx)}")
    leadmap = doms.get(domain) or doms.get(next(iter(doms)))
    tiles = []
    for lh in lead_hours[:max_tiles]:
        path = leadmap.get(lh)
        if path is None:  # 就近取有的时效
            近 = min(leadmap, key=lambda k: abs(k - lh), default=None)
            if 近 is None:
                continue
            path, lh = leadmap[近], 近
        img = Image.open(path).convert("RGB")
        h = int(img.height * tile_width / img.width)
        img = img.resize((tile_width, h))
        draw = ImageDraw.Draw(img)
        label = f"+{lh:03d}h"
        draw.rectangle([4, 4, 4 + 12 * len(label), 30], fill=(20, 40, 90))
        draw.text((10, 8), label, fill=(255, 255, 255))
        tiles.append(img)
    if not tiles:
        raise FileNotFoundError(f"{product}/{domain} 无匹配时效 {lead_hours}")
    cols = min(cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    th = max(t.height for t in tiles)
    board = Image.new("RGB", (cols * tile_width, rows * th), (255, 255, 255))
    for i, t in enumerate(tiles):
        board.paste(t, ((i % cols) * tile_width, (i // cols) * th))
    return board
