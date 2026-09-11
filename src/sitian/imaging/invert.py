"""确定性色标反演：已知渲染器的产品图 → 类别栅格与统计（不经过任何模型）。

方法（自校准）：
1. 按版式比例裁出主图区与右侧色标条；
2. 从色标条自上而下提取有序调色板（该图自己的图例，天然免per图校准）；
3. 主图逐像素最近色匹配（阈值内）→ 类别栅格；等值线/文字/底图为中性色，落在阈值外；
4. 输出类别覆盖率、四象限分布、中心区聚合；若产品在 PRODUCT_LEVELS 里
   声明了"类别→物理量"映射，再给出数值统计。

类别序约定：class 0 = 色标最上格（数值最大端）。
PRODUCT_LEVELS 的数值需对照真实图例人工校准一次（方法见 docs），按类数 keyed。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError("imaging 模块需要 Pillow 与 numpy：pip install 'sitian[imaging]'") from exc

# 版式比例（相对整图宽高），已按 992×764 的 eaget jingjinji 产品图实测校准
# （主图区不含标题/坐标注记/底部VALID行；legend 紧贴色标条内部、掐掉上下箭头尖端）
DEFAULT_LAYOUT = {
    "map": (0.075, 0.10, 0.885, 0.895),  # l, t, r, b
    "legend": (0.928, 0.125, 0.952, 0.885),
}

# 产品的规范刻度（自上而下 = 从大到小，取各色带上边界）。检测类数与规范类数
# 不必相等：按色带在条中的相对位置插值取值，对相邻色带的合并/分裂鲁棒。
# 校准方法：Read 一张该产品真实图，抄图例刻度登记于此。
PRODUCT_SCALES: dict[str, list[float]] = {
    "boundary_layer": [2000 - 100 * i for i in range(20)],   # m，2000..100 步长100
    "temp_inversion": [10 - 2 * i for i in range(11)],       # °C，+10..-10（发散，含白带）
}
# 兼容旧名
PRODUCT_LEVELS = PRODUCT_SCALES


def scale_values(product: Optional[str], n_classes: int) -> Optional[list[float]]:
    canon = PRODUCT_SCALES.get(product or "")
    if not canon or n_classes < 2:
        return None
    m = len(canon) - 1
    return [canon[round(i * m / (n_classes - 1))] for i in range(n_classes)]


@dataclass
class ChartAnalysis:
    n_classes: int
    matched_fraction: float          # 主图中被任一类别覆盖的像素比例
    class_coverage: list[float]      # 各类别占主图像素比例（class 0 = 最大端）
    top_class_present: int           # 出现过的最强类别序号（-1=无）
    quadrants: dict[str, int]        # 每象限的最强出现类别
    center_box: dict                 # 中心 1/3 区域统计
    class_values: Optional[list[float]] = None
    value_max_present: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "n_classes": self.n_classes,
            "matched_fraction": round(self.matched_fraction, 4),
            "class_coverage": [round(c, 4) for c in self.class_coverage],
            "top_class_present": self.top_class_present,
            "quadrants": self.quadrants,
            "center_box": self.center_box,
        }
        if self.class_values is not None:
            d["class_values"] = self.class_values
            d["value_max_present"] = self.value_max_present
        return d


def _crop(img: Image.Image, frac: tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    l, t, r, b = frac
    return img.crop((int(w * l), int(h * t), int(w * r), int(h * b)))


def extract_palette(legend: Image.Image, max_classes: int = 24,
                    min_run: int = 4, change_thresh: float = 40.0) -> list[tuple[int, int, int]]:
    """从紧贴色标条内部的裁剪自上而下提取有序调色板。

    按"颜色变化分段"而非"过滤中性色"——发散色标（如逆温的红-白-蓝）
    中间的白色带也是合法类别，必须保留。分隔线行（黑色细线）自然形成
    短 run，被 min_run 过滤掉。
    """
    arr = np.asarray(legend.convert("RGB"), dtype=np.int16)
    row_colors = np.median(arr, axis=1)  # (H, 3) 每行中位色
    palette: list[tuple[int, int, int]] = []
    run: list[np.ndarray] = []

    def flush() -> None:
        if len(run) >= min_run:
            c = tuple(int(x) for x in np.median(np.stack(run), axis=0))
            if not palette or sum(abs(a - b) for a, b in zip(palette[-1], c)) > 12:
                palette.append(c)

    for row in row_colors:
        if run and float(np.abs(row - np.median(np.stack(run), axis=0)).sum()) > change_thresh:
            flush()
            run = []
        run.append(row)
    flush()
    return palette[:max_classes]


def classify(map_img: Image.Image, palette: list[tuple[int, int, int]],
             max_dist: float = 60.0, downscale: int = 3) -> np.ndarray:
    """主图逐像素最近色匹配。返回类别栅格（-1=未匹配）。降采样加速。"""
    img = map_img.convert("RGB")
    if downscale > 1:
        img = img.resize((max(1, img.width // downscale), max(1, img.height // downscale)),
                         Image.NEAREST)
    arr = np.asarray(img, dtype=np.int16)
    pal = np.array(palette, dtype=np.int16)  # (K,3)
    dist = np.abs(arr[:, :, None, :] - pal[None, None, :, :]).sum(axis=3)  # (H,W,K) L1
    labels = dist.argmin(axis=2)
    labels[dist.min(axis=2) > max_dist] = -1
    return labels


def analyze_chart(path: str | Path, product: Optional[str] = None,
                  layout: Optional[dict] = None) -> ChartAnalysis:
    img = Image.open(path)
    layout = layout or DEFAULT_LAYOUT
    palette = extract_palette(_crop(img, layout["legend"]))
    if not palette:
        raise ValueError(f"无法从图例提取调色板: {path}")
    labels = classify(_crop(img, layout["map"]), palette)
    total = labels.size
    coverage = [float((labels == k).sum()) / total for k in range(len(palette))]
    matched = float((labels >= 0).sum()) / total
    present = [k for k, c in enumerate(coverage) if c > 0.002]
    top_present = min(present) if present else -1  # class 0 = 最大端

    h, w = labels.shape
    quads = {}
    for name, sl in {
        "NW": (slice(0, h // 2), slice(0, w // 2)), "NE": (slice(0, h // 2), slice(w // 2, w)),
        "SW": (slice(h // 2, h), slice(0, w // 2)), "SE": (slice(h // 2, h), slice(w // 2, w)),
    }.items():
        q = labels[sl]
        qp = [k for k in range(len(palette)) if (q == k).sum() / q.size > 0.002]
        quads[name] = min(qp) if qp else -1
    cb = labels[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    cb_present = [k for k in range(len(palette)) if (cb == k).sum() / cb.size > 0.002]
    center = {"top_class_present": min(cb_present) if cb_present else -1,
              "matched_fraction": round(float((cb >= 0).sum()) / cb.size, 4)}

    values = scale_values(product, len(palette))
    vmax = values[top_present] if (values and top_present >= 0) else None
    return ChartAnalysis(
        n_classes=len(palette), matched_fraction=matched, class_coverage=coverage,
        top_class_present=top_present, quadrants=quads, center_box=center,
        class_values=values, value_max_present=vmax,
    )
