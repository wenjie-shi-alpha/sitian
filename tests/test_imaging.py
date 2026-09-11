"""图像层测试：合成图上验证反演与拼图，假 VLM 验证 describe_image 工具。"""
import pytest

PIL = pytest.importorskip("PIL")
np = pytest.importorskip("numpy")

from PIL import Image, ImageDraw  # noqa: E402

from sitian.env import EnvConfig, ForecastEnv  # noqa: E402
from sitian.imaging import analyze_chart, build_mosaic, scale_values  # noqa: E402
from sitian.imaging.invert import DEFAULT_LAYOUT  # noqa: E402
from sitian.synth import make_case  # noqa: E402

BANDS = [(128, 0, 128), (255, 0, 0), (255, 200, 0), (0, 180, 0), (80, 160, 255)]  # 大→小


def _make_chart(tmp_path, w=600, h=450):
    """合成一张符合版式的产品图：图例5个色带，主图上半=band1(红)、下半=band4(蓝)。"""
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ml, mt, mr, mb = [int(f * s) for f, s in zip(DEFAULT_LAYOUT["map"], (w, h, w, h))]
    mid = (mt + mb) // 2
    # 红蓝之间留白缝，避免降采样取整把边界行漏进下象限
    d.rectangle([ml, mt, mr, mid - 6], fill=BANDS[1])
    d.rectangle([ml, mid + 6, mr, mb], fill=BANDS[4])
    ll, lt, lr, lb = [int(f * s) for f, s in zip(DEFAULT_LAYOUT["legend"], (w, h, w, h))]
    seg = (lb - lt) // len(BANDS)
    for i, c in enumerate(BANDS):
        d.rectangle([ll, lt + i * seg, lr, lt + (i + 1) * seg - 1], fill=c)
    path = tmp_path / "toy_2023011412_024_jingjinji.png"
    img.save(path)
    return path


def test_invert_on_synthetic_chart(tmp_path):
    r = analyze_chart(_make_chart(tmp_path)).to_dict()
    assert r["n_classes"] == 5
    assert r["matched_fraction"] > 0.95
    assert r["top_class_present"] == 1                       # 红=第2强类别
    assert r["quadrants"]["NW"] == 1 and r["quadrants"]["SW"] == 4
    assert abs(r["class_coverage"][1] - 0.5) < 0.08
    assert abs(r["class_coverage"][4] - 0.5) < 0.08


def test_scale_values_interpolation():
    assert scale_values("boundary_layer", 21)[:3] == [2000, 1900, 1800]
    v16 = scale_values("boundary_layer", 16)
    assert len(v16) == 16 and v16[0] == 2000 and v16[-1] == 100
    assert scale_values("unknown_product", 10) is None


def _make_cycle(tmp_path):
    cyc = tmp_path / "202312141200"
    sub = cyc / "toyprod"
    sub.mkdir(parents=True)
    for lead in (12, 24, 36):
        img = Image.new("RGB", (200, 150), (200 + lead, 100, 100))
        img.save(sub / f"toyprod_2023121412_{lead:03d}_jingjinji.png")
    return cyc


def test_mosaic_grid(tmp_path):
    cyc = _make_cycle(tmp_path)
    board = build_mosaic(cyc, "toyprod", [12, 24, 36], tile_width=100, cols=2)
    assert board.width == 200 and board.height > 100  # 2列2行
    with pytest.raises(FileNotFoundError):
        build_mosaic(cyc, "nonexistent", [12])


class _FakeVLM:
    def available(self):
        return True

    def describe(self, image, prompt, **kw):
        return f"假描述: 收到 {image.width}x{image.height} 拼图"


def test_env_image_tools(tmp_path):
    bundle = make_case("accumulation", seed=1)
    bundle.meta["image_cycle_dir"] = str(_make_cycle(tmp_path))
    env = ForecastEnv(bundle, EnvConfig(vlm_client=_FakeVLM()))
    env.reset()
    names = {s["name"] for s in env.tool_specs()}
    assert {"analyze_chart", "describe_image"} <= names

    obs, _, _, _ = env.step({"name": "describe_image",
                             "args": {"product": "toyprod", "lead_hours": [12, 36]}})
    assert obs["content"]["available"] is True
    assert "假描述" in obs["content"]["description"]

    obs, _, _, _ = env.step({"name": "analyze_chart",
                             "args": {"product": "nonexistent", "lead_hour": 24}})
    assert "error" in obs["content"]


def test_image_tools_absent_without_cycle_dir():
    env = ForecastEnv(make_case("clean", seed=2))
    env.reset()
    assert "analyze_chart" not in {s["name"] for s in env.tool_specs()}
