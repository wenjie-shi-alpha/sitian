#!/usr/bin/env python3
"""Generate the project LLM fine-tuning technical feasibility report (DOCX)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "reports"
ASSET_DIR = OUT_DIR / "llm_finetuning_feasibility_word" / "assets"
DOCX_PATH = OUT_DIR / "LLM微调技术可行性验证报告_2026-09-03.docx"

BLUE = "275D86"
DARK = "222222"
MID = "6B7280"
LIGHT = "D7DEE5"
HEADER_FILL = "EEF2F5"
WHITE = "FFFFFF"

for _font_path in (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
):
    if Path(_font_path).exists():
        font_manager.fontManager.addfont(_font_path)

CJK_PLOT_FONT = "Noto Sans CJK JP"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, **edges) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge_name, edge_data in edges.items():
        edge = borders.find(qn(f"w:{edge_name}"))
        if edge is None:
            edge = OxmlElement(f"w:{edge_name}")
            borders.append(edge)
        for key, value in edge_data.items():
            edge.set(qn(f"w:{key}"), str(value))


def set_cell_margins(cell, top=80, start=90, bottom=80, end=90) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    cant_split.set(qn("w:val"), "true")
    tr_pr.append(cant_split)


def set_keep_with_next(paragraph, value: bool = True) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    keep = p_pr.find(qn("w:keepNext"))
    if keep is None:
        keep = OxmlElement("w:keepNext")
        p_pr.append(keep)
    keep.set(qn("w:val"), "1" if value else "0")


def set_run_font(run, name: str = "Noto Sans CJK SC", size: float | None = None,
                 bold: bool | None = None, color: str | None = None) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Noto Serif CJK SC"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Serif CJK SC")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.first_line_indent = Pt(21)

    for style_name, size, before, after in (
        ("Title", 24, 0, 16),
        ("Heading 1", 16, 18, 8),
        ("Heading 2", 13, 13, 6),
        ("Heading 3", 11, 10, 4),
    ):
        style = styles[style_name]
        style.font.name = "Noto Sans CJK SC"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(DARK if style_name == "Title" else BLUE)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.first_line_indent = Pt(0)

    if "Report Subtitle" not in styles:
        subtitle = styles.add_style("Report Subtitle", WD_STYLE_TYPE.PARAGRAPH)
    else:
        subtitle = styles["Report Subtitle"]
    subtitle.font.name = "Noto Sans CJK SC"
    subtitle._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")
    subtitle.font.size = Pt(12)
    subtitle.font.color.rgb = RGBColor.from_string(MID)
    subtitle.paragraph_format.space_after = Pt(6)
    subtitle.paragraph_format.first_line_indent = Pt(0)

    if "Figure Caption" not in styles:
        cap = styles.add_style("Figure Caption", WD_STYLE_TYPE.PARAGRAPH)
    else:
        cap = styles["Figure Caption"]
    cap.font.name = "Noto Sans CJK SC"
    cap._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")
    cap.font.size = Pt(9)
    cap.font.color.rgb = RGBColor.from_string(MID)
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(8)
    cap.paragraph_format.first_line_indent = Pt(0)

    if "Source Note" not in styles:
        note = styles.add_style("Source Note", WD_STYLE_TYPE.PARAGRAPH)
    else:
        note = styles["Source Note"]
    note.font.name = "Noto Sans CJK SC"
    note._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")
    note.font.size = Pt(8.5)
    note.font.color.rgb = RGBColor.from_string(MID)
    note.paragraph_format.space_after = Pt(6)
    note.paragraph_format.first_line_indent = Pt(0)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, end])
    set_run_font(run, size=8.5, color=MID)


def configure_page(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.35)
    section.right_margin = Cm(2.15)
    section.header_distance = Cm(1.0)
    section.footer_distance = Cm(1.0)

    header = section.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run("LLM 微调 · 技术研究报告")
    set_run_font(r, size=8.5, color=MID)
    p.paragraph_format.space_after = Pt(0)

    footer = section.footer
    add_page_number(footer.paragraphs[0])


def add_label_paragraph(doc: Document, label: str, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    r1 = p.add_run(label)
    set_run_font(r1, bold=True, color=BLUE)
    r2 = p.add_run(text)
    set_run_font(r2, name="Noto Serif CJK SC")


def add_simple_table(doc: Document, headers: list[str], rows: list[list[str]], widths=None) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.allow_autofit = False
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    prevent_row_split(hdr)
    for idx, text in enumerate(headers):
        cell = hdr.cells[idx]
        cell.text = ""
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, HEADER_FILL)
        set_cell_margins(cell)
        set_cell_border(
            cell,
            top={"val": "single", "sz": "10", "color": BLUE},
            bottom={"val": "single", "sz": "6", "color": BLUE},
        )
        p = cell.paragraphs[0]
        p.paragraph_format.first_line_indent = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(text)
        set_run_font(r, size=9.5, bold=True, color=DARK)
        if widths:
            cell.width = Cm(widths[idx])

    for row_values in rows:
        row = table.add_row()
        prevent_row_split(row)
        for idx, text in enumerate(row_values):
            cell = row.cells[idx]
            cell.text = ""
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
            set_cell_margins(cell)
            set_cell_border(cell, bottom={"val": "single", "sz": "4", "color": LIGHT})
            p = cell.paragraphs[0]
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            set_run_font(r, name="Noto Serif CJK SC", size=9.2)
            if widths:
                cell.width = Cm(widths[idx])
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_picture(doc: Document, path: Path, width_in: float, caption: str, source: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(str(path), width=Inches(width_in))
    cap = doc.add_paragraph(caption, style="Figure Caption")
    set_keep_with_next(cap, True)
    src = doc.add_paragraph(source, style="Source Note")
    src.alignment = WD_ALIGN_PARAGRAPH.CENTER


def make_route_figure(path: Path) -> None:
    plt.rcParams["font.sans-serif"] = [CJK_PLOT_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11.2, 3.45), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 3.45)
    ax.axis("off")

    nodes = [
        (0.25, 1.25, 1.85, 1.0, "领域材料与\n专家会商样本", "术语、证据链、表达规范"),
        (2.45, 1.25, 1.85, 1.0, "隔离训练案例\n与证据接口", "时间安全、可回放"),
        (4.65, 1.25, 1.85, 1.0, "条件式 SFT\n协议热身", "仅在格式门禁需要时启用"),
        (6.85, 1.25, 1.85, 1.0, "分阶段 Agent RL", "先固定证据，再开放工具选择"),
        (9.05, 1.25, 1.85, 1.0, "独立验证与\n效果门禁", "CAMS、表格模型、OOD"),
    ]
    for i, (x, y, w, h, title, sub) in enumerate(nodes):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.2,
            edgecolor="#275D86",
            facecolor="#FFFFFF",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + 0.67, title, ha="center", va="center", fontsize=10.5,
                color="#1F2937", fontweight="bold", linespacing=1.25)
        ax.text(x + w / 2, y + 0.21, sub, ha="center", va="center", fontsize=7.6,
                color="#6B7280")
        if i < len(nodes) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w + 0.05, y + 0.5), (nodes[i + 1][0] - 0.05, y + 0.5),
                arrowstyle="-|>", mutation_scale=11, linewidth=1.0, color="#6B7280"
            ))

    ax.text(0.25, 2.78, "训练闭环", fontsize=10, color="#275D86", fontweight="bold")
    ax.plot([1.2, 10.9], [2.92, 2.92], color="#D7DEE5", linewidth=1.0)
    ax.text(10.9, 3.03, "门禁通过后逐级扩展", fontsize=8.5, color="#6B7280", ha="right")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


def make_readiness_figure(path: Path) -> None:
    plt.rcParams["font.sans-serif"] = [CJK_PLOT_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = ["可用采样组", "过程工具调用", "最终有效提交", "首次提交有效", "语义落地满分"]
    values = [100.0, 98.75, 98.0, 93.75, 70.408]
    fig, ax = plt.subplots(figsize=(8.8, 3.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = list(range(len(labels)))
    bars = ax.barh(y, values, color="#4C7899", height=0.52)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("完成率 / 满分率（%）", color="#4B5563", fontsize=9)
    ax.tick_params(axis="y", labelsize=9.2, colors="#222222", length=0)
    ax.tick_params(axis="x", labelsize=8.2, colors="#6B7280")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#D1D5DB")
    for bar, val in zip(bars, values):
        ax.text(val + 1.0, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", ha="left", fontsize=8.8, color="#222222")
    ax.axvline(95, color="#8A5A44", linewidth=1.0, linestyle="--")
    ax.text(95, -0.68, "格式门禁 95%", ha="center", va="bottom", fontsize=8, color="#8A5A44")
    fig.tight_layout(pad=0.8)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)


def add_cover(doc: Document) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(50)
    p.paragraph_format.space_after = Pt(8)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run("LLM 微调技术可行性验证报告")
    set_run_font(r, size=25, bold=True, color=DARK)

    line = doc.add_paragraph()
    line.paragraph_format.first_line_indent = Pt(0)
    line.paragraph_format.space_after = Pt(26)
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "18")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), BLUE)
    border.append(bottom)
    line._p.get_or_add_pPr().append(border)

    for text in (
        "应用场景：面向空气质量预报业务的领域大模型",
        "研究对象：参数高效微调与 Agent 强化学习",
        "报告性质：技术研究与阶段决策依据",
    ):
        p = doc.add_paragraph(text, style="Report Subtitle")
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc.add_paragraph().paragraph_format.space_after = Pt(85)

    meta = [
        ("版本", "V1.0"),
        ("编制日期", "2026 年 9 月 3 日"),
        ("研究边界", "以当前代码、数据、完成版探针和测试结果为依据"),
    ]
    table = doc.add_table(rows=len(meta), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    for i, (k, v) in enumerate(meta):
        for j, text in enumerate((k, v)):
            cell = table.cell(i, j)
            cell.text = ""
            cell.width = Cm(3.0 if j == 0 else 10.5)
            set_cell_margins(cell, top=70, bottom=70)
            set_cell_border(cell, bottom={"val": "single", "sz": "3", "color": LIGHT})
            p = cell.paragraphs[0]
            p.paragraph_format.first_line_indent = Pt(0)
            r = p.add_run(text)
            set_run_font(r, size=9.5, bold=(j == 0), color=MID if j == 0 else DARK)

    doc.add_page_break()


def build_report() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    route_path = ASSET_DIR / "technical_route.png"
    readiness_path = ASSET_DIR / "readiness_indicators.png"
    make_route_figure(route_path)
    make_readiness_figure(readiness_path)

    doc = Document()
    configure_styles(doc)
    configure_page(doc)
    props = doc.core_properties
    props.title = "LLM 微调技术可行性验证报告"
    props.subject = "LLM 参数高效微调与 Agent 强化学习技术可行性"
    props.author = "技术研究组"
    props.keywords = "LLM微调, LoRA, Agent RL, 空气质量预报, 技术可行性"

    add_cover(doc)

    doc.add_heading("摘要", level=1)
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    r = p.add_run("综合判定：技术路线可行，建议按门禁分阶段进入工程验证。")
    set_run_font(r, name="Noto Sans CJK SC", size=12, bold=True, color=BLUE)

    doc.add_paragraph(
        "现有工作已经形成开展 LLM 微调所需的完整基础。专家会商语料、人工校验稿和训练入口可用于学习领域表达与证据引用协议；空气质量预报任务已经转换为时间安全、结构化、可评分的案例集合；模型输出、工具调用、奖励计算和评测均具有明确接口，能够组成可回放、可复算的训练闭环；独立验证集、业务基线与回归测试也已具备，可对微调收益进行客观判定。"
    )
    doc.add_paragraph(
        "推荐采用“条件式 SFT + 分阶段 Agent RL”的路线：当格式或协议门禁未达标时，先以少量高质量样本完成协议热身；随后在固定证据、单次结构化决策场景中验证同策略更新，再逐步开放多轮工具选择。首轮以 1.7B/4B 参数高效微调验证训练闭环，在独立验证集达到效果门禁后再扩大训练规模。"
    )
    add_label_paragraph(doc, "关键词：", "大语言模型；参数高效微调；LoRA；Agent 强化学习；空气质量预报；可复算评测")

    doc.add_heading("1  验证目标与判定口径", level=1)
    doc.add_heading("1.1 验证目标", level=2)
    doc.add_paragraph(
        "本报告回答的核心问题是：在现有数据、任务定义、训练接口、算力和评测条件下，能否以可控成本开展 LLM 微调，并获得可量化、可复现、可推广的验证结果。报告重点判断技术条件是否闭合，不将一次模型得分等同于最终业务结论。"
    )
    doc.add_heading("1.2 可行性判定标准", level=2)
    doc.add_paragraph(
        "可行性判定从数据、任务、训练和评测四个方面展开。数据方面要求材料来源明确，训练、验证和测试边界清晰，关键样本能够追溯；任务方面要求输入、动作与输出协议稳定，模型行为可以映射为明确的监督信号或奖励信号。训练方面要求基础模型、LoRA、采样和训练运行时能够连接，损失、参数更新和断点恢复均可检查；评测方面要求存在冻结基线、业务基线、独立验证集和统计门禁，从而区分真实改进与偶然波动。"
    )

    doc.add_heading("2  现有基础与可复用资产", level=1)
    doc.add_heading("2.1 领域知识与专家语料", level=2)
    doc.add_paragraph(
        "现有领域材料覆盖污染过程预报、数据结构化、SFT/GRPO 训练入口和调度—执行式智能体设计。当前可直接复用的专家材料包含 24 个 SFT 数据文件、共 272 条会商样本，以及 22 份人工校验稿。样本保留会议日期、发言人、角色、主题、证据块和幻灯片页码等元数据，可承担术语统一、分析步骤学习和引用协议约束。"
    )
    doc.add_paragraph(
        "这批材料规模适合用作高质量协议样本和专家规则来源。面向全国城市泛化时，应将其与结构化预报案例联合使用，使专家材料负责约束推理步骤和证据表达，规模化案例负责覆盖不同城市、季节和污染过程。"
    )

    doc.add_heading("2.2 训练与评测数据", level=2)
    doc.add_paragraph(
        "现有原始案例共 7,548 个，其中 7,057 个通过基础校验；在完成冬季挑战集与空间分布外样本隔离后，最终训练清单包含 3,061 个案例，覆盖 112 个城市。独立评测侧保留 50 个空间分布外验证案例、178 个空间分布外测试案例和 344 个冬季挑战案例。该组织方式能够降低地点、事件和时间泄漏，支持对跨城市、跨季节泛化能力的判断。"
    )

    doc.add_heading("2.3 结构化任务与工程接口", level=2)
    doc.add_paragraph(
        "当前任务将模型动作收敛为 PM2.5、PM10 和 O3 的浓度区间预测，AQI、首要污染物和过程判定由评测端按确定性规则派生。证据工具返回内容可回放，奖励由规则计算，不依赖额外的语言模型裁判。这一设计显著降低了训练目标漂移和评测不一致风险。"
    )
    doc.add_paragraph(
        "训练工程已具备 Qwen3 系列模型、LoRA、veRL/vLLM 同策略运行路径、工具令牌损失掩码约束，以及训练前置检查脚本。当前代码与前期训练模块合计 255 项单元测试全部通过，为后续迭代提供了稳定的回归基础。"
    )

    doc.add_heading("3  技术可行性分析", level=1)
    add_simple_table(
        doc,
        ["可行性维度", "现有证据", "判定"],
        [
            ["数据", "专家会商样本可追溯；3,061 个隔离训练案例覆盖 112 城；独立 OOD 与冬季挑战集已保留。", "可行。能够同时支持协议学习、策略学习和泛化验证。"],
            ["任务", "输出字段有限且结构化；污染物指标由确定性规则派生；证据接口可回放。", "可行。动作空间明确，训练反馈可稳定复算。"],
            ["算法", "已有 SFT/GRPO 入口，并具备 LoRA 与同策略 Agent RL 运行契约。", "可行。适合先以参数高效方式闭环，再逐步扩展。"],
            ["工程", "模型、采样、工具、奖励、检查点和评测接口均有明确边界；255 项单元测试通过。", "可行。主要工作是完成训练烟雾测试和策略同步校验。"],
            ["评测", "完成版正式探针为 50 案例×8 采样；存在 CAMS 与同信息表格基线及独立验证集。", "可行。能够以统计门禁判断微调增益。"],
        ],
        widths=[2.3, 8.0, 5.0],
    )
    doc.add_paragraph(
        "上述条件已经覆盖从数据进入、模型生成、工具交互、奖励回传到独立验证的完整链路。因此，当前需要解决的已不是“是否具备微调条件”，而是“采用何种最小实验能够以最低成本确认增益，并据此决定是否扩大规模”。"
    )

    doc.add_heading("4  推荐微调路线", level=1)
    add_picture(
        doc, route_path, 6.4,
        "图 1  LLM 微调分阶段技术路线",
        "资料来源：根据现有训练设计、结构化数据、工具、奖励和评测链路整理。",
    )
    doc.add_heading("4.1 阶段一：协议门禁与条件式 SFT", level=2)
    doc.add_paragraph(
        "先检查首次提交有效率、空转轮次和语义落地情况。若格式或引用协议未达到门禁，则从专家会商材料和当前结构化案例中抽取高质量样本，以短周期 SFT 学习输出格式、证据引用和工具使用协议；若门禁已经达到，则不安排泛化式 SFT，直接进入强化学习闭环。"
    )
    doc.add_heading("4.2 阶段二：固定证据的单次决策 Agent RL", level=2)
    doc.add_paragraph(
        "首轮强化学习固定或预取证据，模型只承担一次结构化预测。这样能够把奖励变化主要归因于预测策略本身，并检查采样策略、训练策略、LoRA 参数、工具令牌损失掩码和奖励回传是否真实同步。建议先运行 50 个训练步，在第 51 步重载检查点继续训练，以验证参数变化和断点恢复。"
    )
    doc.add_heading("4.3 阶段三：开放多轮工具选择", level=2)
    doc.add_paragraph(
        "只有在阶段二证明策略能够稳定更新、独立验证结果不退化之后，才开放多轮工具调用和证据选择。该阶段的优化目标包括：减少无效调用、提高证据与结论的一致性、改善污染过程识别，同时维持结构化输出的稳定性。"
    )
    doc.add_heading("4.4 阶段四：独立验证与规模决策", level=2)
    doc.add_paragraph(
        "每个训练阶段均在独立验证集上与冻结策略、CAMS 指导和同信息表格模型比较。只有当按发布日期聚类自助法计算的差异置信区间下界大于 0，且冬季挑战集、空间分布外样本和事件样本未出现显著退化时，才进入更大规模训练。"
    )

    doc.add_heading("5  验证证据与验收门禁", level=1)
    doc.add_heading("5.1 已完成探针能够支撑训练闭环", level=2)
    doc.add_paragraph(
        "完成版正式探针覆盖 50 个案例、每例 8 次采样，共 400 条轨迹。结果显示：全部采样组均存在差异性，98.0% 的轨迹最终形成有效提交，98.75% 的轨迹使用了过程工具，语义落地获得满分的轨迹占 70.4%。这说明当前策略已经能够稳定进入任务、调用工具并产生可评分动作，满足开展强化学习探索的基本条件。"
    )
    add_picture(
        doc, readiness_path, 6.2,
        "图 2  完成版正式探针的关键训练就绪指标",
        "注：样本量为 50 个案例×8 次采样。虚线为首次提交有效率的 95% 工程门禁。",
    )
    doc.add_paragraph(
        "首次提交有效率为 93.75%，与 95% 工程门禁相差 1.25 个百分点；该差距适合通过协议样本、解码约束或短周期 SFT 处理。探针中仍需压低空转轮次，并在新奖励版本下复核审计完整性，但这些属于可控制的训练前置项，不改变整体可行性判断。"
    )

    doc.add_heading("5.2 基线为微调收益提供客观标尺", level=2)
    doc.add_paragraph(
        "在同一 50 案例样本上，冻结策略、CAMS 指导和同信息表格模型的结果分分别为 0.535、0.631 和 0.683。该差距给出了明确的学习目标：微调应优先改善区间判断、污染过程识别和证据选择，而不是仅优化语言表述。因为输入信息、案例集合与评分方式一致，后续可以将差异直接用于判断训练是否带来可归因的改进。"
    )

    doc.add_heading("5.3 分阶段验收标准", level=2)
    add_simple_table(
        doc,
        ["阶段", "主要验证内容", "建议通过条件", "交付物"],
        [
            ["协议门禁", "结构化提交、空转轮次、语义落地", "首次提交有效率≥95%；空转轮次≤5%；审计完整", "门禁报告与失败样本清单"],
            ["训练烟雾测试", "LoRA 参数更新、损失掩码、策略同步、断点恢复", "损失有限；适配器参数变化；重载后可继续；采样与训练权重一致", "训练日志、检查点和重载证据"],
            ["阶段 A", "固定证据、单次决策的策略改进", "独立验证不退化；训练/验证奖励趋势一致；格式稳定", "最小可行策略与评测报告"],
            ["阶段 B", "多轮工具选择与证据使用", "工具成本受控；语义落地提升；事件和冬季样本无明显退化", "完整 Agent 策略与误差分析"],
            ["规模扩展", "对业务基线的统计显著改进", "相对 CAMS 的聚类自助法差异下界>0；再对同信息表格模型进行最终研究比较", "规模化训练决策"],
        ],
        widths=[2.0, 4.4, 6.0, 3.0],
    )

    doc.add_heading("6  资源与实施可行性", level=1)
    doc.add_heading("6.1 算力与模型", level=2)
    doc.add_paragraph(
        "现有 RTX 4090 24GB 环境可承担 1.7B/4B 模型的 LoRA 烟雾测试、短周期训练和评测回放，当前环境也已具备可训练模型与推理服务接口。更大规模模型不作为首轮可行性验证的前置条件；在小模型闭环通过并证明独立验证收益后，可按需要转移到 80GB 级云端 GPU 进行扩展。"
    )
    doc.add_heading("6.2 人员与周期", level=2)
    doc.add_paragraph(
        "在数据和接口保持稳定的前提下，建议用两个迭代完成首轮验证：第一个迭代完成协议门禁、50+1 步训练烟雾测试和固定证据训练；第二个迭代完成独立验证、失败样本分析和是否开放多轮工具选择的决策。工作量主要集中在训练运行时联调、轨迹审计和评测复核，不需要重新建设数据平台。"
    )
    doc.add_heading("6.3 复现与运维", level=2)
    doc.add_paragraph(
        "每次训练应固定数据清单、schema 版本、reward 版本、基础模型标识、LoRA 配置、随机种子和推理参数；同时保存训练日志、适配器、优化器状态、验证输出和版本摘要。这样可以在奖励或结构变化时定位差异，避免将接口漂移误判为模型能力变化。"
    )

    doc.add_heading("7  主要风险与控制措施", level=1)
    add_simple_table(
        doc,
        ["风险", "影响", "控制措施"],
        [
            ["数据或时间泄漏", "验证分数被高估，无法代表真实预报场景。", "继续使用训练、OOD、冬季挑战和事件分层；按发布日期聚类统计。"],
            ["协议与奖励版本漂移", "训练目标与评测目标不一致。", "训练产物绑定 schema/reward 版本；每次运行先执行契约和回归测试。"],
            ["SFT 样本与当前协议不匹配", "学习旧字段或只改善表述。", "仅选取可映射到当前动作协议的样本；旧语料作为原料重新导出。"],
            ["策略权重不同步或掩码错误", "表面有训练日志，但实际策略未有效更新。", "强制检查参数差异、工具令牌掩码、策略标识与断点重载。"],
            ["奖励投机与无效调用", "模型得分提高但业务行为恶化。", "保留工具成本、空转轮次和语义落地护栏，并进行人工轨迹抽检。"],
            ["季节与区域泛化不足", "局部提升无法迁移到全国或重污染过程。", "冬季挑战、空间分布外和事件样本分别报告，不以单一平均分替代。"],
        ],
        widths=[3.2, 5.0, 7.2],
    )

    doc.add_heading("8  结论与建议", level=1)
    doc.add_paragraph(
        "综合数据资产、任务结构、训练工程、算力条件和评测体系，开展 LLM 微调具有明确的技术可行性。现有条件已经覆盖可学习的领域材料、规模化且隔离的训练案例、确定性奖励、可回放工具、同策略训练路径和独立效果标尺，能够以较小试验成本验证微调是否带来真实收益。"
    )
    doc.add_paragraph(
        "建议批准进入“门禁校准—训练烟雾测试—固定证据 Agent RL—独立验证”的最小闭环。实施中坚持条件式 SFT、分阶段释放复杂度和以独立验证集统计增益作为规模扩展依据。若首轮闭环通过，则该路线可进一步用于全国城市空气质量预报智能体的持续迭代。"
    )
    add_label_paragraph(doc, "最终判定：", "可行，建议按本报告门禁启动首轮参数高效微调验证。")

    doc.save(DOCX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    build_report()
