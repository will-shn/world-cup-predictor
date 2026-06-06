"""
Build docs/World_Cup_Engine_Guide.pdf from docs/ENGINE_GUIDE.md

Usage (from project root):
    pip install reportlab
    python scripts/generate_pdf_guide.py
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "ENGINE_GUIDE.md"
PDF_PATH = ROOT / "docs" / "World_Cup_Engine_Guide.pdf"


def _strip_markdown_inline(text: str) -> str:
    """Remove markdown links, bold, and backticks for safe PDF text."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def _escape(text: str) -> str:
    """Escape text for ReportLab Paragraph XML."""
    text = _strip_markdown_inline(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = text.replace("\u2212", "-").replace("\u03bb", "lambda")
    text = text.replace("\u03c1", "rho").replace("\u03c4", "tau")
    text = text.replace("\u00f6", "o").replace("\u00e9", "e").replace("\u00e8", "e")
    text = text.replace("\u00e0", "a").replace("\u00f1", "n").replace("\u00e7", "c")
    return text


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CustomTitle",
            parent=base["Title"],
            fontSize=22,
            spaceAfter=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1B5E20"),
        ),
        "h1": ParagraphStyle(
            "CustomH1",
            parent=base["Heading1"],
            fontSize=16,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.HexColor("#1B5E20"),
        ),
        "h2": ParagraphStyle(
            "CustomH2",
            parent=base["Heading2"],
            fontSize=13,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#2E7D32"),
        ),
        "h3": ParagraphStyle(
            "CustomH3",
            parent=base["Heading3"],
            fontSize=11,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "CustomBody",
            parent=base["BodyText"],
            fontSize=9.5,
            leading=13,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "CustomBullet",
            parent=base["BodyText"],
            fontSize=9.5,
            leading=13,
            leftIndent=18,
            bulletIndent=8,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "CustomCode",
            parent=base["Code"],
            fontSize=8,
            leading=10,
            leftIndent=12,
            backColor=colors.HexColor("#F5F5F5"),
            spaceAfter=6,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER,
        ),
    }


def _parse_table_row(line: str) -> list[str]:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells


def _is_table_separator(line: str) -> bool:
    return bool(re.match(r"^\|[\s\-:|]+\|$", line.strip()))


def _build_table(rows: list[list[str]], styles: dict) -> Table:
    data = [[Paragraph(_escape(c), styles["body"]) for c in row] for row in rows]
    col_count = max(len(r) for r in data)
    widths = [5.8 * inch / col_count] * col_count
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F5E9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1B5E20")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def markdown_to_story(md_text: str) -> list:
    styles = _styles()
    story: list = []
    lines = md_text.splitlines()
    i = 0
    in_code = False
    code_buf: list[str] = []
    table_buf: list[list[str]] = []

    def flush_table():
        nonlocal table_buf
        if table_buf:
            story.append(_build_table(table_buf, styles))
            story.append(Spacer(1, 8))
            table_buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                story.append(
                    Paragraph(
                        _escape("\n".join(code_buf)),
                        styles["code"],
                    )
                )
                code_buf = []
                in_code = False
            else:
                flush_table()
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:
            if _is_table_separator(stripped):
                i += 1
                continue
            table_buf.append(_parse_table_row(stripped))
            i += 1
            continue
        else:
            flush_table()

        if stripped == "---":
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
            story.append(Spacer(1, 6))
            i += 1
            continue

        if stripped.startswith("# ") and not stripped.startswith("##"):
            text = stripped[2:].strip()
            if "Technical Reference" in text or text == "World Cup 2026 Prediction Engine":
                story.append(Paragraph(_escape(text), styles["title"]))
            else:
                story.append(Paragraph(_escape(text), styles["h1"]))
            i += 1
            continue

        if stripped.startswith("## "):
            text = re.sub(r"\[.*?\]\(.*?\)", "", stripped[3:]).strip()
            if text.lower().startswith("table of contents"):
                story.append(PageBreak())
            story.append(Paragraph(_escape(text), styles["h2"]))
            i += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(_escape(stripped[4:]), styles["h3"]))
            i += 1
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            body = _escape(stripped[2:].strip())
            story.append(Paragraph(f"&bull; {body}", styles["bullet"]))
            i += 1
            continue

        if stripped.startswith("**") and stripped.endswith("**"):
            story.append(Paragraph(f"<b>{_escape(stripped[2:-2])}</b>", styles["body"]))
            i += 1
            continue

        if not stripped:
            story.append(Spacer(1, 4))
            i += 1
            continue

        story.append(Paragraph(_escape(stripped), styles["body"]))
        i += 1

    flush_table()
    return story


def build_pdf():
    if not MD_PATH.exists():
        raise FileNotFoundError(f"Missing {MD_PATH}")

    md_text = MD_PATH.read_text(encoding="utf-8")
    story = markdown_to_story(md_text)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title="World Cup 2026 Prediction Engine - Technical Guide",
        author="World Cup Simulator Project",
    )

    def on_page(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(
            0.75 * inch,
            0.5 * inch,
            f"World Cup 2026 Prediction Engine  |  Page {doc_.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"PDF written to: {PDF_PATH}")
    print(f"Size: {PDF_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build_pdf()
