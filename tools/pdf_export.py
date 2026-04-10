from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF


class ResearchPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Research Agent Report", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _strip_markdown(text: str) -> str:
    """Light Markdown stripping for plain-text PDF rendering."""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text


def export_pdf(markdown_text: str, output_path: str | Path) -> Path:
    """Convert a Markdown report to a styled PDF using fpdf2."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = ResearchPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    lines = markdown_text.split("\n")
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(4)
            pdf.cell(0, 10, stripped[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(40, 40, 40)
            pdf.ln(3)
            pdf.cell(0, 8, stripped[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(60, 60, 60)
            pdf.ln(2)
            pdf.cell(0, 7, stripped[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif stripped.startswith("---"):
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)
        elif stripped == "":
            pdf.ln(3)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            clean = _strip_markdown(stripped)
            pdf.multi_cell(0, 5, clean)
            pdf.ln(1)

    # Embed chart images if referenced
    chart_pattern = re.compile(r"!\[.*?\]\((.+?)\)")
    for match in chart_pattern.finditer(markdown_text):
        img_path = Path(match.group(1))
        if img_path.exists():
            pdf.add_page()
            pdf.image(str(img_path), x=15, w=180)

    pdf.output(str(output_path))
    return output_path
