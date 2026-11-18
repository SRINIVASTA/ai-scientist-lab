import os
import re
from fpdf import FPDF
from fpdf.fonts import FontFace


def _sanitize_latin1(text: str) -> str:
    replacements = {
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u202f": " ",
        "\u2248": "~",
        "\u2192": "->",
    }
    for old_ch, new_ch in replacements.items():
        text = text.replace(old_ch, new_ch)
    return text.encode("latin-1", "replace").decode("latin-1")


def _render_markdown_table(pdf: FPDF, table_lines: list[str]) -> None:
    """Renders collected Markdown pipe rows as a clean bordered PDF table."""
    rows = []
    for raw_line in table_lines:
        # Skip separator rows like |---|---|
        if re.match(r"^\|?[\s\-:|]+\|?\$", raw_line.strip()):
            continue
        cells = [
            c.strip().replace("**", "")
            for c in raw_line.strip().strip("|").split("|")
        ]
        if cells:
            rows.append(cells)

    if not rows:
        return

    # Ensure uniform column count across all rows
    max_cols = max(len(r) for r in rows)
    normalized_rows = [r + [""] * (max_cols - len(r)) for r in rows]

    pdf.ln(3)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)

    header_style = FontFace(
        emphasis="BOLD", color=(255, 255, 255), fill_color=(25, 65, 120)
    )
    with pdf.table(
        borders_layout="SINGLE_TOP_LINE",
        cell_fill_color=(245, 248, 252),
        cell_fill_mode="ROWS",
        headings_style=header_style,
        line_height=6,
        padding=2,
        width=pdf.epw,
    ) as table:
        for row_data in normalized_rows:
            row = table.row()
            for cell_text in row_data:
                row.cell(cell_text)
    pdf.ln(3)


def generate_research_pdf(
    markdown_text: str, plot_path: str = "outputs/metrics_plot.png"
) -> bytes:
    """Converts the Markdown research paper and diagnostic chart into a publication-grade PDF."""
    pdf = FPDF()
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Extract optional YAML metadata before stripping
    title_match = re.search(
        r'^title:\s*["\']?(.*?)["\']?\s*\$', markdown_text, re.MULTILINE
    )
    date_match = re.search(
        r'^date:\s*["\']?(.*?)["\']?\s*\$', markdown_text, re.MULTILINE
    )
    paper_title = (
        _sanitize_latin1(title_match.group(1))
        if title_match
        else "Autonomous AI Scientist Research Report"
    )
    paper_date = (
        _sanitize_latin1(date_match.group(1)) if date_match else ""
    )

    # Top Header Banner
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(90, 110, 140)
    banner_text = "AUTONOMOUS AI SCIENTIST LAB | QUANTITATIVE RESEARCH NOTE"
    if paper_date and paper_date.lower() != "null":
        banner_text += f"  ({paper_date})"
    pdf.cell(0, 6, banner_text, ln=True)

    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(20, 45, 90)
    pdf.multi_cell(pdf.epw, 8, paper_title, wrapmode="CHAR")

    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.l_margin, pdf.get_y() + 2, pdf.w - pdf.r_margin, pdf.get_y() + 2)
    pdf.ln(6)

    # Strip fenced YAML block and raw top metadata lines
    clean_md = re.sub(r"^---[\s\(\S\)]*?---\s*", "", markdown_text.strip())
    clean_md = _sanitize_latin1(clean_md)

    lines = clean_md.splitlines()
    table_buffer: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip un-fenced frontmatter lines if present at top
        if re.match(r"^(title|date|tags)\s*:", stripped, re.IGNORECASE):
            continue

        # Collect consecutive Markdown table lines
        if stripped.startswith("|") and "|" in stripped[1:]:
            table_buffer.append(stripped)
            continue
        elif table_buffer:
            _render_markdown_table(pdf, table_buffer)
            table_buffer = []

        pdf.set_x(pdf.l_margin)

        if not stripped:
            pdf.ln(3)
            continue

        if stripped.startswith("```"):
            continue

        clean_line = stripped.replace("**", "")

        if stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(15, 35, 75)
            pdf.multi_cell(pdf.epw, 8, clean_line[2:], wrapmode="CHAR")
            pdf.ln(2)
        elif stripped.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(25, 65, 120)
            pdf.multi_cell(pdf.epw, 7, clean_line[3:], wrapmode="CHAR")
            pdf.ln(1)
        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(pdf.epw, 6, clean_line[4:], wrapmode="CHAR")
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(pdf.epw, 5.5, clean_line, wrapmode="CHAR")

    # Flush any trailing table at end of document
    if table_buffer:
        _render_markdown_table(pdf, table_buffer)

    # Embed Validation Metrics Chart if available
    if os.path.exists(plot_path):
        pdf.ln(5)
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(25, 65, 120)
        pdf.cell(0, 8, "Validation Metrics Diagnostic Plot", ln=True)
        pdf.image(plot_path, w=130)

    return bytes(pdf.output())
