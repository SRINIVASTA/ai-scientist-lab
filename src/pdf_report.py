import os
import re
from fpdf import FPDF


def generate_research_pdf(
    markdown_text: str, plot_path: str = "outputs/metrics_plot.png"
) -> bytes:
    """Converts the Markdown research paper and diagnostic chart into a clean PDF."""
    pdf = FPDF()
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header Banner
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 45, 90)
    pdf.cell(
        0, 10, "Autonomous AI Scientist Lab | Research Report", ln=True
    )
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(5)

    # Strip YAML frontmatter block for cleaner PDF reading
    clean_md = re.sub(r"^---[\s\(\S\)]*?---\s*", "", markdown_text.strip())

    # Replace Unicode characters unsupported by core Latin-1 Helvetica
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
        clean_md = clean_md.replace(old_ch, new_ch)
    clean_md = clean_md.encode("latin-1", "replace").decode("latin-1")

    for line in clean_md.splitlines():
        stripped = line.strip()
        pdf.set_x(pdf.l_margin)

        if not stripped:
            pdf.ln(3)
            continue

        # Skip raw markdown table divider lines like |---|---|
        if re.match(r"^\|?[\s\-:|]+\|?\$", stripped):
            continue

        clean_line = stripped.replace("**", "")

        if stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 15)
            pdf.set_text_color(15, 35, 75)
            pdf.multi_cell(pdf.epw, 8, clean_line[2:], wrapmode="CHAR")
            pdf.ln(2)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(25, 65, 120)
            pdf.multi_cell(pdf.epw, 7, clean_line[3:], wrapmode="CHAR")
            pdf.ln(1)
        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(pdf.epw, 6, clean_line[4:], wrapmode="CHAR")
        elif stripped.startswith("```"):
            continue
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(pdf.epw, 5.5, clean_line, wrapmode="CHAR")

    # Embed Validation Metrics Chart if available
    if os.path.exists(plot_path):
        pdf.ln(5)
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(25, 65, 120)
        pdf.cell(0, 8, "Validation Metrics Diagnostic Plot", ln=True)
        pdf.image(plot_path, w=130)

    return bytes(pdf.output())
