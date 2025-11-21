"""Generate PDF documentation for returns_dist_fit_v2.

Requires: fpdf (pip install fpdf)
Falls back to a plain text .txt export if fpdf not installed.
"""
from pathlib import Path
import sys

def load_markdown(md_path: Path) -> str:
    return md_path.read_text(encoding='utf-8')

def write_plain_text(txt_path: Path, content: str):
    txt_path.write_text(content, encoding='utf-8')
    print(f"Plain text fallback written to {txt_path}")

def make_pdf(pdf_path: Path, content: str):
    try:
        from fpdf import FPDF
    except ImportError:
        print("fpdf not installed. Run: pip install fpdf")
        raise
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Courier", size=9)
    for line in content.splitlines():
        # Basic wrapping for long lines
        if len(line) <= 110:
            pdf.cell(0, 5, txt=line, ln=1)
        else:
            # wrap manually
            start = 0
            while start < len(line):
                pdf.cell(0, 5, txt=line[start:start+110], ln=1)
                start += 110
    pdf.output(str(pdf_path))
    print(f"PDF written to {pdf_path}")

def main():
    docs_dir = Path("docs")
    md_file = docs_dir / "returns_dist_fit_v2.md"
    if not md_file.exists():
        print(f"Documentation markdown not found: {md_file}")
        sys.exit(1)
    content = load_markdown(md_file)
    pdf_path = docs_dir / "returns_dist_fit_v2_docs.pdf"
    try:
        make_pdf(pdf_path, content)
    except ImportError:
        # Fallback to .txt if PDF generation not possible
        txt_path = docs_dir / "returns_dist_fit_v2_docs.txt"
        write_plain_text(txt_path, content)

if __name__ == "__main__":
    main()
