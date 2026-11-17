"""Export the executed notebook to a single-file HTML and an A4 PDF.

Chrome caches file:// pages aggressively, and the notebook path contains spaces, so the
HTML is copied to a unique temp name before printing; without that the PDF can silently
come out one build stale.

Usage: python scripts/export_pdf.py
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK = os.path.join(ROOT, "Course Project-2026A.ipynb")
HTML = os.path.join(ROOT, "Course Project-2026A.html")
PDF = os.path.join(ROOT, "Course Project-2026A.pdf")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

PRINT_CSS = """
<style>
  @page { size: A4; margin: 15mm 13mm 15mm 13mm; }
  body { background: #fff; }
  .jp-Notebook { padding: 0 !important; max-width: 100% !important; }
  .jp-Cell { padding: 2px 0 !important; }
  .jp-RenderedImage img, .jp-OutputArea-output img { max-width: 100% !important; }
  .jp-RenderedHTMLCommon table { font-size: 11.5px; }
  .jp-RenderedHTMLCommon pre { white-space: pre-wrap; font-size: 11px; }
  h1 { font-size: 20px; } h2 { font-size: 15.5px; } h3 { font-size: 13.5px; }
  .jp-RenderedHTMLCommon { font-size: 12.4px; line-height: 1.5; }
  .jp-InputArea pre { font-size: 10.5px; }
</style>
"""


def main() -> None:
    subprocess.run([
        sys.executable, "-m", "nbconvert", "--to", "html", "--embed-images",
        NOTEBOOK, "--output", "cp_stage",
    ], check=True, cwd=ROOT)

    staged = os.path.join(ROOT, "cp_stage.html")
    hn = io.open(staged, encoding="utf-8").read().replace("</head>", PRINT_CSS + "</head>", 1)
    with io.open(HTML, "w", encoding="utf-8", newline="") as handle:
        handle.write(hn)

    temp_html = os.path.join(os.environ.get("TEMP", "."), f"cp_print_{int(time.time())}.html")
    with io.open(temp_html, "w", encoding="utf-8", newline="") as handle:
        handle.write(hn)

    if os.path.exists(PDF):
        os.remove(PDF)
    url = "file:///" + temp_html.replace("\\", "/").replace(" ", "%20")
    result = subprocess.run([
        CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
        "--no-pdf-header-footer", f"--print-to-pdf={PDF}", url,
    ], capture_output=True, text=True)
    for _ in range(60):
        if os.path.exists(PDF) and os.path.getsize(PDF) > 10000:
            break
        time.sleep(1)
    os.remove(staged)
    os.remove(temp_html)
    print("pdf rc", result.returncode,
          "bytes", os.path.getsize(PDF) if os.path.exists(PDF) else "MISSING")


if __name__ == "__main__":
    main()
