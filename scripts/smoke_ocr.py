"""Smoke test: EasyOCR reader on one MMHS150K image."""
import os
import time

import easyocr

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K", "img_resized")
name = sorted(os.listdir(ROOT))[0]
path = os.path.join(ROOT, name)

started = time.time()
reader = easyocr.Reader(["en"], gpu=True, download_enabled=True, verbose=False)
print(f"reader ready in {time.time() - started:.1f}s")

started = time.time()
result = reader.readtext(path, detail=1, paragraph=False)
print(f"ocr {time.time() - started:.2f}s on {name}")
for box, text, conf in result:
    print(f"  {conf:.3f} {text!r}")
