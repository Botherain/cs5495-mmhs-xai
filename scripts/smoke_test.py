"""End-to-end smoke test: loader, target function, deletion curves.

Runs without the MMHS150K archive: one synthetic image and one sentence.
Usage: python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.faithfulness import (  # noqa: E402
    content_positions,
    image_aopc,
    patch_grid,
    random_baseline,
    text_aopc,
)
from mmhs.model import Wrapper, load_model  # noqa: E402

warnings.filterwarnings("ignore")

MODEL_DIR = os.environ.get(
    "MMHS_MODEL_DIR",
    r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model",
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()
    model, tokenizer, processor = load_model(MODEL_DIR, device=device)
    wrapper = Wrapper(model, tokenizer, processor, device=device)
    print(f"[1] model loaded in {time.time() - started:.1f}s on {device}")

    from PIL import Image

    rng = np.random.default_rng(0)
    image = Image.fromarray((rng.random((224, 224, 3)) * 255).astype("uint8"))
    text = "these people are filthy animals go back"
    encoded = wrapper.encode_text([text])
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    pix = wrapper.encode_image([image])["pixel_values"]

    p0 = wrapper.target(ids, mask, pix).item()
    heads = wrapper.head_probs(ids, mask, pix)[0].tolist()
    print(f"[2] target={p0:.4f} heads={[round(h, 3) for h in heads]}")

    positions = content_positions(mask, tokenizer)
    print(f"[3] content tokens={len(positions)} ids={ids[0, positions].tolist()}")

    fill = tokenizer.pad_token_id
    importances = []
    for position in positions:
        probe_mask = mask.clone()
        probe_mask[0, position] = 0
        importances.append(p0 - wrapper.target(ids, probe_mask, pix).item())
    ranking = [positions[i] for i in np.argsort(importances)[::-1]]
    print(f"[4] occlusion importances={[round(v, 4) for v in np.sort(importances)[::-1]]}")

    pad_ids = ids.clone()
    pad_ids[0, positions[len(positions) // 2]] = fill
    print(f"[4b] pad-id substitution moves the pooled EOS slot: "
          f"{int((ids[0] == tokenizer.eos_token_id).int().argmax())} -> "
          f"{int((pad_ids[0] == tokenizer.eos_token_id).int().argmax())}")

    ordered = text_aopc(wrapper, ids, mask, pix, ranking)
    random_text = random_baseline(wrapper, ids, mask, pix, "text", positions)
    print(f"[5] text AOPC ordered={ordered}")
    print(f"    text AOPC random ={random_text}")

    patches = patch_grid(224, 7)
    patch_scores = []
    for row_start, row_end, col_start, col_end in patches:
        probe = pix.clone()
        probe[0, :, row_start:row_end, col_start:col_end] = 0.0
        patch_scores.append(p0 - wrapper.target(ids, mask, probe).item())
    patch_ranking = list(np.argsort(patch_scores)[::-1])
    ordered_img = image_aopc(wrapper, ids, mask, pix, patch_ranking)
    random_img = random_baseline(wrapper, ids, mask, pix, "image", list(range(len(patches))))
    print(f"[6] image AOPC ordered={ordered_img}")
    print(f"    image AOPC random ={random_img}")

    print(f"[7] done in {time.time() - started:.1f}s; peak VRAM "
          f"{torch.cuda.max_memory_allocated() / 1e6:.0f} MB" if device == "cuda" else "[7] done")

    checks = {
        "text ordered > random (del)": ordered["aopc_del"] > random_text["aopc_del"],
        "image ordered > random (del)": ordered_img["aopc_del"] > random_img["aopc_del"],
    }
    for name, passed in checks.items():
        print(f"    {'PASS' if passed else 'FAIL'}  {name}")


if __name__ == "__main__":
    main()
