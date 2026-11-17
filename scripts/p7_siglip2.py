"""P7: H1 replication on the second checkpoint (SigLIP2 fusion), H1 only.

Same four coalitions, same blank-content counterfactual, same target (mean of the
four usable heads). The SigLIP2 checkpoint normalises its tower features and adds
interaction terms before the classifier, so its fusion path is re-implemented
here rather than reused.

Criterion: text-dominance share within 0.10 of the CLIP checkpoint, and the
phi_T / phi_I magnitude ordering unchanged.

Output: refine-logs/p7_siglip2_phi.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.attribution import target_of  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
MODEL_DIR = r"C:\Users\123\AppData\Local\Temp\opencode\siglip2_model"
BATCH = 32


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("modeling_siglip_fusion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_siglip(model_dir: str, device: str):
    from safetensors.torch import load_file
    from transformers import AutoProcessor, AutoTokenizer

    module = load_module(os.path.join(model_dir, "modeling_siglip_fusion.py"))
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as handle:
        raw = json.load(handle)
    raw.pop("auto_map", None)
    raw.pop("architectures", None)
    config = module.SiglipFusionConfig(**raw)
    model = module.MultiModalFusionClassifier(config)

    state = load_file(os.path.join(model_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith("backbone.")]
    print(f"[0] missing(non-backbone)={missing} unexpected={list(unexpected)[:5]}", flush=True)
    if missing:
        raise RuntimeError(f"checkpoint keys not matched: {missing}")

    model.eval().to(device)
    encoder = config.encoder_name
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    processor = AutoProcessor.from_pretrained(encoder)
    return model, tokenizer, processor, config


class SiglipWrapper:
    def __init__(self, model, tokenizer, processor, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device

    def encode_text(self, texts):
        """SigLIP2's tokenizer returns no attention mask (fixed-length padding to
        64 slots, pad id 0), and the released forward only passes a mask when one
        exists, so the audited model really does see the pad slots."""
        return self.tokenizer(list(texts), return_tensors="pt", padding="max_length",
                              truncation=True, max_length=64)

    def encode_image(self, images):
        return self.processor(images=list(images), return_tensors="pt")

    def text_features(self, encoded) -> torch.Tensor:
        """transformers 5 returns a BaseModelOutputWithPooling here, not a tensor."""
        kwargs = {"input_ids": encoded["input_ids"].to(self.device)}
        if "attention_mask" in encoded:
            kwargs["attention_mask"] = encoded["attention_mask"].to(self.device)
        return self.model.backbone.get_text_features(**kwargs).pooler_output

    def image_features(self, encoded) -> torch.Tensor:
        return self.model.backbone.get_image_features(
            pixel_values=encoded["pixel_values"].to(self.device)
        ).pooler_output

    @torch.no_grad()
    def features(self, texts, images) -> tuple:
        text_list = list(texts)
        image_list = list(images)
        text_feats, image_feats = [], []
        for start in range(0, len(text_list), BATCH):
            encoded = self.encode_text(text_list[start:start + BATCH])
            text_feats.append(self.text_features(encoded).cpu())
        for start in range(0, len(image_list), BATCH):
            encoded = self.encode_image(image_list[start:start + BATCH])
            image_feats.append(self.image_features(encoded).cpu())
        return torch.cat(text_feats), torch.cat(image_feats)


def fusion_probs(model, tfeat_raw, vfeat_raw,
                 text_present=None, image_present=None) -> torch.Tensor:
    """SigLIP2 fusion: normalise, mask by presence, gate, add interaction terms."""
    if tfeat_raw.dim() == 1:
        tfeat_raw = tfeat_raw.unsqueeze(0)
    if vfeat_raw.dim() == 1:
        vfeat_raw = vfeat_raw.unsqueeze(0)
    batch = tfeat_raw.shape[0]
    device = tfeat_raw.device
    if text_present is None:
        text_present = torch.ones(batch, device=device)
    if image_present is None:
        image_present = torch.ones(batch, device=device)

    tfeat = F.normalize(tfeat_raw, dim=-1) * text_present.unsqueeze(1)
    vfeat = F.normalize(vfeat_raw, dim=-1) * image_present.unsqueeze(1)
    t_p = model.proj_t(tfeat)
    v_p = model.proj_i(vfeat)
    zt = torch.tanh(model.g_t(t_p))
    zi = torch.tanh(model.g_i(v_p))
    presence = torch.stack([text_present, image_present], dim=1)
    gate = torch.sigmoid(model.gate(torch.cat([t_p, v_p, presence], dim=1)))
    fused = torch.where(
        (image_present < 0.5).unsqueeze(1), zt,
        torch.where((text_present < 0.5).unsqueeze(1), zi, gate * zt + (1.0 - gate) * zi),
    )
    fused = model.ln_fused(fused)
    feat = torch.cat([fused, t_p, v_p, torch.abs(t_p - v_p), t_p * v_p], dim=1)
    return torch.sigmoid(model.cls(feat))


def main() -> None:
    from PIL import Image

    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    with open(os.path.join(OUT, "p2_phi.jsonl"), encoding="utf-8") as handle:
        ids = [json.loads(line)["id"] for line in handle if line.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor, config = load_siglip(MODEL_DIR, device)
    wrapper = SiglipWrapper(model, tokenizer, processor, device)

    texts = [gt[tweet_id].get("tweet_text", "") for tweet_id in ids]
    images = [Image.open(os.path.join(ROOT, "img_resized", f"{t}.jpg")).convert("RGB") for t in ids]
    tfeat, vfeat = wrapper.features(texts, images)
    print(f"[1] features {tfeat.shape} {vfeat.shape}", flush=True)

    with torch.no_grad():
        blank = wrapper.encode_text([""])
        tblank = wrapper.text_features(blank).cpu().squeeze(0)
        blank_pixel = torch.zeros(1, 3, 224, 224, device=device)
        vblank = model.backbone.get_image_features(pixel_values=blank_pixel).pooler_output.cpu().squeeze(0)

    rows = []
    phi_t_all, phi_i_all = [], []
    for start in range(0, len(ids), 128):
        stop = min(start + 128, len(ids))
        t = tfeat[start:stop].to(device)
        v = vfeat[start:stop].to(device)
        tb = tblank.to(device).expand_as(t)
        vb = vblank.to(device).expand_as(v)
        with torch.no_grad():
            v_empty = target_of(fusion_probs(model, tb, vb))
            v_text = target_of(fusion_probs(model, t, vb))
            v_image = target_of(fusion_probs(model, tb, v))
            v_both = target_of(fusion_probs(model, t, v))
            phi_t = 0.5 * (v_text - v_empty) + 0.5 * (v_both - v_image)
            phi_i = 0.5 * (v_image - v_empty) + 0.5 * (v_both - v_text)
        phi_t_all.append(phi_t.cpu().numpy())
        phi_i_all.append(phi_i.cpu().numpy())
        for row in range(stop - start):
            rows.append({
                "id": ids[start + row],
                "phi_t": float(phi_t[row]), "phi_i": float(phi_i[row]),
                "v_empty": float(v_empty[row]), "v_both": float(v_both[row]),
                "target": float(v_both[row]),
            })
        print(f"    {stop}/{len(ids)}", flush=True)

    phi_t_all = np.concatenate(phi_t_all)
    phi_i_all = np.concatenate(phi_i_all)
    dominant = float((np.abs(phi_t_all) > np.abs(phi_i_all)).mean())

    with open(os.path.join(OUT, "p7_siglip2_phi.jsonl"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    with open(os.path.join(OUT, "p2_phi.jsonl"), encoding="utf-8") as handle:
        main_rows = [json.loads(line) for line in handle if line.strip()]
    main_phi_t = np.array([row["phi_t"] for row in main_rows])
    main_phi_i = np.array([row["phi_i"] for row in main_rows])
    main_dominant = float((np.abs(main_phi_t) > np.abs(main_phi_i)).mean())

    summary = {
        "n": len(rows),
        "checkpoint": "Amirhossein75/siglip2-base-mmhs150k-fusion",
        "encoder": config.encoder_name,
        "thresholds": config.thresholds,
        "phi_t": {"mean": float(phi_t_all.mean()), "median": float(np.median(phi_t_all))},
        "phi_i": {"mean": float(phi_i_all.mean()), "median": float(np.median(phi_i_all))},
        "text_dominance_share": dominant,
        "phi_i_negative_share": float((phi_i_all < 0).mean()),
        "main_text_dominance_share": main_dominant,
        "dominance_gap": abs(dominant - main_dominant),
        "criterion": "|share_siglip2 - share_main| < 0.10 and the same magnitude ordering",
        "passes": bool(abs(dominant - main_dominant) < 0.10
                       and (np.abs(phi_t_all).mean() > np.abs(phi_i_all).mean())),
        "scale_note": "SigLIP2 fusion adds interaction terms, so absolute phi values are not comparable across checkpoints; only shares and ordering are.",
    }
    with open(os.path.join(OUT, "p7_siglip2_phi.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[2] " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
