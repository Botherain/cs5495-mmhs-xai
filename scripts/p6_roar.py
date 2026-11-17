"""P6: modality-level ROAR --- retrain the fusion head under modality dropout.

Three arms, all evaluated on the same held-out 1,000 samples:

  arm1  released fusion head (proj / gate / shared / heads untouched)
  arm2  head re-initialised and refit on 2,000 samples with both modalities
  arm3  same refit, plus modality dropout (p=0.3 per modality, replaced by the
        blank feature) so the head cannot lean on text alone

Encoders stay frozen; the cached tower features are the only input. Arm 2 is the
control for arm 3 (capacity and data held fixed), which is the pre-registered
comparison: phi_I share rises while macro-F1 drops by at most 0.01.

Output: refine-logs/p6_roar.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmhs.attribution import encode_features, fusion_probs, target_of  # noqa: E402
from mmhs.data import normalize_labels  # noqa: E402
from mmhs.model import Wrapper, load_model  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MMHS150K")
OUT = os.path.join(os.path.dirname(__file__), "..", "refine-logs")
THRESHOLDS = torch.tensor([0.3, 0.7, 0.5, 0.3, 0.55])
SEEDS = (0, 1, 2)
DROPOUT = 0.3
EPOCHS = 150
BATCH = 128
LR = 1e-3


class FusionHead(nn.Module):
    """Same topology as the checkpoint's fusion block, re-initialised.

    The released config has head_hidden_dim=256, so each task head is a two-layer
    MLP, not a single linear map; keeping that shape means arm 2 differs from arm 1
    only in its weights, not in capacity.
    """

    def __init__(self, base, fusion_dim: int = 512, heads: int = 5, head_hidden: int = 256):
        super().__init__()
        self.proj_t = nn.Linear(base.proj_t.in_features, fusion_dim)
        self.proj_i = nn.Linear(base.proj_i.in_features, fusion_dim)
        self.g_t = nn.Linear(fusion_dim, fusion_dim)
        self.g_i = nn.Linear(fusion_dim, fusion_dim)
        self.gate = nn.Linear(fusion_dim * 2 + 2, fusion_dim)
        self.shared_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(fusion_dim, fusion_dim), nn.GELU(), nn.Dropout(0.2)
        )
        if head_hidden and head_hidden > 0:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(fusion_dim, head_hidden), nn.GELU(),
                    nn.Dropout(0.1), nn.Linear(head_hidden, 1),
                ) for _ in range(heads)
            ])
        else:
            self.heads = nn.ModuleList([nn.Linear(fusion_dim, 1) for _ in range(heads)])

    def forward(self, tfeat, vfeat, text_present=None, image_present=None):
        t = self.proj_t(tfeat)
        v = self.proj_i(vfeat)
        batch = t.shape[0]
        device = t.device
        if text_present is None:
            text_present = torch.ones(batch, device=device)
        if image_present is None:
            image_present = torch.ones(batch, device=device)
        presence = torch.stack([text_present, image_present], dim=1)
        zt = torch.tanh(self.g_t(t))
        zi = torch.tanh(self.g_i(v))
        gate = torch.sigmoid(self.gate(torch.cat([t, v, presence], dim=1)))
        fused = torch.where(
            (image_present < 0.5).unsqueeze(1), zt,
            torch.where((text_present < 0.5).unsqueeze(1), zi, gate * zt + (1.0 - gate) * zi),
        )
        shared = self.shared_head(fused)
        return torch.stack([head(shared).squeeze(-1) for head in self.heads], dim=1)


def macro_f1(probs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    scores = []
    for index in range(labels.shape[1]):
        y = labels[:, index]
        pred = (probs[:, index] > float(THRESHOLDS[index])).astype(int)
        scores.append(f1_score(y, pred, zero_division=0))
    return float(np.mean(scores))


def modality_shares(head, tfeat, vfeat, tblank, vblank, device) -> tuple:
    with torch.no_grad():
        tf = torch.as_tensor(tfeat, device=device)
        vf = torch.as_tensor(vfeat, device=device)
        tb = torch.as_tensor(tblank, device=device).expand_as(tf)
        vb = torch.as_tensor(vblank, device=device).expand_as(vf)
        v_empty = target_of(torch.sigmoid(head(tb, vb)))
        v_text = target_of(torch.sigmoid(head(tf, vb)))
        v_image = target_of(torch.sigmoid(head(tb, vf)))
        v_both = target_of(torch.sigmoid(head(tf, vf)))
        phi_t = 0.5 * (v_text - v_empty) + 0.5 * (v_both - v_image)
        phi_i = 0.5 * (v_image - v_empty) + 0.5 * (v_both - v_text)
    share = (phi_i.abs() / (phi_t.abs() + phi_i.abs() + 1e-12)).cpu().numpy()
    dominance = (phi_i.abs() > phi_t.abs()).float().mean().item()
    return float(np.mean(share)), float(dominance)


def main() -> None:
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--out", default="p6_roar.json")
    args = parser.parse_args()
    epochs = args.epochs
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(
        os.environ.get("MMHS_MODEL_DIR", r"C:\Users\123\AppData\Local\Temp\opencode\mmhs_model"),
        device=device,
    )
    wrapper = Wrapper(model, tokenizer, processor, device=device)

    with open(os.path.join(ROOT, "MMHS150K_GT.json"), encoding="utf-8") as handle:
        gt = json.load(handle)
    with open(os.path.join(OUT, "p2_phi.jsonl"), encoding="utf-8") as handle:
        ids = [json.loads(line)["id"] for line in handle if line.strip()]

    texts = [gt[tweet_id].get("tweet_text", "") for tweet_id in ids]
    images = [Image.open(os.path.join(ROOT, "img_resized", f"{tweet_id}.jpg")).convert("RGB")
              for tweet_id in ids]
    labels = np.zeros((len(ids), 5), dtype=np.float32)
    for row, tweet_id in enumerate(ids):
        for label in set(normalize_labels(gt[tweet_id]["labels"])):
            if label > 0:
                labels[row, label - 1] = 1.0

    tfeat, vfeat = encode_features(wrapper, texts, images)
    print(f"[1] features {tfeat.shape} {vfeat.shape}", flush=True)

    with torch.no_grad():
        blank_ids, blank_mask = wrapper.blank_text(1)
        tblank = model._encode_text(blank_ids.to(device), blank_mask.to(device)).cpu().squeeze(0)
        vblank = model._encode_image(torch.zeros(1, 3, 224, 224, device=device)).cpu().squeeze(0)

    rng = np.random.default_rng(0)
    # stratified split: 2,000 refit / 1,000 evaluation
    primary = labels.argmax(axis=1)
    refit_index, eval_index = [], []
    for value in np.unique(primary):
        pool = np.where(primary == value)[0]
        rng.shuffle(pool)
        cut = int(round(len(pool) * 2 / 3))
        refit_index.extend(pool[:cut].tolist())
        eval_index.extend(pool[cut:].tolist())
    refit_index = np.array(refit_index)
    eval_index = np.array(eval_index)
    print(f"[2] refit={len(refit_index)} eval={len(eval_index)}", flush=True)

    eval_labels = labels[eval_index]
    eval_t = tfeat[eval_index].to(device)
    eval_v = vfeat[eval_index].to(device)

    with torch.no_grad():
        released = fusion_probs(model, eval_t, eval_v).cpu().numpy()
    results = {
        "n_refit": int(len(refit_index)),
        "n_eval": int(len(eval_index)),
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "arms": {
            "arm1_released": {
                "macro_f1": macro_f1(released, eval_labels),
                "phi_i_share": None,
                "image_dominance": None,
            }
        },
    }
    # arm1 shares computed with the released fusion block, using its own forward
    with torch.no_grad():
        tf, vf = eval_t, eval_v
        tb = tblank.to(device).expand_as(tf)
        vb = vblank.to(device).expand_as(vf)
        v_empty = target_of(fusion_probs(model, tb, vb))
        v_text = target_of(fusion_probs(model, tf, vb))
        v_image = target_of(fusion_probs(model, tb, vf))
        v_both = target_of(fusion_probs(model, tf, vf))
        phi_t = 0.5 * (v_text - v_empty) + 0.5 * (v_both - v_image)
        phi_i = 0.5 * (v_image - v_empty) + 0.5 * (v_both - v_text)
    results["arms"]["arm1_released"]["phi_i_share"] = float(
        (phi_i.abs() / (phi_t.abs() + phi_i.abs() + 1e-12)).mean()
    )
    results["arms"]["arm1_released"]["image_dominance"] = float(
        (phi_i.abs() > phi_t.abs()).float().mean()
    )
    print(f"    arm1 released: F1={results['arms']['arm1_released']['macro_f1']:.4f} "
          f"phi_I share={results['arms']['arm1_released']['phi_i_share']:.4f}", flush=True)

    refit_t = tfeat[refit_index]
    refit_v = vfeat[refit_index]
    refit_labels = torch.as_tensor(labels[refit_index], device=device)
    # torch's pos_weight multiplies the POSITIVE term of the BCE, so balancing a rare
    # head needs n_neg / n_pos (> 1), not the other way round. The first version of
    # this script used n_pos / n_pos-complement, which down-weighted exactly the rare
    # heads (religion at 2.7% prevalence) and partly explains the refit arms' low F1.
    positives = refit_labels.sum(dim=0)
    negatives = len(refit_index) - positives
    pos_weight = (negatives / (positives + 1e-6)).clamp(0.1, 50.0)

    for arm_name, dropout in (("arm2_refit", 0.0), ("arm3_refit_dropout", DROPOUT)):
        f1s, train_f1s, shares, dominance = [], [], [], []
        for seed in seeds:
            torch.manual_seed(seed)
            head = FusionHead(model).to(device)
            optimiser = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
            generator = torch.Generator().manual_seed(seed)
            for epoch in range(epochs):
                head.train()
                order = torch.randperm(len(refit_index), generator=generator)
                for start in range(0, len(order), BATCH):
                    chunk = order[start:start + BATCH]
                    t = refit_t[chunk].to(device)
                    v = refit_v[chunk].to(device)
                    if dropout > 0:
                        drop_t = (torch.rand(len(chunk), generator=generator) < dropout).to(device)
                        drop_v = (torch.rand(len(chunk), generator=generator) < dropout).to(device)
                        t = torch.where(drop_t.unsqueeze(1), tblank.to(device).expand_as(t), t)
                        v = torch.where(drop_v.unsqueeze(1), vblank.to(device).expand_as(v), v)
                    logits = head(t, v)
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, refit_labels[chunk], pos_weight=pos_weight
                    )
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    optimiser.step()

            head.eval()
            with torch.no_grad():
                probs = torch.sigmoid(head(eval_t, eval_v)).cpu().numpy()
                # train-set F1 is the sanity check on the refit itself: if the head
                # cannot fit 2,000 samples it cannot be a fair comparison arm.
                train_probs = torch.sigmoid(
                    head(refit_t.to(device), refit_v.to(device))
                ).cpu().numpy()
            share, dom = modality_shares(head, eval_t, eval_v, tblank, vblank, device)
            f1s.append(macro_f1(probs, eval_labels))
            train_f1s.append(macro_f1(train_probs, labels[refit_index]))
            shares.append(share)
            dominance.append(dom)
            print(f"    {arm_name} seed={seed} evalF1={f1s[-1]:.4f} trainF1={train_f1s[-1]:.4f} "
                  f"phi_I share={share:.4f} dominance={dom:.3f}", flush=True)

        results["arms"][arm_name] = {
            "macro_f1": float(np.mean(f1s)),
            "macro_f1_sd": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "train_macro_f1": float(np.mean(train_f1s)),
            "phi_i_share": float(np.mean(shares)),
            "phi_i_share_sd": float(np.std(shares, ddof=1)) if len(shares) > 1 else 0.0,
            "image_dominance": float(np.mean(dominance)),
            "f1_per_seed": f1s,
            "phi_i_share_per_seed": shares,
        }

    arm2 = results["arms"]["arm2_refit"]
    arm3 = results["arms"]["arm3_refit_dropout"]
    delta_share = arm3["phi_i_share"] - arm2["phi_i_share"]
    delta_f1 = arm3["macro_f1"] - arm2["macro_f1"]
    results["comparison"] = {
        "delta_phi_i_share": float(delta_share),
        "delta_macro_f1": float(delta_f1),
        "criterion": "arm3 - arm2 phi_I share > 0 and macro-F1 drop <= 0.01",
        "passes": bool(delta_share > 0 and delta_f1 >= -0.01),
    }
    results["epochs"] = int(epochs)
    with open(os.path.join(OUT, args.out), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("[3] " + json.dumps(results["comparison"], indent=2))


if __name__ == "__main__":
    main()
