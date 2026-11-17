"""Deletion / insertion faithfulness curves for text tokens and image patches.

Text deletion does *not* replace token ids: CLIP's pad and EOS share id 49407 and
the text tower pools the first EOS by argmax, so writing a pad into a sentence
moves the pooled position forward and truncates the tail. Deletion therefore sets
``attention_mask`` to 0 at the chosen positions and leaves ``input_ids`` untouched.
Image deletion replaces patches with the mean colour (zeros in CLIP-normalised
space). Insertion starts from the blank modality and restores features in ranking
order, keeping the other modality real; its baseline is the target with that
modality blanked.

Every function returns the full per-fraction curve, the AOPC (curve mean), and
comprehensiveness / sufficiency at 20% and 50%.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

FRACTIONS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
HEADLINE_FRACTIONS = (0.2, 0.5)

# Published per-head thresholds shipped with the checkpoint (racist..otherhate).
HEAD_THRESHOLDS = (0.3, 0.7, 0.5, 0.3, 0.55)
USABLE_HEADS = (0, 1, 2, 4)


def content_positions(attention_mask: torch.Tensor, tokenizer) -> List[int]:
    """Positions of real tokens, excluding the two CLIP sentinels."""
    mask = attention_mask[0].tolist()
    positions = [i for i, flag in enumerate(mask) if flag == 1]
    if len(positions) <= 2:
        return positions
    return positions[1:-1]


def patch_grid(image_size: int = 224, grid: int = 7) -> List[tuple]:
    step = image_size // grid
    return [
        (row * step, (row + 1) * step, col * step, (col + 1) * step)
        for row in range(grid)
        for col in range(grid)
    ]


def predicted_label_set(probs: Sequence[float]) -> tuple:
    """Binary decision per usable head using the checkpoint's own thresholds."""
    return tuple(int(probs[head] > HEAD_THRESHOLDS[head]) for head in USABLE_HEADS)


def is_flip(before: Sequence[float], after: Sequence[float]) -> int:
    return int(predicted_label_set(before) != predicted_label_set(after))


def _text_probe(wrapper, ids, mask, pix, chosen, direction: str, positions) -> float:
    """Reveal or hide tokens by flipping ``attention_mask``; ids never change.

    Writing a pad id into the middle of the sentence would move CLIP's argmax-EOS
    pooling position, so masking is the only substitution that keeps the pooled
    slot fixed.
    """
    probe_mask = mask.clone()
    chosen = set(chosen)
    if direction == "ins":
        for position in positions:
            probe_mask[0, position] = 0
        for position in chosen:
            probe_mask[0, position] = mask[0, position]
    elif direction == "keep":
        for position in positions:
            if position not in chosen:
                probe_mask[0, position] = 0
    else:
        for position in chosen:
            probe_mask[0, position] = 0
    return wrapper.target(ids, probe_mask, pix).item()


def _image_probe(wrapper, ids, mask, pix, chosen, direction: str) -> float:
    if direction == "ins":
        probe = torch.zeros_like(pix)
    elif direction == "keep":
        probe = torch.zeros_like(pix)
    else:
        probe = pix.clone()
    chosen_set = {tuple(patch) for patch in chosen}
    for patch in patch_grid(pix.shape[-1], 7):
        row_start, row_end, col_start, col_end = patch
        if direction == "ins":
            if patch in chosen_set:
                probe[0, :, row_start:row_end, col_start:col_end] = pix[0, :, row_start:row_end, col_start:col_end]
        elif direction == "keep":
            if patch not in chosen_set:
                continue
            probe[0, :, row_start:row_end, col_start:col_end] = pix[0, :, row_start:row_end, col_start:col_end]
        else:
            if patch in chosen_set:
                probe[0, :, row_start:row_end, col_start:col_end] = 0.0
    if direction == "keep":
        probe = torch.zeros_like(pix)
        for row_start, row_end, col_start, col_end in chosen:
            probe[0, :, row_start:row_end, col_start:col_end] = pix[0, :, row_start:row_end, col_start:col_end]
    return wrapper.target(ids, mask, probe).item()


def _summarise(fractions, del_values, ins_values) -> Dict[str, object]:
    result = {
        "fractions": list(fractions),
        "curve_del": [float(v) for v in del_values],
        "curve_ins": [float(v) for v in ins_values],
        "aopc_del": float(np.mean(del_values)),
        "aopc_ins": float(np.mean(ins_values)),
    }
    for fraction in HEADLINE_FRACTIONS:
        index = min(range(len(fractions)), key=lambda i: abs(fractions[i] - fraction))
        result[f"comp_{int(fraction * 100)}"] = float(del_values[index])
        result[f"suff_{int(fraction * 100)}"] = float(ins_values[index])
    return result


def text_aopc(wrapper, ids, mask, pix, ranking: Sequence[int],
              fractions: Sequence[float] = FRACTIONS) -> Dict[str, object]:
    """AOPC over tokens; ``ranking`` holds absolute positions in the 77-slot row."""
    positions = content_positions(mask, wrapper.tokenizer)
    if not positions:
        return {"aopc_del": float("nan"), "aopc_ins": float("nan")}

    ranking = [p for p in ranking if p in set(positions)] or list(positions)
    steps = sorted({max(1, int(round(len(positions) * fraction))) for fraction in fractions})
    kept_fractions = [step / len(positions) for step in steps]

    p_real = wrapper.target(ids, mask, pix).item()
    blank_ids, blank_mask = wrapper.blank_text(1)
    p_blank = wrapper.target(blank_ids, blank_mask, pix).item()

    del_values, ins_values = [], []
    for step in steps:
        chosen = ranking[:step]
        del_values.append(p_real - _text_probe(wrapper, ids, mask, pix, chosen, "del", positions))
        ins_values.append(_text_probe(wrapper, ids, mask, pix, chosen, "ins", positions) - p_blank)

    return _summarise(kept_fractions, del_values, ins_values)


def image_aopc(wrapper, ids, mask, pix, ranking: Sequence[int],
               grid: int = 7, fractions: Sequence[float] = FRACTIONS) -> Dict[str, object]:
    patches = patch_grid(pix.shape[-1], grid)
    ranking = list(ranking) if len(ranking) else list(range(len(patches)))
    steps = sorted({max(1, int(round(len(patches) * fraction))) for fraction in fractions})
    kept_fractions = [step / len(patches) for step in steps]

    p_real = wrapper.target(ids, mask, pix).item()
    p_blank = wrapper.target(ids, mask, torch.zeros_like(pix)).item()

    del_values, ins_values = [], []
    for step in steps:
        chosen = [patches[index] for index in ranking[:step]]
        del_values.append(p_real - _image_probe(wrapper, ids, mask, pix, chosen, "del"))
        ins_values.append(_image_probe(wrapper, ids, mask, pix, chosen, "ins") - p_blank)

    return _summarise(kept_fractions, del_values, ins_values)


def random_ranking(items: Sequence[int], repeats: int = 5, seed: int = 0) -> List[List[int]]:
    """Permutations of the actual item identities (token positions or patch ids)."""
    rng = np.random.default_rng(seed)
    base = list(items)
    return [list(rng.permutation(base)) for _ in range(repeats)]


def random_baseline(wrapper, ids, mask, pix, mode: str, items: Sequence[int],
                    repeats: int = 5, seed: int = 0) -> Dict[str, float]:
    scores = []
    for index, ranking in enumerate(random_ranking(items, repeats=repeats, seed=seed)):
        if mode == "text":
            scores.append(text_aopc(wrapper, ids, mask, pix, ranking))
        else:
            scores.append(image_aopc(wrapper, ids, mask, pix, ranking))
    return {
        key: float(np.mean([score[key] for score in scores]))
        for key in ("aopc_del", "aopc_ins", "comp_20", "suff_20")
        if key in scores[0]
    }
