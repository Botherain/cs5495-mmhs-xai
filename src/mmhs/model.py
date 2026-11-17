"""Loader for the audited MMHS150K checkpoint.

transformers 5.x cannot load this repo directly: the checkpoint stores v4-style
tower prefixes and the custom module calls ``from_pretrained`` inside ``__init__``,
which trips the meta-device guard. We therefore build the class by hand and
remap the keys.
"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Optional

import torch
from safetensors.torch import load_file

RELIGION_IDX = 3
USABLE_HEADS = [0, 1, 2, 4]  # racist, sexist, homophobe, otherhate
TASK_NAMES = ["racist", "sexist", "homophobe", "religion", "otherhate"]


def _load_module(model_dir: str):
    path = os.path.join(model_dir, "modeling_clip_mtl.py")
    spec = importlib.util.spec_from_file_location("modeling_clip_mtl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(model_dir: str, device: str = "cuda"):
    """Return (model, tokenizer, image_processor) ready for inference."""
    from transformers import AutoImageProcessor, AutoTokenizer

    module = _load_module(model_dir)
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as handle:
        config = module.MultiTaskClassifierConfig(**json.load(handle))

    model = module.MultiTaskClassifier(config)
    state = load_file(os.path.join(model_dir, "model.safetensors"))
    state = {
        key.replace("tower_txt.text_model.", "tower_txt.")
        .replace("tower_img.vision_model.", "tower_img."): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, tokenizer, processor


def target_probability(logits: torch.Tensor) -> torch.Tensor:
    """Scalar target per sample: mean probability over the four usable heads."""
    probs = torch.sigmoid(logits)
    return probs[..., USABLE_HEADS].mean(dim=-1)


class Wrapper:
    """Batched inference with an optional missing-modality flag."""

    def __init__(self, model, tokenizer, processor, device: str = "cuda", max_len: int = 77):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.max_len = max_len

    def encode_text(self, texts):
        return self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

    def encode_image(self, images):
        return self.processor(images=list(images), return_tensors="pt")

    @torch.no_grad()
    def target(self, input_ids, attention_mask, pixel_values,
               text_present: Optional[torch.Tensor] = None,
               image_present: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            pixel_values=pixel_values.to(self.device),
            text_present=text_present,
            image_present=image_present,
        )["logits"]
        return target_probability(logits).cpu()

    @torch.no_grad()
    def head_probs(self, input_ids, attention_mask, pixel_values) -> torch.Tensor:
        logits = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            pixel_values=pixel_values.to(self.device),
        )["logits"]
        return torch.sigmoid(logits).cpu()

    def blank_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Mean-colour image: zeros in CLIP-normalised space."""
        return torch.zeros_like(pixel_values)

    @torch.no_grad()
    def gate_stats(self, input_ids, attention_mask, pixel_values):
        """Contribution norms of the two fusion branches: ||g*z_t|| and ||(1-g)*z_i||."""
        captured = {}
        handles = []

        def make_hook(name):
            def hook(module, inputs, output):
                captured[name] = output.detach()
            return hook

        handles.append(self.model.g_t.register_forward_hook(make_hook("t")))
        handles.append(self.model.g_i.register_forward_hook(make_hook("i")))
        handles.append(self.model.gate.register_forward_hook(make_hook("g")))
        try:
            self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                pixel_values=pixel_values.to(self.device),
            )
        finally:
            for handle in handles:
                handle.remove()

        z_t = torch.tanh(captured["t"])
        z_i = torch.tanh(captured["i"])
        gate = torch.sigmoid(captured["g"])
        text_norm = (gate * z_t).norm(dim=-1).cpu()
        image_norm = ((1.0 - gate) * z_i).norm(dim=-1).cpu()
        return text_norm, image_norm

    def blank_text(self, batch_size: int):
        encoded = self.encode_text([""] * batch_size)
        return encoded["input_ids"], encoded["attention_mask"]
