"""Inference loading extracted from the paper's frozen LeWM runtime."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import argparse
import tokenizers  # Import before transformers in the recorded runtime.
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel
from .jepa import JEPA
from .module import ARPredictor, Embedder, MLP

@dataclass
class ExperimentConfig:
    image_size: int = 224
    latent_dim: int = 192
    sigreg_weight: float = 0.09

def prepare_pixels(pixels: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """Convert raw HWC/CHW uint8 frames to ImageNet-normalized ViT input."""

    if pixels.ndim != 5:
        raise ValueError(f"expected (B,T,C,H,W), received {tuple(pixels.shape)}")
    batch, steps = pixels.shape[:2]
    image = pixels.reshape(batch * steps, *pixels.shape[2:]).float()
    if image.max() > 1.5:
        image = image.div(255.0)
    if image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (image - mean) / std
def vit_tiny_patch14(image_size: int) -> ViTModel:
    """Construct the same from-scratch ViT-Tiny used by the base config.

    ``stable_pretraining.vit_hf`` currently groups an obsolete Transformers
    export with ``ViTModel`` in one optional import.  Instantiating the native
    classes directly retains the exact Tiny/patch-14 architecture without
    relying on that unrelated optional export.
    """

    config = ViTConfig(
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
        image_size=image_size,
        patch_size=14,
    )
    model = ViTModel(config, add_pooling_layer=False, use_mask_token=False)
    model.config.interpolate_pos_encoding = True
    return model
class OfficialLeWMAdapter(torch.nn.Module):
    """Inference-only adapter for the unmodified official JEPA modules."""

    arm = "A0"

    def __init__(self, official: JEPA, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.official = official
        self.cfg = cfg

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        batch, steps = pixels.shape[:2]
        image = prepare_pixels(pixels, self.cfg.image_size)
        output = self.official.encoder(image, interpolate_pos_encoding=True)
        latent = self.official.projector(output.last_hidden_state[:, 0])
        return latent.reshape(batch, steps, -1)

def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size}

def _load_encoder_for_task(args: argparse.Namespace, device: torch.device):
    state = torch.load(args.official_checkpoint, map_location="cpu", weights_only=False)
    if not any(
        key.startswith(("encoder.encoder.layer.", "encoder.layers."))
        for key in state
    ):
        raise ValueError("Expected the released official LeWM state dictionary")
    action_dim = int(state["action_encoder.patch_embed.weight"].shape[1])
    cfg = ExperimentConfig(sigreg_weight=0.09)
    official = JEPA(
        encoder=vit_tiny_patch14(cfg.image_size),
        predictor=ARPredictor(
            num_frames=3,
            input_dim=cfg.latent_dim,
            hidden_dim=cfg.latent_dim,
            output_dim=cfg.latent_dim,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.0,
        ),
        action_encoder=Embedder(input_dim=action_dim, emb_dim=cfg.latent_dim),
        projector=MLP(cfg.latent_dim, 2048, cfg.latent_dim, norm_fn=torch.nn.BatchNorm1d),
        pred_proj=MLP(cfg.latent_dim, 2048, cfg.latent_dim, norm_fn=torch.nn.BatchNorm1d),
    )
    target_keys = official.state_dict().keys()
    target_uses_new_layout = any(key.startswith("encoder.layers.") for key in target_keys)
    source_uses_new_layout = any(key.startswith("encoder.layers.") for key in state)
    if target_uses_new_layout == source_uses_new_layout:
        replacements = ()
    elif target_uses_new_layout:
        replacements = (
            ("encoder.encoder.layer.", "encoder.layers."),
            (".attention.attention.query.", ".attention.q_proj."),
            (".attention.attention.key.", ".attention.k_proj."),
            (".attention.attention.value.", ".attention.v_proj."),
            (".attention.output.dense.", ".attention.o_proj."),
            (".intermediate.dense.", ".mlp.fc1."),
            (".output.dense.", ".mlp.fc2."),
        )
    else:
        replacements = (
            ("encoder.layers.", "encoder.encoder.layer."),
            (".attention.q_proj.", ".attention.attention.query."),
            (".attention.k_proj.", ".attention.attention.key."),
            (".attention.v_proj.", ".attention.attention.value."),
            (".attention.o_proj.", ".attention.output.dense."),
            (".mlp.fc1.", ".intermediate.dense."),
            (".mlp.fc2.", ".output.dense."),
        )
    mapped = {}
    for key, value in state.items():
        mapped_key = key
        for source, target in replacements:
            mapped_key = mapped_key.replace(source, target)
        mapped[mapped_key] = value
    incompatible = official.load_state_dict(mapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys or len(mapped) != 303:
        raise RuntimeError("official HF LeWM checkpoint failed strict key conversion")
    official.to(device).eval().requires_grad_(False)
    identity = {
        **_checkpoint_metadata(args.official_checkpoint),
        "state_dict_keys": len(mapped),
        "action_dim": action_dim,
        "source_format": "stable_pretraining_hf",
        "inference_only": True,
        "fine_tuned": False,
    }
    return OfficialLeWMAdapter(official, cfg), identity

@torch.no_grad()
def _encode_pixels(
    model: torch.nn.Module, pixels: np.ndarray, device: torch.device
) -> torch.Tensor:
    # Renderers may expose a vertically flipped view with negative strides;
    # torch.from_numpy requires a contiguous positive-stride buffer.
    pixels = np.ascontiguousarray(pixels)
    tensor = (
        torch.from_numpy(pixels)
        .permute(0, 3, 1, 2)
        .unsqueeze(1)
        .to(device)
    )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        return model.encode(tensor).float()[:, 0]

def load_model(checkpoint, device):
    return _load_encoder_for_task(SimpleNamespace(official_checkpoint=Path(checkpoint)), device)[0]

def encode_retrieval(model, frames, device):
    return _encode_pixels(model, np.ascontiguousarray(frames), device).cpu().numpy().astype(np.float32)
