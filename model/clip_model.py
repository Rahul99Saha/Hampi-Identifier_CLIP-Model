"""
clip_model.py — CLIP zero-shot monument identifier for Hampi (T12.5)

Supports three prediction modes:
  1. Zero-Shot     — Pure CLIP cosine-similarity with prompt ensembling
  2. Linear Probe  — Logistic Regression trained on frozen CLIP features (light fine-tuning)
  3. Hybrid        — Weighted blend of Zero-Shot + Linear Probe scores

Run with:
    from model.clip_model import get_model
    model = get_model()
    predictions, latency = model.predict(image, mode="hybrid", ensemble_weight=0.7)
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Monument class names and rich CLIP prompt templates
# ---------------------------------------------------------------------------

MONUMENT_NAMES = [
    "Lotus Mahal",
    "Virupaksha Temple",
    "Vittala Temple",
    "Elephant Stables",
    "Hampi Bazaar",
    "Zenana Enclosure",
    "Royal Centre",
    "Queen's Bath",
    "Hemakuta temple hill complex",
    "Monolithic Bull",
]

# CLIP performs best with descriptive, context-rich prompts rather than bare names.
# Multiple prompts per monument improve recall via prompt ensembling.
MONUMENT_PROMPTS = {
    "Lotus Mahal": [
        "a photo of Lotus Mahal pavilion in Hampi with its arched Indo-Islamic architecture",
        "a clear daylight photo of Lotus Mahal two-storey ornate pavilion in Hampi",
        "a tourist photo of Lotus Mahal in the Zenana Enclosure of Hampi Karnataka",
        "an architectural photo of Kamal Mahal with lotus bud arches in Hampi",
        "a heritage monument photo of Lotus Mahal Vijayanagara style palace Hampi",
    ],
    "Virupaksha Temple": [
        "a photo of Virupaksha Temple in Hampi with its tall gopuram tower",
        "a clear daylight photo of the ancient Virupaksha Pampapati temple Karnataka",
        "a tourist photo of Virupaksha Temple at Hampi Bazaar in Hampi",
        "an architectural photo of Virupaksha Temple towering gateway Hampi India",
        "a heritage monument photo of Virupaksha Temple 7th-century Shiva temple Hampi",
    ],
    "Vittala Temple": [
        "a photo of the stone chariot of Vittala Temple in Hampi",
        "a clear daylight photo of the famous stone chariot and musical pillars Vittala Temple",
        "a tourist photo of Vijaya Vittala Temple complex in Hampi",
        "an architectural photo of ornate stone chariot shrine at Vittala Temple Hampi",
        "a heritage monument photo of Vittala Temple with musical pillars Hampi Karnataka",
    ],
    "Elephant Stables": [
        "a photo of the Elephant Stables with domed chambers in Hampi",
        "a clear daylight photo of a row of domed elephant stable chambers Vijayanagara",
        "a tourist photo of Elephant Stables in Hampi",
        "an architectural photo of Gajashala elephant stables with varied domes Hampi",
        "a heritage monument photo of Elephant Stables eleven domed chambers Hampi Karnataka",
    ],
    "Hampi Bazaar": [
        "a photo of the long colonnaded Hampi Bazaar street",
        "a clear daylight photo of Hampi Bazaar leading to Virupaksha Temple",
        "a tourist photo of Hampi Bazaar ancient marketplace in Hampi",
        "an architectural photo of stone pillared mandapas along Hampi Bazaar",
        "a heritage monument photo of Hampi Bazaar market street with ruins Karnataka",
    ],
    "Zenana Enclosure": [
        "a photo of the Zenana Enclosure fortified area in Hampi",
        "a clear daylight photo of Zenana Enclosure reserved for royal women Hampi",
        "a tourist photo of Zenana Enclosure with watch towers in Hampi",
        "an architectural photo of Zenana Enclosure fortified walls and pavilions Hampi",
        "a heritage monument photo of Zenana Enclosure Vijayanagara royal quarters Hampi Karnataka",
    ],
    "Royal Centre": [
        "a photo of the Royal Centre area in Hampi",
        "a clear daylight photo of Royal Centre with courtly structures Hampi",
        "a tourist photo of Royal Centre military and administrative buildings in Hampi",
        "an architectural photo of Royal Centre stepped tank and throne platform Hampi",
        "a heritage monument photo of Royal Centre Vijayanagara king's court Hampi Karnataka",
    ],
    "Queen's Bath": [
        "a photo of Queen's Bath royal pool pavilion in Hampi with arched balconies",
        "a clear daylight photo of the ornate bathing enclosure Queen's Bath Hampi",
        "a tourist photo of Queen's Bath in Hampi",
        "an architectural photo of rectangular pool inside Queen's Bath pavilion Hampi",
        "a heritage monument photo of Queen's Bath Vijayanagara royal bathing complex Karnataka",
    ],
    "Hemakuta temple hill complex": [
        "a photo of Hemakuta temple hill complex with ancient temples in Hampi",
        "a clear daylight photo of Hemakuta Hill temples overlooking Hampi",
        "a tourist photo of Hemakuta temple hill complex in Hampi",
        "an architectural photo of Jain and Shaiva temples on Hemakuta Hill Hampi",
        "a heritage monument photo of Hemakuta temple hill complex sunset view Hampi Karnataka",
    ],
    "Monolithic Bull": [
        "a photo of the large Monolithic Bull Nandi statue in Hampi",
        "a clear daylight photo of Monolithic Bull stone Nandi sculpture Hampi",
        "a tourist photo of Monolithic Bull in Hampi",
        "an architectural photo of the carved monolithic Nandi bull in Hampi",
        "a heritage monument photo of Monolithic Bull Yeduru Basavanna Hampi Karnataka",
    ],
}


# Prediction modes
MODE_ZERO_SHOT   = "zero_shot"
MODE_LINEAR_PROBE = "linear_probe"
MODE_HYBRID      = "hybrid"
ALL_MODES = [MODE_ZERO_SHOT, MODE_LINEAR_PROBE, MODE_HYBRID]


def load_prompts_from_file(prompts_path: str) -> dict | None:
    """Load text prompts from a JSON file.  Returns None on failure."""
    try:
        with open(prompts_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Folder name ↔ class name mapping
# ---------------------------------------------------------------------------

FOLDER_TO_CLASS = {
    "Lotus_Mahal": "Lotus Mahal",
    "Virupaksha_Temple": "Virupaksha Temple",
    "Vittala_Temple": "Vittala Temple",
    "Elephant_Stables": "Elephant Stables",
    "Hampi_Bazaar": "Hampi Bazaar",
    "Zenana_Enclosure": "Zenana Enclosure",
    "Royal_Centre": "Royal Centre",
    "Queen_s_Bath": "Queen's Bath",
    "Hemakuta_temple_hill_complex": "Hemakuta temple hill complex",
    "Monolithic_Bull": "Monolithic Bull",
}

CLASS_TO_FOLDER = {v: k for k, v in FOLDER_TO_CLASS.items()}


def folder_name_to_class(folder_name: str) -> str | None:
    """Convert a dataset folder name to its CLIP class label."""
    return FOLDER_TO_CLASS.get(folder_name, None)


def class_to_folder_name(class_name: str) -> str | None:
    """Convert a CLIP class label to its dataset folder name."""
    return CLASS_TO_FOLDER.get(class_name, None)


# Path to the dataset directory (relative to this file)
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_MODEL_DIR = os.path.dirname(__file__)
_PROBE_PATH = os.path.join(_MODEL_DIR, "hampi_classifier.pkl")


# ---------------------------------------------------------------------------
# HampiCLIPModel
# ---------------------------------------------------------------------------

class HampiCLIPModel:
    """
    Unified Hampi monument classifier supporting three modes:

    ┌──────────────────┬────────────────────────────────────────────────────┐
    │ Mode             │ Description                                        │
    ├──────────────────┼────────────────────────────────────────────────────┤
    │ zero_shot        │ Pure CLIP cosine-similarity + prompt ensembling    │
    │ linear_probe     │ Logistic Regression on frozen CLIP features        │
    │ hybrid           │ Weighted blend: w*probe + (1-w)*zero_shot          │
    └──────────────────┴────────────────────────────────────────────────────┘

    Model variants:
    - "openai/clip-vit-base-patch32": Base model, good speed (default)
    - "openai/clip-vit-large-patch14": Larger model (untested on this dataset)
    """

    MODEL_VARIANTS = {
        "base": "openai/clip-vit-base-patch32",
        "large": "openai/clip-vit-large-patch14",
    }

    def __init__(self, model_variant: str = "base", device: str | None = None):
        if model_variant not in self.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model variant: {model_variant}. "
                f"Choose from: {list(self.MODEL_VARIANTS.keys())}"
            )
        self.model_variant = model_variant
        self.MODEL_ID = self.MODEL_VARIANTS[model_variant]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._text_features_cache: dict | None = None
        self._loaded = False
        self._probe = None          # LinearProbeClassifier, loaded on demand

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self):
        """Download / load CLIP model weights (HuggingFace cache)."""
        if self._loaded:
            return
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
        self.model = CLIPModel.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()
        self._precompute_text_features()
        self._loaded = True

    def load_with_prompts(self, prompts_file: str | None = None):
        """Load CLIP and use a specific prompts JSON file."""
        if self._loaded:
            return
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
        self.model = CLIPModel.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()
        self._precompute_text_features(prompts_file)
        self._loaded = True

    def _precompute_text_features(self, custom_prompts_path: str | None = None):
        """
        Encode all monument text prompts once and cache them.
        Priority: custom_prompts_path → data/prompts.json → hardcoded MONUMENT_PROMPTS
        """
        prompts_path = custom_prompts_path or os.path.join(_DATA_DIR, "prompts.json")
        file_prompts = load_prompts_from_file(prompts_path) if prompts_path else None
        active_prompts = file_prompts if file_prompts is not None else MONUMENT_PROMPTS

        all_features = {}
        with torch.no_grad():
            for name, prompts in active_prompts.items():
                inputs = self.processor(
                    text=prompts, return_tensors="pt", padding=True
                ).to(self.device)
                text_features = self.model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                pooled_output = text_features.pooler_output
                text_embeds = self.model.text_projection(pooled_output)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                feats = text_embeds.mean(dim=0)
                feats = feats / feats.norm()
                all_features[name] = feats

        self._text_features_cache = {
            "matrix": torch.stack(list(all_features.values())),
            "labels": list(all_features.keys()),
        }

    # ------------------------------------------------------------------
    # Linear Probe management
    # ------------------------------------------------------------------

    def load_probe(self, probe_path: str | None = None) -> bool:
        """
        Load the linear probe from disk.  Returns True if successful.
        """
        from model.linear_probe import LinearProbeClassifier
        path = probe_path or _PROBE_PATH
        if LinearProbeClassifier.exists(path):
            try:
                self._probe = LinearProbeClassifier.load(path)
                return True
            except Exception:
                self._probe = None
        return False

    def has_probe(self) -> bool:
        """Return True if a trained linear probe is loaded."""
        return self._probe is not None

    def train_and_save_probe(self, verbose: bool = True) -> "LinearProbeClassifier":
        """
        Train a Linear Probe on the train_images/ data and cache it.
        The CLIP model must already be loaded.
        """
        if not self._loaded:
            self.load()
        from model.linear_probe import train_linear_probe
        self._probe = train_linear_probe(self, verbose=verbose)
        return self._probe

    # ------------------------------------------------------------------
    # Feature extraction (shared between modes)
    # ------------------------------------------------------------------

    def _get_image_features(self, image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
        """
        Encode a PIL image with CLIP.
        Returns (img_tensor (1,D), img_numpy (1,D)).
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_out.pooler_output
            img_feats = self.model.visual_projection(pooled)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        return img_feats, img_feats.cpu().numpy()

    # ------------------------------------------------------------------
    # Zero-shot scoring
    # ------------------------------------------------------------------

    def _zero_shot_scores(self, img_feats: torch.Tensor) -> np.ndarray:
        """Return softmax probabilities (array of shape (n_classes,)) via CLIP."""
        text_matrix = self._text_features_cache["matrix"].to(self.device)
        logit_scale = self.model.logit_scale.exp()
        logits = (logit_scale * img_feats @ text_matrix.T).squeeze(0)
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        return probs

    # ------------------------------------------------------------------
    # Main predict()
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Image.Image,
        top_k: int = 3,
        mode: str = MODE_ZERO_SHOT,
        ensemble_weight: float = 0.7,
    ) -> tuple[list[dict], float]:
        """
        Classify a PIL image and return top-k predictions.

        Args:
            image           : PIL.Image.Image (RGB, any size)
            top_k           : Number of top predictions to return
            mode            : "zero_shot" | "linear_probe" | "hybrid"
            ensemble_weight : (only for "hybrid") weight for probe score.
                              Final score = w * probe + (1-w) * zero_shot
                              Range: 0.0 (pure zero-shot) → 1.0 (pure probe)

        Returns:
            (results, latency_ms)
            results: list of dicts:
              { name, confidence, confidence_pct, rank, source }
        """
        if not self._loaded:
            self.load()

        if mode not in ALL_MODES:
            raise ValueError(f"mode must be one of {ALL_MODES}, got '{mode}'")

        t0 = time.time()

        # ── Step 1: Extract image features ───────────────────────────
        img_feats_tensor, img_feats_np = self._get_image_features(image)

        labels = self._text_features_cache["labels"]
        n_classes = len(labels)
        top_k = min(top_k, n_classes)

        # ── Step 2: Compute scores based on mode ──────────────────────
        if mode == MODE_ZERO_SHOT:
            probs = self._zero_shot_scores(img_feats_tensor)
            source = "Zero-Shot CLIP"

        elif mode == MODE_LINEAR_PROBE:
            if self._probe is None:
                # Fall back gracefully to zero-shot if probe unavailable
                probs = self._zero_shot_scores(img_feats_tensor)
                source = "Zero-Shot CLIP (probe not available)"
            else:
                # Get probe probs in the same label order as CLIP text cache
                probe_results, _ = self._probe.predict_from_features(img_feats_np, top_k=n_classes)
                probe_map = {r["name"]: r["confidence"] for r in probe_results}
                probs = np.array([probe_map.get(lbl, 0.0) for lbl in labels])
                # Re-normalise so they sum to 1 (they should, but floating point)
                total = probs.sum()
                if total > 0:
                    probs = probs / total
                source = "Linear Probe (light fine-tuning)"

        elif mode == MODE_HYBRID:
            zs_probs = self._zero_shot_scores(img_feats_tensor)

            if self._probe is None:
                probs = zs_probs
                source = "Zero-Shot CLIP (probe not available)"
            else:
                probe_results, _ = self._probe.predict_from_features(img_feats_np, top_k=n_classes)
                probe_map = {r["name"]: r["confidence"] for r in probe_results}
                probe_probs = np.array([probe_map.get(lbl, 0.0) for lbl in labels])

                # Normalise probe probs for the classes the probe was trained on
                total = probe_probs.sum()
                if total > 0:
                    probe_probs = probe_probs / total

                w = float(np.clip(ensemble_weight, 0.0, 1.0))
                probs = w * probe_probs + (1.0 - w) * zs_probs
                # Final re-normalise
                total = probs.sum()
                if total > 0:
                    probs = probs / total
                source = f"Hybrid (probe {w:.0%} + zero-shot {1-w:.0%})"

        # ── Step 3: Build top-k results ───────────────────────────────
        top_indices = np.argsort(probs)[::-1][:top_k]
        results = [
            {
                "name": labels[i],
                "confidence": float(probs[i]),
                "confidence_pct": f"{probs[i]*100:.1f}%",
                "rank": rank + 1,
                "source": source,
            }
            for rank, i in enumerate(top_indices)
        ]

        latency = (time.time() - t0) * 1000
        return results, latency

    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-loaded)
# ---------------------------------------------------------------------------

_model_instance: HampiCLIPModel | None = None


def get_model(use_enhanced_prompts: bool = True) -> HampiCLIPModel:
    """
    Return the singleton HampiCLIPModel instance (auto-loads on first call).

    Args:
        use_enhanced_prompts: If True, load data/prompts.json (default).
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = HampiCLIPModel()
        enhanced_prompts_path = (
            os.path.join(_DATA_DIR, "prompts.json") if use_enhanced_prompts else None
        )
        _model_instance.load_with_prompts(enhanced_prompts_path)
    return _model_instance


def reset_model_cache():
    """Force recreation of the model singleton (useful for testing)."""
    global _model_instance
    _model_instance = None
