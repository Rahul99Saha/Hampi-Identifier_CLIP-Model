"""
clip_model.py — CLIP zero-shot monument identifier for Hampi (T12.5)
Uses openai/clip-vit-base-patch32 with rich text prompts for best accuracy.
"""

import json
import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import time

# ---------------------------------------------------------------------------
# Monument class names and rich CLIP prompt templates
# Updated to match the new Hampi Zero-Shot CLIP Dataset (10 classes, 120 images)
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


def load_prompts_from_file(prompts_path: str) -> dict | None:
    """
    Load text prompts from a JSON file (e.g., data/prompts.json).
    Returns None if the file does not exist or cannot be parsed.
    """
    try:
        with open(prompts_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Folder name ↔ class name mapping
# The dataset uses folder names with underscores (e.g. "Elephant_Stables",
# "Queen_s_Bath") but the CLIP class labels use proper names (e.g.
# "Elephant Stables", "Queen's Bath"). This mapping bridges the two.
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


class HampiCLIPModel:
    """
    Zero-shot monument classifier using CLIP.
    Supports prompt ensembling and returns top-k predictions with confidence.
    """

    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._text_features_cache: dict | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self):
        """Download / load model weights (cached by HuggingFace hub)."""
        if self._loaded:
            return
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
        self.model = CLIPModel.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()
        self._precompute_text_features()
        self._loaded = True

    def _precompute_text_features(self):
        """
        Encode all monument text prompts once and cache them.
        Tries to load prompts from data/prompts.json first; falls back to
        the hardcoded MONUMENT_PROMPTS dict.
        During inference only image encoding is needed — speeds up prediction.
        """
        # Try loading prompts from the dataset file
        prompts_path = os.path.join(_DATA_DIR, "prompts.json")
        file_prompts = load_prompts_from_file(prompts_path)
        active_prompts = file_prompts if file_prompts is not None else MONUMENT_PROMPTS

        all_features = {}
        with torch.no_grad():
            for name, prompts in active_prompts.items():
                inputs = self.processor(
                    text=prompts, return_tensors="pt", padding=True
                ).to(self.device)
                feats = self.model.get_text_features(**inputs)  # (n_prompts, 512)
                # Ensemble by mean-pooling across prompts
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats = feats.mean(dim=0)  # (512,)
                feats = feats / feats.norm()
                all_features[name] = feats
        # Stack into matrix (n_classes, 512)
        self._text_features_cache = {
            "matrix": torch.stack(list(all_features.values())),  # (10, 512)
            "labels": list(all_features.keys()),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        """
        Classify a PIL image and return top-k predictions.

        Returns:
            List of dicts: [{"name": str, "confidence": float}, ...]
            Sorted by confidence descending.
        """
        if not self._loaded:
            self.load()

        t0 = time.time()

        # Encode image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_features = self.model.get_image_features(**inputs)  # (1, 512)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        # Cosine similarity with text matrix
        text_matrix = self._text_features_cache["matrix"].to(self.device)  # (10, 512)
        labels = self._text_features_cache["labels"]

        # logit_scale learned by CLIP
        logit_scale = self.model.logit_scale.exp()
        logits = (logit_scale * img_features @ text_matrix.T).squeeze(0)  # (10,)

        # Softmax probabilities
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()

        latency = (time.time() - t0) * 1000  # ms

        # Build top-k results
        top_k = min(top_k, len(labels))
        top_indices = np.argsort(probs)[::-1][:top_k]
        results = [
            {
                "name": labels[i],
                "confidence": float(probs[i]),
                "confidence_pct": f"{probs[i]*100:.1f}%",
                "rank": rank + 1,
            }
            for rank, i in enumerate(top_indices)
        ]

        return results, latency

    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-loaded)
# ---------------------------------------------------------------------------

_model_instance: HampiCLIPModel | None = None


def get_model() -> HampiCLIPModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = HampiCLIPModel()
    return _model_instance
