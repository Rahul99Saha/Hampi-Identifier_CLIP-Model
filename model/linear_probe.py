"""
linear_probe.py — Linear Probe classifier trained on frozen CLIP features (T12.5)

Pipeline:
  1. Extract CLIP image embeddings from train_images/ (frozen — no CLIP weights updated)
  2. Fit a scikit-learn LogisticRegression (L-BFGS) on those embeddings
  3. Save the trained model to model/hampi_classifier.pkl
  4. At inference time load the probe and return top-k predictions with probabilities

NOTE: Only 6 of the 10 classes have training images.  For the remaining 4 classes the
      probe falls back to zero-shot CLIP scores so every class can still be returned.
"""

from __future__ import annotations

import os
import pickle
import time
import numpy as np
from PIL import Image
from pathlib import Path

# ── Lazy imports so Streamlit startup is fast ─────────────────────────────────
def _import_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.exceptions import NotFittedError
        return LogisticRegression, LabelEncoder, NotFittedError
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for the Linear Probe.  "
            "Install it with:  pip install scikit-learn"
        ) from e


# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROBE_PATH = _HERE / "hampi_classifier.pkl"
_DATA_DIR = _HERE.parent / "data"
_TRAIN_DIR = _DATA_DIR / "train_images"


# ── Folder → class name mapping (must match clip_model.py) ────────────────────
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

ALL_CLASS_NAMES = [
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


def _extract_clip_features(
    image_paths: list[Path],
    clip_model,
    clip_processor,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Extract normalised CLIP image embeddings for a list of image paths.

    Returns ndarray of shape (N, embed_dim).
    """
    import torch

    all_features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception:
                # Skip corrupt images
                continue

        if not images:
            continue

        inputs = clip_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_out.pooler_output
            feats = clip_model.visual_projection(pooled)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_features.append(feats.cpu().numpy())

    return np.vstack(all_features) if all_features else np.array([])


def train_linear_probe(
    clip_model_obj,
    save_path: str | Path | None = None,
    verbose: bool = True,
) -> "LinearProbeClassifier":
    """
    Train a Logistic Regression probe on top of frozen CLIP features.

    Args:
        clip_model_obj : A loaded HampiCLIPModel instance (from clip_model.py)
        save_path      : Where to save the .pkl.  Defaults to model/hampi_classifier.pkl
        verbose        : Print progress messages

    Returns:
        Trained LinearProbeClassifier
    """
    LogisticRegression, LabelEncoder, _ = _import_sklearn()

    if not _TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training directory not found: {_TRAIN_DIR}")

    # --- Collect image paths & labels ----------------------------------------
    image_paths: list[Path] = []
    labels: list[str] = []

    train_classes_found = []
    for folder in sorted(_TRAIN_DIR.iterdir()):
        if not folder.is_dir():
            continue
        class_name = FOLDER_TO_CLASS.get(folder.name)
        if class_name is None:
            if verbose:
                print(f"  [WARN] Unknown folder '{folder.name}' — skipping")
            continue
        files = [
            p for p in folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
        if not files:
            continue
        train_classes_found.append(class_name)
        for p in files:
            image_paths.append(p)
            labels.append(class_name)

    if verbose:
        print(f"[LinearProbe] Training classes ({len(train_classes_found)}): {train_classes_found}")
        print(f"[LinearProbe] Total training images: {len(image_paths)}")

    # --- Extract CLIP features ------------------------------------------------
    if verbose:
        print("[LinearProbe] Extracting CLIP features…")

    features = _extract_clip_features(
        image_paths,
        clip_model_obj.model,
        clip_model_obj.processor,
        clip_model_obj.device,
    )

    if len(features) == 0:
        raise RuntimeError("No features extracted — check that train_images/ contains valid images.")

    # --- Encode labels --------------------------------------------------------
    le = LabelEncoder()
    le.fit(train_classes_found)           # only trained classes
    y = le.transform(labels)

    # --- Fit Logistic Regression ----------------------------------------------
    if verbose:
        print("[LinearProbe] Fitting Logistic Regression…")

    clf = LogisticRegression(
        C=4.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        verbose=0,
    )
    clf.fit(features, y)

    train_acc = clf.score(features, y)
    if verbose:
        print(f"[LinearProbe] Training accuracy (on train set): {train_acc*100:.1f}%")

    # --- Bundle and save ------------------------------------------------------
    probe = LinearProbeClassifier(
        clf=clf,
        label_encoder=le,
        trained_classes=train_classes_found,
        all_classes=ALL_CLASS_NAMES,
        embed_dim=features.shape[1],
    )

    save_path = Path(save_path) if save_path else _PROBE_PATH
    probe.save(save_path)
    if verbose:
        print(f"[LinearProbe] Saved probe to {save_path}")

    return probe


class LinearProbeClassifier:
    """
    A trained Linear Probe that wraps a scikit-learn LogisticRegression.

    Exposes predict() with the same signature as HampiCLIPModel.predict().
    For classes not in the training set the probe returns probability 0 and
    the caller should blend in zero-shot scores.
    """

    def __init__(
        self,
        clf,
        label_encoder,
        trained_classes: list[str],
        all_classes: list[str],
        embed_dim: int,
    ):
        self.clf = clf
        self.label_encoder = label_encoder
        self.trained_classes = trained_classes      # classes the probe knows
        self.all_classes = all_classes              # all 10 monument classes
        self.embed_dim = embed_dim

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path = _PROBE_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path = _PROBE_PATH) -> "LinearProbeClassifier":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Probe file not found: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, LinearProbeClassifier):
            raise TypeError(f"Expected LinearProbeClassifier, got {type(obj)}")
        return obj

    @staticmethod
    def exists(path: str | Path = _PROBE_PATH) -> bool:
        return Path(path).exists()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_from_features(
        self,
        img_features: "np.ndarray",
        top_k: int = 3,
    ) -> tuple[list[dict], float]:
        """
        Given a normalised CLIP image embedding (1, D) numpy array, return
        top-k predictions.

        Returns (results, latency_ms)
        """
        t0 = time.time()

        # Probabilities for trained classes only
        proba_trained = self.clf.predict_proba(img_features.reshape(1, -1))[0]
        trained_class_names = list(self.label_encoder.classes_)

        # Map to full 10-class space (unseen classes → 0)
        full_proba = np.zeros(len(self.all_classes), dtype=float)
        for cls_name, p in zip(trained_class_names, proba_trained):
            idx = self.all_classes.index(cls_name)
            full_proba[idx] = p

        latency = (time.time() - t0) * 1000

        top_k = min(top_k, len(self.all_classes))
        top_indices = np.argsort(full_proba)[::-1][:top_k]
        results = [
            {
                "name": self.all_classes[i],
                "confidence": float(full_proba[i]),
                "confidence_pct": f"{full_proba[i]*100:.1f}%",
                "rank": rank + 1,
                "in_training": self.all_classes[i] in self.trained_classes,
            }
            for rank, i in enumerate(top_indices)
        ]
        return results, latency

    def predict(
        self,
        image: Image.Image,
        clip_model,
        clip_processor,
        device: str,
        top_k: int = 3,
    ) -> tuple[list[dict], float]:
        """
        Full inference pipeline: PIL image → top-k predictions.
        Extracts CLIP features internally.
        """
        import torch

        t0 = time.time()
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_out.pooler_output
            feats = clip_model.visual_projection(pooled)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        img_np = feats.cpu().numpy()

        results, latency = self.predict_from_features(img_np, top_k=top_k)
        latency = (time.time() - t0) * 1000
        return results, latency


# ── Module-level singleton ────────────────────────────────────────────────────

_probe_instance: LinearProbeClassifier | None = None


def get_probe(path: str | Path = _PROBE_PATH) -> LinearProbeClassifier | None:
    """
    Return the cached LinearProbeClassifier, or None if no .pkl exists yet.
    Call train_linear_probe() first to create the .pkl.
    """
    global _probe_instance
    if _probe_instance is None:
        if LinearProbeClassifier.exists(path):
            _probe_instance = LinearProbeClassifier.load(path)
    return _probe_instance


def reset_probe_cache():
    """Force reload of the probe singleton (e.g. after retraining)."""
    global _probe_instance
    _probe_instance = None
