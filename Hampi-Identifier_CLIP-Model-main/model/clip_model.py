"""
clip_model.py — Improved CLIP zero-shot model for Hampi
Fixes:
- Uses mean pooling over prompts
- Better prompt handling
- Upgraded model weights
- Fixed temperature scaling
"""

import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel
import time
import os
import pickle

# ------------------------------------------------------------
# Labels MUST match the keys in PROMPTS
# ------------------------------------------------------------
LABELS = [
    "Hampi_Bazaar",
    "Hemakuta_temple_hill_complex",
    "Lotus_Mahal",
    "Elephant_Stables",
    "Royal_Centre",
    "Matanga_Hill",
    "Monolithic_Bull",
]

# ------------------------------------------------------------
# Prompts (Improved descriptions)
# ------------------------------------------------------------
PROMPTS = {
    "Hampi_Bazaar": [
        "a photo of Hampi Bazaar street with old stone pavilions",
        "long ancient street lined with stone pillars at Hampi",
        "a colonnaded street ancient marketplace ruins in India"
    ],
    "Hemakuta_temple_hill_complex": [
        "a photo of Hemakuta hill temples in Hampi, stepped architecture",
        "ancient stone shrines and ruins on a rocky hill in India",
        "Hemakuta hill monuments complex at sunset"
    ],
    "Lotus_Mahal": [
        "a photo of the Lotus Mahal in Hampi, a symmetric arched palace building",
        "Indo-Islamic style pavilion with scalloped arches",
        "Lotus Mahal ancient structure built with lime mortar and brick"
    ],
    "Elephant_Stables": [
        "a photo of the Elephant Stables in Hampi, a long building with large domes",
        "multiple domed chambers and arches for royal elephants",
        "Indo-Islamic architecture row of domes in India"
    ],
    "Royal_Centre": [
        "a photo of the Royal Centre ruins in Hampi, stone foundations and walls",
        "great platform Mahanavami Dibba in Hampi royal enclosure",
        "ancient royal palace ruins and stepped tanks in India"
    ],
    "Matanga_Hill": [
        "a photo of the rocky landscape of Matanga Hill in Hampi",
        "panoramic view from Matanga Hill showing ruins and boulders",
        "ancient temple on top of a rocky mountain in Hampi"
    ],
    "Monolithic_Bull": [
        "a photo of the giant Monolithic Nandi Bull statue in Hampi",
        "large carved stone bull sculpture in a pavilion",
        "massive Nandi bull carved from a single boulder"
    ]
}


class HampiCLIPModel:
    MODEL_ID = "openai/clip-vit-large-patch14"

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.classifier = None # Holds the sklearn LogisticRegression model
        self.loaded = False

    def load(self):
        if self.loaded: return

        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
        self.model = CLIPModel.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()
        
        # Load the Linear Probe if it exists
        classifier_path = os.path.join(os.path.dirname(__file__), 'hampi_classifier.pkl')
        if os.path.exists(classifier_path):
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
            print("✅ Linear Probe Classifier loaded!")
        else:
            print("⚠️ No classifier.pkl found! Falling back to Zero-Shot Text Prompts.")

        self.loaded = True
        print("✅ CLIP loaded on", self.device)

    def _encode_image(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    # ... (Keep existing _compute_logits as backup) ...

    def predict(self, image: Image.Image, top_k=3):
        if not self.loaded: self.load()
        start = time.time()
        image = image.convert("RGB")
        
        # Extract features vector
        img_feat_tensor = self._encode_image(image)
        
        # If the tuned Probe exists, use it!
        if self.classifier is not None:
            # Flatten to 1D array for sklearn
            feat_array = img_feat_tensor.cpu().numpy().reshape(1, -1)
            
            # Predict probabilities
            probs = self.classifier.predict_proba(feat_array)[0]
            classes = self.classifier.classes_
            
            # Sort top K
            indices = np.argsort(probs)[::-1][:top_k]
            
            results = []
            for rank, idx in enumerate(indices):
                results.append({
                    "name": classes[idx],
                    "confidence": float(probs[idx]),
                    "rank": rank + 1
                })
        else:
            # Fallback to standard Zero-Shot Text Prompts (your existing code)
            logits = self._compute_logits(img_feat_tensor)
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            indices = np.argsort(probs)[::-1][:top_k]
            results = []
            for rank, idx in enumerate(indices):
                results.append({
                    "name": LABELS[idx],
                    "confidence": float(probs[idx]),
                    "rank": rank + 1
                })

        latency = (time.time() - start) * 1000
        return results, latency

# Singleton
_model = None

def get_model():
    global _model
    if _model is None:
        _model = HampiCLIPModel()
    return _model