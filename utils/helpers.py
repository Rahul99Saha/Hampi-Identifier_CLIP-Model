"""
helpers.py — Utility functions for metadata, formatting, and UI helpers.
"""

import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE.parent / "data"
_METADATA_PATH = _DATA_DIR / "metadata.json"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

_metadata_cache: dict | None = None


def load_metadata() -> dict:
    """Load monument metadata from JSON, with in-memory caching."""
    global _metadata_cache
    if _metadata_cache is None:
        with open(_METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata_cache = json.load(f)
    return _metadata_cache


def get_monument_info(name: str) -> dict | None:
    """
    Return metadata dict for a monument name.
    Falls back to fuzzy matching if exact name not found.
    """
    metadata = load_metadata()
    if name in metadata:
        return metadata[name]

    # Case-insensitive fallback
    name_lower = name.lower()
    for key, val in metadata.items():
        if key.lower() == name_lower:
            return val
        # Check aliases
        for alias in val.get("aliases", []):
            if alias.lower() == name_lower:
                return val

    return None


def get_all_monument_names() -> list[str]:
    """Return sorted list of all monument names in metadata."""
    return sorted(load_metadata().keys())


# ---------------------------------------------------------------------------
# Confidence formatting
# ---------------------------------------------------------------------------

def confidence_color(confidence: float) -> str:
    """Return a hex color string based on confidence level."""
    if confidence >= 0.70:
        return "#2ecc71"   # green
    elif confidence >= 0.45:
        return "#f39c12"   # amber
    else:
        return "#e74c3c"   # red


def confidence_label(confidence: float) -> str:
    """Return a human-readable confidence label."""
    if confidence >= 0.70:
        return "High confidence"
    elif confidence >= 0.45:
        return "Moderate confidence"
    elif confidence >= 0.25:
        return "Low confidence"
    else:
        return "Very low confidence — try a clearer image"


def confidence_emoji(confidence: float) -> str:
    if confidence >= 0.70:
        return "✅"
    elif confidence >= 0.45:
        return "🟡"
    else:
        return "🔴"


# ---------------------------------------------------------------------------
# Maps & external links
# ---------------------------------------------------------------------------

def make_maps_url(monument_info: dict) -> str:
    """Build Google Maps URL from stored maps_url or coordinates."""
    if "maps_url" in monument_info:
        return monument_info["maps_url"]
    coords = monument_info.get("coordinates", {})
    if "lat" in coords and "lng" in coords:
        return f"https://maps.google.com/?q={coords['lat']},{coords['lng']}"
    return f"https://maps.google.com/?q={monument_info['name'].replace(' ', '+')}+Hampi"


def make_wikipedia_url(name: str) -> str:
    """Build Wikipedia URL for a monument name."""
    slug = name.replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{slug}"


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def truncate(text: str, max_chars: int = 350) -> str:
    """Truncate text to max_chars with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def format_tags(tags: list[str]) -> str:
    """Format tag list as pill-style string."""
    return "  ".join(f"`{t}`" for t in tags)


def tier_badge(tier: int) -> str:
    return "🥇 Tier 1" if tier == 1 else "🥈 Tier 2"
