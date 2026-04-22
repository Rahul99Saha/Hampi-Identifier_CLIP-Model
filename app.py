"""
app.py — Hampi Monument Identifier  |  T12.5
Streamlit frontend using CLIP zero-shot classification.

Run with:
    streamlit run app.py
"""

import sys
import os

# Ensure sibling packages are importable
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from PIL import Image
import time

from model.clip_model import get_model, MONUMENT_NAMES
from utils.preprocess import load_image_from_upload, prepare_for_clip, validate_image_quality
from utils.helpers import (
    get_monument_info,
    confidence_color,
    confidence_label,
    confidence_emoji,
    make_maps_url,
    make_wikipedia_url,
    truncate,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Hampi Monument Identifier",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---- Hero banner ---- */
    .hero-banner {
        background: linear-gradient(135deg, #1a0a00 0%, #5c2d0e 50%, #b85c2c 100%);
        padding: 2.5rem 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #f5d78e;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        color: #e0c08a;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(245,215,142,0.15);
        border: 1px solid rgba(245,215,142,0.3);
        color: #f5d78e;
        border-radius: 20px;
        padding: 3px 14px;
        font-size: 0.8rem;
        margin-top: 0.8rem;
        letter-spacing: 0.5px;
    }

    /* ---- Monument result card ---- */
    .result-card {
        background: #fff;
        border-radius: 14px;
        padding: 1.6rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08);
        border-left: 5px solid #b85c2c;
        margin-bottom: 1.2rem;
    }
    .monument-name {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1a0a00;
        margin: 0;
    }
    .confidence-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 6px;
    }

    /* ---- Info card ---- */
    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-top: 1rem;
    }
    .info-cell {
        background: #fdf6ec;
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }
    .info-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #7a5230;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 3px;
    }
    .info-value {
        font-size: 0.93rem;
        color: #1a0a00;
        font-weight: 500;
    }

    /* ---- History box ---- */
    .history-box {
        background: #fdf9f4;
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #3d2010;
        border: 1px solid #e8d5bb;
    }

    /* ---- Top-3 bar ---- */
    .top3-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.5rem;
    }
    .top3-label { min-width: 200px; font-size: 0.88rem; font-weight: 500; color: #3d2010; }
    .top3-bar-bg {
        flex: 1; height: 10px; background: #e8d5bb;
        border-radius: 6px; overflow: hidden;
    }
    .top3-bar-fill { height: 100%; border-radius: 6px; }
    .top3-pct { font-size: 0.82rem; color: #7a5230; min-width: 45px; text-align: right; }

    /* ---- Buttons ---- */
    div[data-testid="stButton"] button {
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    /* ---- Upload area ---- */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    /* ---- Sidebar ---- */
    .stSidebar { background: #fdf6ec; }

    /* ---- Footer ---- */
    .footer {
        text-align: center;
        color: #a08060;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e8d5bb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────

if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "image_pil" not in st.session_state:
    st.session_state.image_pil = None
if "latency" not in st.session_state:
    st.session_state.latency = None
if "show_full_history" not in st.session_state:
    st.session_state.show_full_history = False

# ──────────────────────────────────────────────
# Hero banner
# ──────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-banner">
        <div style="font-size:2.8rem;margin-bottom:0.3rem;">🏛️</div>
        <p class="hero-title">Hampi Monument Identifier</p>
        <p class="hero-sub">Upload a photo of any Hampi monument — get instant name, history, and visiting details</p>
        <span class="hero-badge">🤖 Powered by OpenAI CLIP · Zero-Shot · T12.5</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Sidebar — model info & monument list
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown(
        "This app uses **CLIP zero-shot classification** to identify "
        "monuments at Hampi, a UNESCO World Heritage Site in Karnataka, India."
    )
    st.divider()
    st.markdown("### 🏛️ Supported Monuments")
    for m in sorted(MONUMENT_NAMES):
        st.markdown(f"- {m}")
    st.divider()
    st.markdown("### ⚙️ Model")
    st.code("openai/clip-vit-base-patch32", language=None)
    st.markdown("3-prompt ensemble per monument · Softmax probabilities")
    st.divider()
    st.markdown(
        "[📂 Dataset: Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Group_of_monuments_at_Hampi)",
        unsafe_allow_html=False,
    )

# ──────────────────────────────────────────────
# Main layout: upload (left) | results (right)
# ──────────────────────────────────────────────

col_left, col_right = st.columns([1, 1.3], gap="large")

# ── LEFT: Upload & Preview ──────────────────

with col_left:
    st.markdown("#### 📤 Upload Monument Photo")

    uploaded_file = st.file_uploader(
        label="Drop an image or click to browse",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Load and validate
        try:
            image_pil = load_image_from_upload(uploaded_file)
            quality = validate_image_quality(image_pil)

            if quality["warnings"]:
                for w in quality["warnings"]:
                    st.warning(w)

            # Display preview
            st.image(
                image_pil,
                caption=f"📷 {uploaded_file.name}  ({image_pil.width}×{image_pil.height}px)",
                use_container_width=True,
            )

            st.session_state.image_pil = image_pil
            st.session_state.predictions = None  # reset on new upload

        except ValueError as e:
            st.error(f"❌ {e}")
            st.session_state.image_pil = None

    else:
        # Placeholder
        st.markdown(
            """
            <div style="
                border: 2px dashed #d4aa80;
                border-radius: 12px;
                padding: 3rem 1rem;
                text-align: center;
                color: #a08060;
                background: #fdf9f4;
            ">
                <div style="font-size:2.5rem">📸</div>
                <div style="margin-top:0.5rem;font-size:0.95rem">
                    Supports JPG · PNG · WEBP · BMP
                </div>
                <div style="font-size:0.8rem;margin-top:0.3rem;opacity:0.7">
                    Max 10 MB
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Identify button
    st.markdown("<br>", unsafe_allow_html=True)
    identify_clicked = st.button(
        "🔍  Identify Monument",
        type="primary",
        disabled=(st.session_state.image_pil is None),
        use_container_width=True,
    )

# ── RIGHT: Results ──────────────────────────

with col_right:
    if identify_clicked and st.session_state.image_pil is not None:
        # Run model
        with st.spinner("🔍 Analysing monument with CLIP…"):
            try:
                model = get_model()
                if not model.is_loaded():
                    st.info("📦 Loading CLIP model (first run — ~1 min)…")

                prepared = prepare_for_clip(st.session_state.image_pil)
                predictions, latency = model.predict(prepared, top_k=3)
                st.session_state.predictions = predictions
                st.session_state.latency = latency
                st.session_state.show_full_history = False

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.session_state.predictions = None

    # ---- Display results ----
    if st.session_state.predictions:
        predictions = st.session_state.predictions
        top = predictions[0]
        conf = top["confidence"]
        name = top["name"]

        info = get_monument_info(name)

        # ── Result card ──
        color = confidence_color(conf)
        label = confidence_label(conf)
        emoji = confidence_emoji(conf)

        st.markdown(
            f"""
            <div class="result-card">
                <p class="monument-name">🏛️ {name}</p>
                <span class="confidence-badge" style="background:{color}22;color:{color};border:1.5px solid {color};">
                    {emoji} {label} — {conf*100:.1f}%
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Confidence bar for top-3 ──
        st.markdown("##### 📊 Top-3 Predictions")
        bar_html = ""
        for pred in predictions:
            bar_color = confidence_color(pred["confidence"])
            width_pct = pred["confidence"] * 100
            bar_html += f"""
            <div class="top3-row">
                <span class="top3-label">{'🥇' if pred['rank']==1 else '🥈' if pred['rank']==2 else '🥉'} {pred['name']}</span>
                <div class="top3-bar-bg">
                    <div class="top3-bar-fill" style="width:{width_pct:.1f}%;background:{bar_color};"></div>
                </div>
                <span class="top3-pct">{width_pct:.1f}%</span>
            </div>
            """
        st.markdown(bar_html, unsafe_allow_html=True)

        if st.session_state.latency:
            st.caption(f"⚡ Inference time: {st.session_state.latency:.0f} ms")

        st.divider()

        if info:
            # ── History ──
            st.markdown("##### 📜 History")
            history_text = info.get("history", "No history available.")
            short = truncate(history_text, 350)

            if st.session_state.show_full_history:
                st.markdown(
                    f'<div class="history-box">{history_text}</div>',
                    unsafe_allow_html=True,
                )
                if st.button("▲ Show less", key="less"):
                    st.session_state.show_full_history = False
                    st.rerun()
            else:
                st.markdown(
                    f'<div class="history-box">{short}</div>',
                    unsafe_allow_html=True,
                )
                if len(history_text) > 350:
                    if st.button("▼ Read full history", key="more"):
                        st.session_state.show_full_history = True
                        st.rerun()

            st.divider()

            # ── Visit info card ──
            st.markdown("##### 🗺️ Visiting Information")
            st.markdown(
                f"""
                <div class="info-grid">
                    <div class="info-cell">
                        <div class="info-label">⏰ Timings</div>
                        <div class="info-value">{info.get("timings", "N/A")}</div>
                    </div>
                    <div class="info-cell">
                        <div class="info-label">🎟️ Ticket Price</div>
                        <div class="info-value">{info.get("ticket_price", "N/A")}</div>
                    </div>
                    <div class="info-cell">
                        <div class="info-label">📍 Address</div>
                        <div class="info-value">{info.get("location", "Hampi, Karnataka")}</div>
                    </div>
                    <div class="info-cell">
                        <div class="info-label">🌅 Best Time</div>
                        <div class="info-value">{info.get("best_time", "N/A")}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Tags
            tags = info.get("tags", [])
            if tags:
                st.markdown(
                    "  ".join(f"`{t}`" for t in tags),
                    unsafe_allow_html=False,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Action buttons ──
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                maps_url = make_maps_url(info)
                st.link_button(
                    "📍 Open in Google Maps",
                    url=maps_url,
                    use_container_width=True,
                )
            with btn_col2:
                wiki_url = make_wikipedia_url(name)
                st.link_button(
                    "📖 Wikipedia",
                    url=wiki_url,
                    use_container_width=True,
                )

        else:
            st.warning("⚠️ Monument metadata not found — try another image.")

    elif not identify_clicked:
        # Instructions placeholder
        st.markdown(
            """
            <div style="
                border-radius: 14px;
                padding: 2.5rem 2rem;
                background: #fdf9f4;
                border: 1.5px solid #e8d5bb;
                text-align: center;
                color: #7a5230;
            ">
                <div style="font-size:3rem">🏯</div>
                <p style="font-size:1.05rem;font-weight:600;margin-top:0.8rem;color:#3d2010;">
                    Ready to identify a monument
                </p>
                <p style="font-size:0.9rem;line-height:1.6;opacity:0.8">
                    Upload a clear photo of any Hampi monument on the left,<br>
                    then click <strong>Identify Monument</strong> to get instant results.
                </p>
                <hr style="border-color:#e8d5bb;margin:1.2rem 0">
                <p style="font-size:0.82rem;opacity:0.7">
                    🏛️ Supports 10 monuments — Virupaksha Temple, Stone Chariot, 
                    Lotus Mahal, Elephant Stables &amp; more
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.markdown(
    """
    <div class="footer">
        🏛️ Hampi Monument Identifier · T12.5 · Built with OpenAI CLIP &amp; Streamlit
        · Data: Wikimedia Commons &amp; Wikipedia
        · Hampi is a UNESCO World Heritage Site since 1986
    </div>
    """,
    unsafe_allow_html=True,
)
