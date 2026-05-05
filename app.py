"""
app.py — Hampi Monument Identifier  |  T12.5
Streamlit frontend supporting three classification modes:
  1. Zero-Shot CLIP        — Pure cosine similarity with prompt ensembling
  2. Linear Probe          — Logistic Regression on frozen CLIP features (light fine-tuning)
  3. Hybrid Ensemble       — Weighted blend of both

Run with:
    streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from PIL import Image
import time

from model.clip_model import (
    get_model,
    MONUMENT_NAMES,
    MODE_ZERO_SHOT,
    MODE_LINEAR_PROBE,
    MODE_HYBRID,
)
from model.linear_probe import LinearProbeClassifier, _PROBE_PATH
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
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown(
    """
    <style>
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

    /* ---- Mode badge ---- */
    .mode-badge-zs  { background:#e3f2fd;color:#1565c0;border:1.5px solid #1565c0; }
    .mode-badge-lp  { background:#e8f5e9;color:#2e7d32;border:1.5px solid #2e7d32; }
    .mode-badge-hyb { background:#fff3e0;color:#e65100;border:1.5px solid #e65100; }
    .mode-badge-zs, .mode-badge-lp, .mode-badge-hyb {
        display:inline-block;padding:3px 12px;border-radius:20px;
        font-size:0.8rem;font-weight:600;margin-bottom:0.6rem;
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
    .source-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        background: #f5f0eb;
        color: #7a5230;
        margin-top: 4px;
        margin-left: 6px;
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
    .top3-label { min-width: 210px; font-size: 0.88rem; font-weight: 500; color: #3d2010; }
    .top3-bar-bg {
        flex: 1; height: 10px; background: #e8d5bb;
        border-radius: 6px; overflow: hidden;
    }
    .top3-bar-fill { height: 100%; border-radius: 6px; }
    .top3-pct { font-size: 0.82rem; color: #7a5230; min-width: 45px; text-align: right; }

    /* ---- Accuracy comparison table ---- */
    .acc-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        margin-top: 0.8rem;
    }
    .acc-table th {
        background: #5c2d0e;
        color: #f5d78e;
        padding: 8px 12px;
        text-align: left;
        font-weight: 600;
    }
    .acc-table td {
        padding: 7px 12px;
        border-bottom: 1px solid #e8d5bb;
        color: #3d2010;
    }
    .acc-table tr:nth-child(even) td { background: #fdf9f4; }
    .acc-best { font-weight: 700; color: #2e7d32; }

    /* ---- Buttons ---- */
    div[data-testid="stButton"] button {
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    /* ---- Upload area ---- */
    [data-testid="stFileUploader"] { border-radius: 12px; }

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

    /* ---- Probe training status ---- */
    .probe-trained {
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: #2e7d32;
    }
    .probe-missing {
        background: #fff3e0;
        border-left: 4px solid #e65100;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: #e65100;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────

defaults = {
    "predictions": None,
    "image_pil": None,
    "latency": None,
    "show_full_history": False,
    "mode": MODE_ZERO_SHOT,
    "ensemble_weight": 0.7,
    "probe_trained": LinearProbeClassifier.exists(_PROBE_PATH),
    "probe_training": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
# Hero banner
# ──────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-banner">
        <div style="font-size:2.8rem;margin-bottom:0.3rem;">🏛️</div>
        <p class="hero-title">Hampi Monument Identifier</p>
        <p class="hero-sub">Upload a photo of any Hampi monument — get instant name, history, and visiting details</p>
        <span class="hero-badge">🤖 Powered by OpenAI CLIP · Zero-Shot + Light Fine-Tuning · T12.5</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Sidebar — model config, probe training, monument list
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🤖 Classification Mode")

    mode_labels = {
        MODE_ZERO_SHOT:    "🔵 Zero-Shot CLIP",
        MODE_LINEAR_PROBE: "🟢 Linear Probe (Fine-Tuned)",
        MODE_HYBRID:       "🟠 Hybrid Ensemble",
    }
    selected_mode = st.radio(
        "Select prediction mode",
        options=list(mode_labels.keys()),
        format_func=lambda x: mode_labels[x],
        index=list(mode_labels.keys()).index(st.session_state.mode),
        label_visibility="collapsed",
    )
    st.session_state.mode = selected_mode

    # Mode description
    mode_descriptions = {
        MODE_ZERO_SHOT: (
            "**Zero-Shot CLIP** — No training required. Uses descriptive text prompts "
            "to match the image via cosine similarity. Accuracy: ~52–58%."
        ),
        MODE_LINEAR_PROBE: (
            "**Linear Probe** — A Logistic Regression trained on frozen CLIP image features "
            "(light fine-tuning). Requires training data. Best on trained classes."
        ),
        MODE_HYBRID: (
            "**Hybrid Ensemble** — Blends Zero-Shot and Linear Probe scores. "
            "Use the weight slider to control the balance."
        ),
    }
    st.caption(mode_descriptions[selected_mode])

    if selected_mode == MODE_HYBRID:
        weight = st.slider(
            "Probe weight (↑ more fine-tuning, ↓ more zero-shot)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.ensemble_weight,
            step=0.05,
            format="%.2f",
        )
        st.session_state.ensemble_weight = weight
        st.caption(
            f"Blending: **{weight:.0%} Linear Probe** + **{1-weight:.0%} Zero-Shot**"
        )

    st.divider()

    # ── Linear Probe training section ──
    st.markdown("### 🏋️ Linear Probe")
    probe_exists = LinearProbeClassifier.exists(_PROBE_PATH)
    st.session_state.probe_trained = probe_exists

    if probe_exists:
        st.markdown(
            '<div class="probe-trained">✅ Probe trained &amp; ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="probe-missing">⚠️ Probe not trained yet</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "Training uses 49 images across 6 classes from `data/train_images/`. "
        "Takes ~30–60 seconds on CPU."
    )

    if st.button("🚀 Train Linear Probe", use_container_width=True, type="primary"):
        with st.spinner("⚙️ Extracting CLIP features & training probe…"):
            try:
                model = get_model(use_enhanced_prompts=True)
                if not model.is_loaded():
                    st.info("📦 Loading CLIP model first (may take ~1 min)…")
                probe = model.train_and_save_probe(verbose=False)
                st.session_state.probe_trained = True
                st.success(f"✅ Probe trained on {len(probe.trained_classes)} classes!")
                # Reload probe into the model instance
                model.load_probe()
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

    st.divider()

    # ── Monument list ──
    st.markdown("### 🏛️ Supported Monuments")
    for m in sorted(MONUMENT_NAMES):
        st.markdown(f"- {m}")

    st.divider()
    st.markdown("### ⚙️ Model")
    st.code("openai/clip-vit-base-patch32", language=None)
    st.markdown("10-prompt ensemble per monument · Softmax probabilities")
    st.divider()
    st.markdown(
        "[📂 Dataset: Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Group_of_monuments_at_Hampi)",
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
        try:
            image_pil = load_image_from_upload(uploaded_file)
            quality = validate_image_quality(image_pil)

            if quality["warnings"]:
                for w in quality["warnings"]:
                    st.warning(w)

            st.image(
                image_pil,
                caption=f"📷 {uploaded_file.name}  ({image_pil.width}×{image_pil.height}px)",
                use_container_width=True,
            )

            st.session_state.image_pil = image_pil
            st.session_state.predictions = None

        except ValueError as e:
            st.error(f"❌ {e}")
            st.session_state.image_pil = None

    else:
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

    st.markdown("<br>", unsafe_allow_html=True)

    # Mode indicator pill
    mode_pill_class = {
        MODE_ZERO_SHOT:    "mode-badge-zs",
        MODE_LINEAR_PROBE: "mode-badge-lp",
        MODE_HYBRID:       "mode-badge-hyb",
    }[st.session_state.mode]
    mode_pill_text = mode_labels[st.session_state.mode]
    st.markdown(
        f'<span class="{mode_pill_class}">{mode_pill_text}</span>',
        unsafe_allow_html=True,
    )

    identify_clicked = st.button(
        "🔍  Identify Monument",
        type="primary",
        disabled=(st.session_state.image_pil is None),
        use_container_width=True,
    )

# ── RIGHT: Results ──────────────────────────

with col_right:
    if identify_clicked and st.session_state.image_pil is not None:
        with st.spinner("🔍 Analysing monument with CLIP…"):
            try:
                model = get_model(use_enhanced_prompts=True)
                if not model.is_loaded():
                    st.info("📦 Loading CLIP model (first run — ~1 min)…")

                # Load probe if needed
                if st.session_state.mode in (MODE_LINEAR_PROBE, MODE_HYBRID):
                    if not model.has_probe():
                        model.load_probe()

                prepared = prepare_for_clip(st.session_state.image_pil)
                predictions, latency = model.predict(
                    prepared,
                    top_k=3,
                    mode=st.session_state.mode,
                    ensemble_weight=st.session_state.ensemble_weight,
                )
                st.session_state.predictions = predictions
                st.session_state.latency = latency
                st.session_state.show_full_history = False

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.session_state.predictions = None

    # ── Display results ──
    if st.session_state.predictions:
        predictions = st.session_state.predictions
        top = predictions[0]
        conf = top["confidence"]
        name = top["name"]
        source = top.get("source", "CLIP")

        info = get_monument_info(name)

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
                <span class="source-tag">📡 {source}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Top-3 predictions bar ──
        st.markdown("##### 📊 Top-3 Predictions")
        bar_html = ""
        for pred in predictions:
            bar_color = confidence_color(pred["confidence"])
            width_pct = pred["confidence"] * 100
            rank_emoji = "🥇" if pred["rank"] == 1 else "🥈" if pred["rank"] == 2 else "🥉"
            bar_html += f"""
            <div class="top3-row">
                <span class="top3-label">{rank_emoji} {pred['name']}</span>
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
            history_text = info.get("summary", info.get("history", "No history available."))
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
                        <div class="info-value">{info.get("timings", "Sunrise to Sunset")}</div>
                    </div>
                    <div class="info-cell">
                        <div class="info-label">🎟️ Ticket Price</div>
                        <div class="info-value">{info.get("ticket_price", "₹40 Indian / ₹600 Foreign")}</div>
                    </div>
                    <div class="info-cell">
                        <div class="info-label">📍 Address</div>
                        <div class="info-value">{info.get("location", "Hampi, Karnataka, India")}</div>
                    </div>
                    <div class="info-cell">
                        <div class="info-label">🌅 Best Time</div>
                        <div class="info-value">{info.get("best_time", "Oct – Mar")}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            tags = info.get("tags", [])
            if tags:
                st.markdown(
                    "  ".join(f"`{t}`" for t in tags),
                    unsafe_allow_html=False,
                )

            st.markdown("<br>", unsafe_allow_html=True)

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
            st.warning("⚠️ Monument metadata not found in database.")

    elif not identify_clicked:
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
                    select your <strong>Classification Mode</strong> from the sidebar,<br>
                    then click <strong>Identify Monument</strong>.
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
