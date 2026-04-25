# 🏛️ Hampi Monument Identifier — T12.5

> **Upload a photo of any Hampi monument → get instant name, history, and visiting details.**

Built with **OpenAI CLIP** (zero-shot) + **Streamlit** as part of the Monuments & Heritage Identifier project (T12 family).

---

## 📸 Demo

| Step | Screenshot |
|------|-----------|
| Upload image | Clean file-upload panel with preview |
| Click "Identify" | CLIP inference in ~200–800 ms |
| See results | Monument name, confidence bar, history, timings, tickets, Maps link |

---

## 🏯 Supported Monuments (10)

| Monument | Type | Entry |
|----------|------|-------|
| Virupaksha Temple | Active temple | Free |
| Vittala Temple (Stone Chariot) | UNESCO icon | ₹40 / ₹600 |
| Lotus Mahal | Palace pavilion | ₹40 / ₹600 |
| Elephant Stables | Royal stables | ₹40 / ₹600 |
| Hazara Rama Temple | Royal chapel | ₹40 / ₹600 |
| Achyutaraya Temple | Ruined temple | Free |
| Matanga Hill | Viewpoint / trek | Free |
| Underground Shiva Temple | Subterranean shrine | Free |
| Queen's Bath | Royal bath | ₹40 / ₹600 |
| Hampi Bazaar | Ancient market street | Free |

---

## ⚙️ Setup

### 1. Clone / extract the project
```bash
cd hampi_identifier
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** First run downloads the CLIP model weights (~600 MB) from HuggingFace Hub.  
> Weights are cached in `~/.cache/huggingface/` for subsequent runs.

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🧠 How it Works

```
User uploads image
       │
       ▼
  PIL + EXIF fix
       │
       ▼
CLIP image encoder  ──→  512-d embedding
       │
       ▼
Cosine similarity vs  ──→  10 monument text embeddings
3-prompt ensemble          (pre-computed at startup)
       │
       ▼
Softmax → top-3 predictions with confidence %
       │
       ▼
metadata.json lookup → history, timings, tickets, Maps URL
       │
       ▼
Streamlit UI renders results
```

### Prompt Ensembling
Each monument has **3 descriptive text prompts** (e.g. *"a photo of the stone chariot at Vittala Temple Hampi"*). Their CLIP text embeddings are mean-pooled into a single representative vector. This substantially improves zero-shot accuracy over bare monument names.

---

## 📁 Project Structure

```
hampi_identifier/
├── app.py                    ← Streamlit frontend
├── requirements.txt
├── README.md
│
├── model/
│   ├── __init__.py
│   └── clip_model.py         ← CLIP zero-shot classifier
│
├── data/
│   └── metadata.json         ← Monument info (history, timings, tickets, maps)
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py         ← Image loading + quality checks
│   └── helpers.py            ← Metadata access, formatting
│
└── notebooks/
    └── evaluation.ipynb      ← Batch eval + confidence plots
```

---

## 📊 Expected Performance

| Metric | Typical value |
|--------|--------------|
| Top-1 accuracy (good photos) | ~65–80% |
| Top-3 accuracy | ~85–95% |
| Inference latency (CPU) | 300–900 ms |
| Inference latency (GPU) | 30–80 ms |

> CLIP is zero-shot — no training required. Accuracy improves with clear, well-lit photos showing distinctive architectural features.

---

## 🚀 Deployment

### Streamlit Cloud
1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Deploy (free tier available)

### Hugging Face Spaces
1. Create a new Space with **Streamlit** SDK
2. Upload all files
3. HF auto-installs `requirements.txt`

---

## 🔧 Extending the App

### Add more monuments
1. Add entry to `data/metadata.json`
2. Add monument name to `MONUMENT_NAMES` in `model/clip_model.py`
3. Add 3 descriptive prompts to `MONUMENT_PROMPTS`

### Switch to fine-tuned model
Replace the `predict()` method in `model/clip_model.py` with a ResNet/EfficientNet classifier.  
Keep the same return format: `[{"name": str, "confidence": float, "rank": int}, ...]`

### Add audio narration
```python
from gtts import gTTS
tts = gTTS(info['history'][:500], lang='en')
tts.save('/tmp/narration.mp3')
st.audio('/tmp/narration.mp3')
```

---

## 📚 Data Sources

- **Images:** [Wikimedia Commons — Group of Monuments at Hampi](https://commons.wikimedia.org/wiki/Category:Group_of_monuments_at_Hampi)
- **Metadata:** Wikipedia articles for each monument
- **Model:** [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) via HuggingFace Transformers

---

## 🏷️ License

Code: MIT  
Monument images: Creative Commons (Wikimedia Commons)  
Metadata: Wikipedia CC BY-SA 3.0

---

*Hampi is a UNESCO World Heritage Site since 1986. The Vijayanagara Empire (1336–1646 CE) made it one of the largest cities in the medieval world.*
