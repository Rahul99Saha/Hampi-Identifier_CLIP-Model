# 🏛️ Hampi Monument Identifier — T12.5

> **Upload a photo of any Hampi monument → get instant name, history, and visiting details.**

Built with **OpenAI CLIP** + **Streamlit** as part of the Monuments & Heritage Identifier project (T12 family).  
Supports **three classification modes**: Pure Zero-Shot, Linear Probe (light fine-tuning), and a Hybrid Ensemble.

---

## 🏯 Supported Monuments (10)

| Monument | Type | Training Data |
|----------|------|:---:|
| Virupaksha Temple | Active temple (7th c.) | — |
| Vittala Temple (Stone Chariot) | UNESCO icon | — |
| Lotus Mahal | Palace pavilion | ✅ |
| Elephant Stables | Royal stables | ✅ |
| Hampi Bazaar | Ancient marketplace | ✅ |
| Zenana Enclosure | Royal women's quarters | — |
| Royal Centre | King's court | ✅ |
| Queen's Bath | Royal bathing complex | — |
| Hemakuta temple hill complex | Hillside temples | ✅ |
| Monolithic Bull (Nandi) | Stone sculpture | ✅ |

> ✅ = has training images for Linear Probe · — = zero-shot only

---

## 🤖 Classification Modes

| Mode | Method | Training Required | Expected Top-1 Acc |
|------|--------|:-----------------:|:-----------------:|
| **Zero-Shot CLIP** | Cosine similarity vs 10-prompt ensembles | ❌ No | ~52–58% |
| **Linear Probe** | Logistic Regression on frozen CLIP features | ✅ Yes (6 classes) | ~60–70% |
| **Hybrid Ensemble** | Weighted blend of both | ✅ Yes | Configurable |

The **Hybrid** mode lets you pick any blend via a sidebar slider:
- Weight = 1.0 → pure Linear Probe
- Weight = 0.5 → equal mix
- Weight = 0.0 → pure Zero-Shot

---

## ⚙️ Setup

### 1. Clone / extract the project
```bash
cd hampi_identifier
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # Linux / macOS
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** First run downloads the CLIP model weights (~600 MB) from HuggingFace Hub.  
> Weights are cached in `~/.cache/huggingface/` for subsequent runs.

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🏋️ Training the Linear Probe

**Option A — from the Streamlit UI (recommended)**  
Click **"🚀 Train Linear Probe"** in the sidebar.  
Training takes ~30–60 seconds on CPU.

**Option B — from the command line**
```bash
python evaluate.py --train
```

This trains the Logistic Regression on `data/train_images/` (49 images, 6 classes)  
and saves the model to `model/hampi_classifier.pkl`.

---

## 📊 Running the Evaluation

```bash
# Evaluate ALL three modes and compare
python evaluate.py --train

# Evaluate a single mode
python evaluate.py --mode zero_shot
python evaluate.py --mode linear_probe
python evaluate.py --mode hybrid --weight 0.7

# Save results to JSON
python evaluate.py --save-results results.json
```

Sample output:
```
──────────────────────────────────────────────────────────────────────────────
  🏛️  Hampi Monument Identifier — Evaluation Suite  |  T12.5
──────────────────────────────────────────────────────────────────────────────

  📈 ACCURACY COMPARISON SUMMARY
  Mode                                     Top-1    Top-3    Avg Conf   Latency
  ─────────────────────────────────────────────────────────────────────────────
  ★ Hybrid (probe 70% + zero-shot 30%)    65.0%   88.3%     62.1%     310ms  ← BEST
    Linear Probe (fine-tuned)             63.3%   86.7%     58.4%     305ms
    Zero-Shot CLIP                        58.3%   80.8%     59.6%     108ms
```

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
       ├──[Zero-Shot]──→  Cosine similarity vs 10 monument text embeddings
       │                   (10-prompt ensemble, pre-computed at startup)
       │
       ├──[Linear Probe]──→  LogisticRegression.predict_proba()
       │                       trained on 49 images from 6 classes
       │
       └──[Hybrid]──→  w * probe_probs + (1-w) * zero_shot_probs
                         (weight slider in sidebar, default 0.70)
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

---

## 📁 Project Structure

```
hampi_identifier/
├── app.py                       ← Streamlit frontend (3-mode UI)
├── evaluate.py                  ← CLI evaluation script (all 3 modes)
├── requirements.txt
├── README.md
│
├── model/
│   ├── __init__.py
│   ├── clip_model.py            ← Unified classifier (zero-shot + hybrid)
│   ├── linear_probe.py          ← Linear Probe training & inference
│   └── hampi_classifier.pkl     ← Trained probe (auto-generated)
│
├── data/
│   ├── metadata.json            ← Monument info (history, timings, tickets)
│   ├── prompts.json             ← Enhanced 10-prompt ensembles per class
│   ├── classes.json             ← Class name list
│   ├── train_images/            ← 49 images across 6 classes (probe training)
│   │   ├── Elephant_Stables/    (7 images)
│   │   ├── Hampi_Bazaar/        (8 images)
│   │   ├── Hemakuta_temple_hill_complex/  (9 images)
│   │   ├── Lotus_Mahal/         (10 images)
│   │   ├── Monolithic_Bull/     (7 images)
│   │   └── Royal_Centre/        (8 images)
│   └── test_images/             ← 120 images across all 10 classes
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py            ← Image loading + quality checks
│   └── helpers.py               ← Metadata access, formatting
│
└── notebooks/
    └── evaluation.ipynb         ← Interactive evaluation & plots
```

---

## 📊 Expected Performance

| Mode | Top-1 Acc | Top-3 Acc | Latency (CPU) |
|------|-----------|-----------|---------------|
| Zero-Shot CLIP | ~52–58% | ~78–85% | 100–400ms |
| Linear Probe | ~60–68% | ~82–90% | 300–700ms |
| Hybrid (w=0.7) | ~63–70% | ~85–92% | 300–700ms |

> **Why Linear Probe > Zero-Shot on Hampi?**  
> All Hampi monuments are built from the same brown Deccan granite in the same environment.  
> CLIP has seen few labelled images of these specific ruins during pre-training.  
> The Linear Probe learns discriminative visual features directly from your dataset — 
> this is exactly the "light fine-tuning" the assignment describes.

---

## 🚀 Deployment

### Streamlit Cloud
1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Deploy (free tier available)

> **Note:** Include `model/hampi_classifier.pkl` in your repo so the probe is available in the cloud deployment.

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
