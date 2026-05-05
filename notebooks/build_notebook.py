"""
build_notebook.py
Run this script to regenerate evaluation.ipynb with full 3-mode support.
    python notebooks/build_notebook.py
"""
import nbformat as nbf
import os
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}

cells = []

# ── CELL 0 – Title ───────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
# T12.5 — Hampi Monument Identifier: Evaluation Notebook

This notebook lets you **interactively** evaluate and compare all three classification modes:

| Mode | Description |
|------|-------------|
| `zero_shot` | Pure CLIP cosine-similarity + 10-prompt ensembling (no training) |
| `linear_probe` | Logistic Regression on frozen CLIP features (light fine-tuning) |
| `hybrid` | Weighted blend of both (configurable `ensemble_weight`) |

**Sections**
1. Setup & model loading
2. Train / load the Linear Probe
3. Single-image prediction (all 3 modes)
4. Batch evaluation on test set
5. Per-class accuracy table
6. Accuracy comparison bar chart
7. Confusion matrices
8. Confidence distribution plots
"""))

# ── CELL 1 – Imports ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
import sys, os
sys.path.insert(0, os.path.abspath('..'))

from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

from model.clip_model import (
    HampiCLIPModel, MONUMENT_NAMES, FOLDER_TO_CLASS,
    MODE_ZERO_SHOT, MODE_LINEAR_PROBE, MODE_HYBRID,
)
from model.linear_probe import LinearProbeClassifier, _PROBE_PATH
from utils.preprocess import prepare_for_clip

plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.family'] = 'DejaVu Sans'

DATA_DIR  = Path('..') / 'data'
TEST_DIR  = DATA_DIR / 'test_images'
TRAIN_DIR = DATA_DIR / 'train_images'

print("Imports OK")
print(f"Test dir exists : {TEST_DIR.exists()}")
print(f"Train dir exists: {TRAIN_DIR.exists()}")
"""))

# ── CELL 2 – Load CLIP model ──────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 1. Load CLIP Model"))
cells.append(nbf.v4.new_code_cell("""\
import time

t0 = time.time()
model = HampiCLIPModel()
prompts_path = str(DATA_DIR / 'prompts.json')
model.load_with_prompts(prompts_path if os.path.exists(prompts_path) else None)
print(f"CLIP loaded in {time.time()-t0:.1f}s  |  device: {model.device}")
"""))

# ── CELL 3 – Train / load probe ───────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 2. Train / Load the Linear Probe"))
cells.append(nbf.v4.new_code_cell("""\
# Set FORCE_RETRAIN = True to always retrain, False to reuse existing .pkl
FORCE_RETRAIN = False

if FORCE_RETRAIN or not LinearProbeClassifier.exists(_PROBE_PATH):
    print("Training Linear Probe ...")
    probe = model.train_and_save_probe(verbose=True)
    print(f"Trained on classes: {probe.trained_classes}")
else:
    model.load_probe()
    print("Probe loaded from disk.")
    probe = model._probe
    print(f"Probe trained classes: {probe.trained_classes}")
    print(f"Probe embed dim: {probe.embed_dim}")
"""))

# ── CELL 4 – Single image prediction ─────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 3. Single Image Prediction — All 3 Modes"))
cells.append(nbf.v4.new_code_cell("""\
# Change this path to any image you want to test
IMG_PATH = str(TEST_DIR / 'Lotus_Mahal' / '000_lotus_mahal.jpg')

img = Image.open(IMG_PATH).convert('RGB')
img = prepare_for_clip(img)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].imshow(img)
axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

mode_labels = {
    MODE_ZERO_SHOT:    'Zero-Shot CLIP',
    MODE_LINEAR_PROBE: 'Linear Probe',
    MODE_HYBRID:       'Hybrid (w=0.7)',
}
colors = {MODE_ZERO_SHOT: '#1565C0', MODE_LINEAR_PROBE: '#2E7D32', MODE_HYBRID: '#E65100'}

for ax, mode in zip(axes[1:], [MODE_ZERO_SHOT, MODE_LINEAR_PROBE, MODE_HYBRID]):
    preds, lat = model.predict(img, top_k=5, mode=mode, ensemble_weight=0.7)
    names  = [p['name'] for p in preds]
    confs  = [p['confidence']*100 for p in preds]
    bars = ax.barh(names[::-1], confs[::-1], color=colors[mode], alpha=0.85)
    ax.set_xlabel('Confidence (%)')
    ax.set_title(f"{mode_labels[mode]}\\n({lat:.0f} ms)", fontsize=10, fontweight='bold')
    ax.set_xlim(0, 100)
    for bar, val in zip(bars, confs[::-1]):
        ax.text(val+0.5, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=8)

plt.suptitle('Top-5 Predictions per Mode', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

# ── CELL 5 – Batch eval helper ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 4. Batch Evaluation on Test Set"))
cells.append(nbf.v4.new_code_cell("""\
def collect_test_images(test_dir):
    samples = []
    for folder in sorted(Path(test_dir).iterdir()):
        if not folder.is_dir():
            continue
        cls = FOLDER_TO_CLASS.get(folder.name)
        if cls is None:
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}:
                samples.append((p, cls))
    return samples

def run_evaluation(model, samples, mode, ensemble_weight=0.7, verbose=True):
    per_class_t1 = defaultdict(int)
    per_class_t3 = defaultdict(int)
    per_class_n  = defaultdict(int)
    per_class_conf = defaultdict(list)
    total_t1 = total_t3 = total_lat = 0

    for i, (img_path, true_cls) in enumerate(samples):
        if verbose and (i+1) % 20 == 0:
            print(f"  {i+1}/{len(samples)}")
        try:
            img = Image.open(img_path).convert('RGB')
            img = prepare_for_clip(img)
        except Exception:
            continue

        preds, lat = model.predict(img, top_k=3, mode=mode, ensemble_weight=ensemble_weight)
        pred_names = [p['name'] for p in preds]

        t1 = int(pred_names[0] == true_cls if pred_names else False)
        t3 = int(true_cls in pred_names)
        total_t1  += t1
        total_t3  += t3
        total_lat += lat
        per_class_n[true_cls]    += 1
        per_class_t1[true_cls]   += t1
        per_class_t3[true_cls]   += t3
        if preds:
            per_class_conf[true_cls].append(preds[0]['confidence'])

    n = sum(per_class_n.values())
    per_class = {}
    for cls in MONUMENT_NAMES:
        tot  = per_class_n.get(cls, 0)
        c1   = per_class_t1.get(cls, 0)
        c3   = per_class_t3.get(cls, 0)
        confs = per_class_conf.get(cls, [])
        per_class[cls] = {
            'total': tot, 'top1_acc': c1/tot if tot else 0,
            'top3_acc': c3/tot if tot else 0,
            'avg_conf': float(np.mean(confs)) if confs else 0,
        }
    return {
        'mode': mode, 'n': n,
        'top1': total_t1/n if n else 0,
        'top3': total_t3/n if n else 0,
        'avg_conf': float(np.mean([v for vals in per_class_conf.values() for v in vals])),
        'avg_lat': total_lat/n if n else 0,
        'per_class': per_class,
    }

samples = collect_test_images(TEST_DIR)
# Filter to only real (non-stub) images
real_samples = []
for p, cls in samples:
    try:
        if p.stat().st_size > 1000:
            real_samples.append((p, cls))
    except Exception:
        pass

print(f"Real test images: {len(real_samples)}")
cls_counts = defaultdict(int)
for _, c in real_samples:
    cls_counts[c] += 1
for cls in MONUMENT_NAMES:
    cnt = cls_counts.get(cls, 0)
    if cnt:
        print(f"  {cls:<40} {cnt}")
"""))

# ── CELL 6 – Run eval ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
ENSEMBLE_WEIGHT = 0.7  # adjust as desired

print("Evaluating Zero-Shot ...")
r_zs = run_evaluation(model, real_samples, MODE_ZERO_SHOT)
print(f"  Top-1: {r_zs['top1']*100:.1f}%  Top-3: {r_zs['top3']*100:.1f}%")

print("\\nEvaluating Linear Probe ...")
r_lp = run_evaluation(model, real_samples, MODE_LINEAR_PROBE)
print(f"  Top-1: {r_lp['top1']*100:.1f}%  Top-3: {r_lp['top3']*100:.1f}%")

print("\\nEvaluating Hybrid ...")
r_hy = run_evaluation(model, real_samples, MODE_HYBRID, ENSEMBLE_WEIGHT)
print(f"  Top-1: {r_hy['top1']*100:.1f}%  Top-3: {r_hy['top3']*100:.1f}%")

all_results = [r_zs, r_lp, r_hy]
print("\\nDone.")
"""))

# ── CELL 7 – Per-class table ──────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 5. Per-Class Accuracy Table"))
cells.append(nbf.v4.new_code_cell("""\
trained_cls = set(probe.trained_classes) if probe else set()

header = f"{'Monument':<38} {'N':>4}  {'ZS T1':>6}  {'LP T1':>6}  {'HY T1':>6}  {'ZS T3':>6}  {'LP T3':>6}  {'Trained?':>8}"
print(header)
print('-' * len(header))

for cls in MONUMENT_NAMES:
    r_zs_c = r_zs['per_class'].get(cls, {})
    r_lp_c = r_lp['per_class'].get(cls, {})
    r_hy_c = r_hy['per_class'].get(cls, {})
    n   = r_zs_c.get('total', 0)
    if n == 0:
        continue
    zst1 = r_zs_c.get('top1_acc', 0)*100
    lpt1 = r_lp_c.get('top1_acc', 0)*100
    hyt1 = r_hy_c.get('top1_acc', 0)*100
    zst3 = r_zs_c.get('top3_acc', 0)*100
    lpt3 = r_lp_c.get('top3_acc', 0)*100
    tr   = 'Yes' if cls in trained_cls else '-'
    print(f"{cls:<38} {n:>4}  {zst1:>5.1f}%  {lpt1:>5.1f}%  {hyt1:>5.1f}%  {zst3:>5.1f}%  {lpt3:>5.1f}%  {tr:>8}")

print('-' * len(header))
print(f"{'OVERALL':<38} {r_zs['n']:>4}  {r_zs['top1']*100:>5.1f}%  {r_lp['top1']*100:>5.1f}%  {r_hy['top1']*100:>5.1f}%  {r_zs['top3']*100:>5.1f}%  {r_lp['top3']*100:>5.1f}%")
"""))

# ── CELL 8 – Bar chart comparison ────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 6. Accuracy Comparison Bar Chart"))
cells.append(nbf.v4.new_code_cell("""\
labels   = ['Zero-Shot CLIP', 'Linear Probe\\n(fine-tuned)', f'Hybrid\\n(w={ENSEMBLE_WEIGHT})']
top1_vals = [r['top1']*100 for r in all_results]
top3_vals = [r['top3']*100 for r in all_results]
colors_bar = ['#1565C0', '#2E7D32', '#E65100']

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - width/2, top1_vals, width, label='Top-1 Accuracy', color=colors_bar, alpha=0.9)
b2 = ax.bar(x + width/2, top3_vals, width, label='Top-3 Accuracy', color=colors_bar, alpha=0.45, hatch='//')

for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, color='#555')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('T12.5 — Hampi Monument Classifier: Mode Comparison', fontsize=13, fontweight='bold')
ax.set_ylim(0, 105)
ax.axhline(y=50, color='grey', linestyle='--', alpha=0.4, label='Random baseline (10 classes)')
legend_patches = [
    mpatches.Patch(facecolor='grey', label='Top-1 (solid)'),
    mpatches.Patch(facecolor='grey', alpha=0.4, hatch='//', label='Top-3 (hatched)'),
    plt.Line2D([0],[0], color='grey', linestyle='--', label='Random (50%)'),
]
ax.legend(handles=legend_patches, fontsize=9, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('mode_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: mode_comparison.png")
"""))

# ── CELL 9 – Per-class bar chart ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 7. Per-Class Top-1 Accuracy (all modes)"))
cells.append(nbf.v4.new_code_cell("""\
active_cls = [c for c in MONUMENT_NAMES if r_zs['per_class'].get(c,{}).get('total',0) > 0]
x = np.arange(len(active_cls))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - width,   [r_zs['per_class'][c]['top1_acc']*100 for c in active_cls], width, label='Zero-Shot', color='#1565C0', alpha=0.85)
ax.bar(x,           [r_lp['per_class'][c]['top1_acc']*100 for c in active_cls], width, label='Linear Probe', color='#2E7D32', alpha=0.85)
ax.bar(x + width,   [r_hy['per_class'][c]['top1_acc']*100 for c in active_cls], width, label=f'Hybrid (w={ENSEMBLE_WEIGHT})', color='#E65100', alpha=0.85)

ax.set_xticks(x)
short_names = [c.replace('temple hill complex','hill').replace('Hemakuta ','Hemakuta\\n') for c in active_cls]
ax.set_xticklabels(short_names, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
ax.set_title('Per-Class Top-1 Accuracy — All Modes', fontsize=13, fontweight='bold')
ax.set_ylim(0, 110)
ax.axhline(y=100, color='green', linestyle=':', alpha=0.5)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: per_class_accuracy.png")
"""))

# ── CELL 10 – Confidence distributions ───────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 8. Confidence Distributions"))
cells.append(nbf.v4.new_code_cell("""\
def get_all_confs(model, samples, mode, ew=0.7):
    correct, wrong = [], []
    for img_path, true_cls in samples:
        try:
            img = Image.open(img_path).convert('RGB')
            img = prepare_for_clip(img)
            preds, _ = model.predict(img, top_k=1, mode=mode, ensemble_weight=ew)
            if not preds: continue
            c = preds[0]['confidence']
            (correct if preds[0]['name']==true_cls else wrong).append(c)
        except Exception:
            pass
    return correct, wrong

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
mode_info = [
    (MODE_ZERO_SHOT,    'Zero-Shot CLIP',    '#1565C0'),
    (MODE_LINEAR_PROBE, 'Linear Probe',      '#2E7D32'),
    (MODE_HYBRID,       f'Hybrid (w={ENSEMBLE_WEIGHT})', '#E65100'),
]
for ax, (mode, lbl, col) in zip(axes, mode_info):
    cor, wro = get_all_confs(model, real_samples, mode, ENSEMBLE_WEIGHT)
    ax.hist(cor, bins=15, alpha=0.75, color=col,   label=f'Correct (n={len(cor)})', density=True)
    ax.hist(wro, bins=15, alpha=0.45, color='red',  label=f'Wrong   (n={len(wro)})', density=True)
    ax.set_title(lbl, fontsize=11, fontweight='bold')
    ax.set_xlabel('Top-1 Confidence')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.suptitle('Confidence Distribution: Correct vs Wrong Predictions', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confidence_dist.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: confidence_dist.png")
"""))

# ── CELL 11 – Summary ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 9. Summary Table"))
cells.append(nbf.v4.new_code_cell("""\
print("=" * 72)
print(f"  {'Mode':<30} {'Top-1':>7}  {'Top-3':>7}  {'Avg Conf':>9}  {'Latency':>8}")
print("-" * 72)
for r, lbl in zip(all_results, ['Zero-Shot CLIP', 'Linear Probe', f'Hybrid (w={ENSEMBLE_WEIGHT})']):
    marker = '  <- BEST' if r['top1'] == max(x['top1'] for x in all_results) else ''
    print(f"  {lbl:<30} {r['top1']*100:>6.1f}%  {r['top3']*100:>6.1f}%  {r['avg_conf']*100:>8.1f}%  {r['avg_lat']:>6.0f}ms{marker}")
print("=" * 72)
print(f"\\nTest images: {r_zs['n']} | Classes with data: {len(active_cls)}/10")
print("Note: 4 classes (Virupaksha, Vittala, Zenana, Queen's Bath) have only")
print("placeholder files in test_images/ — they were not included in evaluation.")
"""))

nb.cells = cells

out_path = Path(__file__).parent / "evaluation.ipynb"
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook written to: {out_path}")
