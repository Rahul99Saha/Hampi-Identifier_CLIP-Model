# -*- coding: utf-8 -*-
"""
evaluate.py — Benchmark all three prediction modes on the Hampi test set (T12.5)

Metrics computed:
  • Top-1 Accuracy (per class + overall)
  • Top-3 Accuracy (per class + overall)
  • Avg confidence score

Modes evaluated:
  1. Zero-Shot CLIP        (no training data)
  2. Linear Probe           (Logistic Regression on frozen CLIP features)
  3. Hybrid Ensemble        (configurable weight blend)

Usage:
    python evaluate.py                      # uses default ensemble_weight=0.7
    python evaluate.py --weight 0.8         # custom hybrid weight
    python evaluate.py --mode zero_shot     # evaluate single mode only
    python evaluate.py --train              # re-train probe before evaluation
    python evaluate.py --save-results results.json

Results are printed to console AND optionally saved as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from PIL import Image
from model.clip_model import (
    HampiCLIPModel,
    MONUMENT_NAMES,
    FOLDER_TO_CLASS,
    MODE_ZERO_SHOT,
    MODE_LINEAR_PROBE,
    MODE_HYBRID,
    ALL_MODES,
)
from model.linear_probe import LinearProbeClassifier, _PROBE_PATH

_TEST_DIR  = _ROOT / "data" / "test_images"
_TRAIN_DIR = _ROOT / "data" / "train_images"

SEP = "-" * 78


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_test_images(test_dir: Path) -> list[tuple[Path, str]]:
    """Return list of (image_path, true_class_name) from test_images/."""
    samples = []
    for folder in sorted(test_dir.iterdir()):
        if not folder.is_dir():
            continue
        class_name = FOLDER_TO_CLASS.get(folder.name)
        if class_name is None:
            print(f"  [WARN] Unknown test folder '{folder.name}' — skipping")
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                samples.append((p, class_name))
    return samples


def evaluate_mode(
    model: HampiCLIPModel,
    samples: list[tuple[Path, str]],
    mode: str,
    ensemble_weight: float = 0.7,
    verbose: bool = True,
) -> dict:
    """
    Run inference on all test samples for a given mode.

    Returns a results dict with per-class and overall accuracy.
    """
    from utils.preprocess import prepare_for_clip

    per_class_correct_top1: dict[str, int] = defaultdict(int)
    per_class_correct_top3: dict[str, int] = defaultdict(int)
    per_class_total:        dict[str, int] = defaultdict(int)
    per_class_conf:         dict[str, list] = defaultdict(list)

    total_top1 = 0
    total_top3 = 0
    total_latency = 0.0

    n = len(samples)
    for idx, (img_path, true_class) in enumerate(samples, 1):
        if verbose and idx % 20 == 0:
            print(f"    [{idx}/{n}] …")

        try:
            img = Image.open(img_path).convert("RGB")
            img = prepare_for_clip(img)
        except Exception as e:
            print(f"  [WARN] Cannot load {img_path.name}: {e}")
            continue

        preds, lat = model.predict(
            img,
            top_k=3,
            mode=mode,
            ensemble_weight=ensemble_weight,
        )

        pred_names = [p["name"] for p in preds]
        top1_correct = (pred_names[0] == true_class) if pred_names else False
        top3_correct = (true_class in pred_names)

        total_top1     += int(top1_correct)
        total_top3     += int(top3_correct)
        total_latency  += lat

        per_class_total[true_class]  += 1
        per_class_correct_top1[true_class] += int(top1_correct)
        per_class_correct_top3[true_class] += int(top3_correct)
        if preds:
            per_class_conf[true_class].append(preds[0]["confidence"])

    n_valid = sum(per_class_total.values())

    # Build per-class breakdown
    per_class = {}
    for cls in MONUMENT_NAMES:
        tot = per_class_total.get(cls, 0)
        c1  = per_class_correct_top1.get(cls, 0)
        c3  = per_class_correct_top3.get(cls, 0)
        confs = per_class_conf.get(cls, [])
        per_class[cls] = {
            "total":          tot,
            "top1_correct":   c1,
            "top3_correct":   c3,
            "top1_acc":       c1 / tot if tot else 0.0,
            "top3_acc":       c3 / tot if tot else 0.0,
            "avg_confidence": float(sum(confs) / len(confs)) if confs else 0.0,
        }

    return {
        "mode":            mode,
        "ensemble_weight": ensemble_weight if mode == MODE_HYBRID else None,
        "n_images":        n_valid,
        "top1_accuracy":   total_top1 / n_valid if n_valid else 0.0,
        "top3_accuracy":   total_top3 / n_valid if n_valid else 0.0,
        "avg_confidence":  0.0,   # filled in by _recompute_avg_conf()
        "avg_latency_ms":  total_latency / n_valid if n_valid else 0.0,
        "per_class":       per_class,
    }


def _recompute_avg_conf(results: dict) -> dict:
    """Recompute avg_confidence from per_class breakdown."""
    total_conf = 0.0
    total_n    = 0
    for cls_info in results["per_class"].values():
        n = cls_info["total"]
        total_conf += cls_info["avg_confidence"] * n
        total_n    += n
    results["avg_confidence"] = total_conf / total_n if total_n else 0.0
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def _mode_label(mode: str, weight: float | None) -> str:
    if mode == MODE_HYBRID and weight is not None:
        return f"Hybrid (probe {weight:.0%} + zero-shot {1-weight:.0%})"
    return {
        MODE_ZERO_SHOT:    "Zero-Shot CLIP",
        MODE_LINEAR_PROBE: "Linear Probe (fine-tuned)",
        MODE_HYBRID:       "Hybrid Ensemble",
    }.get(mode, mode)


def print_per_class_table(results: dict):
    """Print a per-class accuracy breakdown table."""
    label = _mode_label(results["mode"], results.get("ensemble_weight"))
    print(f"\n{'  📊 Per-Class Results — ' + label:}")
    print(SEP)

    # Check which classes had training data
    from model.linear_probe import _TRAIN_DIR, FOLDER_TO_CLASS as F2C
    trained_classes = set()
    if _TRAIN_DIR.exists():
        for d in _TRAIN_DIR.iterdir():
            if d.is_dir() and F2C.get(d.name):
                trained_classes.add(F2C[d.name])

    header = f"  {'Monument':<35} {'N':>4}  {'Top-1':>6}  {'Top-3':>6}  {'Avg Conf':>9}  {'Trained?':>8}"
    print(header)
    print("  " + "-" * 74)

    for cls in MONUMENT_NAMES:
        info = results["per_class"].get(cls, {})
        tot  = info.get("total", 0)
        t1   = info.get("top1_acc", 0.0)
        t3   = info.get("top3_acc", 0.0)
        conf = info.get("avg_confidence", 0.0)
        tr   = "✅" if cls in trained_classes else "—"

        status = "✅" if t1 >= 0.70 else "⚠️" if t1 >= 0.45 else "❌"
        print(
            f"  {status} {cls:<33} {tot:>4}  {t1*100:>5.1f}%  {t3*100:>5.1f}%  {conf*100:>8.1f}%  {tr:>8}"
        )

    print("  " + "-" * 74)
    print(
        f"  {'OVERALL':<35} {results['n_images']:>4}  "
        f"{results['top1_accuracy']*100:>5.1f}%  "
        f"{results['top3_accuracy']*100:>5.1f}%  "
        f"{results['avg_confidence']*100:>8.1f}%"
    )
    print(SEP)


def print_summary_comparison(all_results: list[dict]):
    """Print a side-by-side summary comparison of all modes."""
    print(f"\n{'  📈 ACCURACY COMPARISON SUMMARY':}")
    print(SEP)

    header = f"  {'Mode':<40} {'Top-1':>7}  {'Top-3':>7}  {'Avg Conf':>9}  {'Latency':>8}"
    print(header)
    print("  " + "-" * 74)

    best_top1 = max(r["top1_accuracy"] for r in all_results)

    for r in all_results:
        label  = _mode_label(r["mode"], r.get("ensemble_weight"))
        t1     = r["top1_accuracy"]
        t3     = r["top3_accuracy"]
        conf   = r["avg_confidence"]
        lat    = r["avg_latency_ms"]
        marker = "  ← BEST" if t1 == best_top1 else ""
        bold   = "★ " if t1 == best_top1 else "  "
        print(
            f"  {bold}{label:<38} {t1*100:>6.1f}%  {t3*100:>6.1f}%  "
            f"{conf*100:>8.1f}%  {lat:>6.0f}ms{marker}"
        )

    print(SEP)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Ensure Unicode output works on Windows
    import sys, io
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="Evaluate Hampi Monument Identifier accuracy"
    )
    parser.add_argument(
        "--mode",
        choices=ALL_MODES + ["all"],
        default="all",
        help="Which mode to evaluate (default: all)",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=0.7,
        help="Ensemble weight for Hybrid mode (default: 0.7)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Re-train Linear Probe before evaluation",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        metavar="PATH",
        help="Save evaluation results as JSON to this path",
    )
    args = parser.parse_args()

    print(SEP)
    print("  🏛️  Hampi Monument Identifier — Evaluation Suite  |  T12.5")
    print(SEP)

    # ── 1. Collect test images ────────────────────────────────────────────
    if not _TEST_DIR.exists():
        print(f"❌ Test directory not found: {_TEST_DIR}")
        sys.exit(1)

    samples = collect_test_images(_TEST_DIR)
    print(f"  📁 Test images found: {len(samples)}")

    class_counts = defaultdict(int)
    for _, cls in samples:
        class_counts[cls] += 1
    for cls in MONUMENT_NAMES:
        cnt = class_counts.get(cls, 0)
        print(f"     {cls:<40} {cnt:>3} images")
    print()

    # ── 2. Load CLIP model ────────────────────────────────────────────────
    print("  🔄 Loading CLIP model (openai/clip-vit-base-patch32)…")
    t_load = time.time()
    model = HampiCLIPModel()
    prompts_path = str(_ROOT / "data" / "prompts.json")
    model.load_with_prompts(prompts_path if os.path.exists(prompts_path) else None)
    print(f"  ✅ CLIP loaded in {time.time()-t_load:.1f}s\n")

    # ── 3. Train / load probe ─────────────────────────────────────────────
    probe_available = False

    if args.train:
        print("  🏋️  Training Linear Probe…")
        probe = model.train_and_save_probe(verbose=True)
        probe_available = True
        print(f"  ✅ Probe trained on classes: {probe.trained_classes}\n")
    elif LinearProbeClassifier.exists(_PROBE_PATH):
        print("  📦 Loading existing Linear Probe…")
        probe_available = model.load_probe()
        if probe_available:
            print(f"  ✅ Probe loaded from {_PROBE_PATH}\n")
        else:
            print("  ⚠️  Probe file corrupt — run with --train to rebuild\n")
    else:
        print(
            "  ⚠️  No trained probe found.\n"
            "     Linear Probe / Hybrid modes will fall back to Zero-Shot.\n"
            "     Run with --train to train the probe first.\n"
        )

    # ── 4. Determine modes to evaluate ───────────────────────────────────
    modes_to_eval: list[tuple[str, float]] = []

    if args.mode == "all":
        modes_to_eval = [
            (MODE_ZERO_SHOT,    args.weight),
            (MODE_LINEAR_PROBE, args.weight),
            (MODE_HYBRID,       args.weight),
        ]
    else:
        modes_to_eval = [(args.mode, args.weight)]

    # ── 5. Evaluate each mode ─────────────────────────────────────────────
    all_results = []

    for mode, weight in modes_to_eval:
        label = _mode_label(mode, weight if mode == MODE_HYBRID else None)
        print(SEP)
        print(f"  🔍 Evaluating: {label}")
        print(SEP)

        t0 = time.time()
        results = evaluate_mode(model, samples, mode, weight, verbose=True)
        results = _recompute_avg_conf(results)
        elapsed = time.time() - t0

        print(f"  ⏱️  Completed in {elapsed:.1f}s")
        print_per_class_table(results)
        all_results.append(results)

    # ── 6. Summary comparison ─────────────────────────────────────────────
    if len(all_results) > 1:
        print_summary_comparison(all_results)

    # ── 7. Save results ───────────────────────────────────────────────────
    if args.save_results:
        out_path = Path(args.save_results)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  💾 Results saved to {out_path}\n")

    print("  Done. ✅")
    print(SEP)


if __name__ == "__main__":
    main()
