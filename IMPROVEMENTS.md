# Hampi Monument Identifier - Performance Improvement Guide

**Status**: ✅ **Accuracy improved to 58.3% (+4.2%)** using enhanced prompts

## Current Performance

| Metric | Baseline | Improved | Gain |
|--------|----------|----------|------|
| **Top-1 Accuracy** | 54.2% | **58.3%** | +4.2% |
| **Top-3 Accuracy** | 72.2% | 77.8% | +5.6% |
| **Avg Confidence** | 53.9% | 59.6% | +5.7% |
| **Speed (latency)** | 110ms | 110ms | No change |

## Per-Class Accuracy (Enhanced Model)

| Monument | Accuracy | Status | Notes |
|----------|----------|--------|-------|
| Lotus Mahal | 91.7% | ✅ Excellent | Good distinguishing features (arches, pavilion) |
| Hemakuta temple hill complex | 91.7% | ✅ Excellent | Hillside location helps discrimination |
| Monolithic Bull | 75.0% | ✅ Good | Distinctive statue, some confusion with temples |
| Elephant Stables | 66.7% | ⚠️ Fair | Confused with Lotus Mahal, Zenana Enclosure |
| Hampi Bazaar | 50.0% | ⚠️ Fair | Often confused with Hemakuta Hill temples |
| Royal Centre | 25.0% | ❌ Poor | Confused with Zenana Enclosure |

## What Was Improved

### 1. **Enhanced Prompts (V2)** ✅ +4.2%
- Increased from 5 to 10 prompts per class
- More descriptive architectural details
- Added diverse descriptions (tourist photo, architectural photo, heritage photo, etc.)
- Included architectural terminology (gopuram, mandapas, arches, domes, etc.)
- Better location and context information

**Example improvements**:
```
OLD: "a photo of Lotus Mahal pavilion in Hampi"
NEW: "the ornamental lotus palace with distinctive arched stone balconies"
      + "small elegant palace with lotus motifs and symmetrical architecture"
      + "garden pavilion structure with intricate stone latticework..."
```

### 2. **Model Variants Tested**

#### Base Model (ViT-B/32) - RECOMMENDED ✅
- Top-1 Accuracy: 58.3% (with enhanced prompts)
- Speed: 110ms per image
- Model size: 63M parameters
- Best accuracy/speed trade-off

#### Large Model (ViT-L/14) ⚠️
- Top-1 Accuracy: 55.6% (lower with same prompts!)
- Speed: 792ms per image (7.2x slower)
- Model size: 304M parameters
- Higher confidence (70.5%) but slower and lower accuracy
- **Not recommended** for this task

### 3. **Distinguished Prompts Approach** ❌ Failed
Tested adding negative descriptors ("NOT marketplace", "NOT pavilion") but this:
- Reduced accuracy to 43.1% (-15.3%)
- CLIP models don't handle negation well
- **Lesson**: Stick to positive descriptions, avoid negatives

## Remaining Challenges

### 1. **Hampi Bazaar (50%)** - Confused with Hemakuta temples
**Problem**: Both are linear structures with pillars/steps
**Tried**: 
- Adding "marketplace" in prompts
- Emphasizing "shopping street" vs "temple hill"
**Still needed**:
- Image preprocessing (brightness/contrast enhancement)
- Fine-tuning on marketplace-specific features
- Ensemble approach with preprocessing

### 2. **Royal Centre (25%)** - Confused with Zenana Enclosure
**Problem**: Both are fortified structures with platforms
**Tried**:
- "raised platform" vs "fortified walls"
- "administrative center" vs "women's quarters"
**Still needed**:
- More training data specific to Royal Centre
- Fine-tuning with labeled Royal Centre images
- Specialized image preprocessing

### 3. **Elephant Stables (66.7%)** - Confused with Lotus Mahal
**Problem**: Both have arches and domed structures
**Tried**:
- Emphasizing "domed chambers for animals" vs "ornate pavilion"
- "utilitarian structure" vs "decorative palace"
**Still needed**:
- Better texture-based descriptors
- Fine-tuning on animal structure features

## Advanced Improvement Strategies (Not Yet Tried)

### 1. **Fine-tuning CLIP** 
```python
# Would require ~100 labeled images per class
from transformers import CLIPModel, CLIPProcessor
# Use LoRA or full fine-tuning with labeled Hampi dataset
# Expected improvement: +5-10%
```

### 2. **Image Preprocessing**
```python
from PIL import ImageEnhance

def enhance_image(image):
    # Increase contrast for outdoor photos
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Adjust brightness if too dark
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)
    
    return image
```

### 3. **Multi-Crop Evaluation**
```python
# Run CLIP on 5 crops: center + 4 corners
# Average predictions for more robust classification
# Expected improvement: +2-3%
```

### 4. **Ensemble Multiple Models**
```python
# Combine predictions from:
# - Base model + Enhanced prompts
# - Large model + Enhanced prompts
# - Fine-tuned model (if available)
# Voting or weighted ensemble
# Expected improvement: +3-5%
```

### 5. **Confidence Thresholding**
```python
# If confidence < 60%, request user feedback
# Fall back to top-3 suggestions
# Reduces false positives significantly
```

### 6. **Custom Text Embeddings**
```python
# Use domain expert descriptions for prompts
# Archaeological/historical terminology
# Vijayanagara architectural features
# Expected improvement: +2-3%
```

## Implementation Recommendations

### Short-term (Easy, High Impact)
1. ✅ **DONE**: Enhanced prompts → 58.3% accuracy
2. 📋 **TODO**: Deploy enhanced model to Streamlit app
3. 📋 **TODO**: Add confidence threshold UI (suggest alternatives if <60%)
4. 📋 **TODO**: Image preprocessing (contrast enhancement)

### Medium-term (Moderate, Medium Impact)
1. Collect user feedback for misclassified images
2. Analyze failure patterns (already done in evaluation)
3. Create fine-tuning dataset from user corrections
4. Test multi-crop evaluation

### Long-term (Complex, Higher Impact)
1. Fine-tune CLIP on Hampi dataset (if 100+ labeled images available)
2. Custom prompt generation from architectural descriptions
3. Ensemble approach with multiple models
4. Deploy as production API with versioning

## How to Use Different Prompt Versions

```python
from model.clip_model import HampiCLIPModel

# Option 1: Enhanced prompts (RECOMMENDED)
model = HampiCLIPModel()
model.load_with_prompts("data/prompts.json")  # Enhanced prompts

# Option 2: Original prompts (fallback)
model = HampiCLIPModel()
model.load()  # Uses hardcoded defaults

# Option 3: Custom prompts
model = HampiCLIPModel()
model.load_with_prompts("data/custom_prompts.json")
```

## Files Structure

```
data/
├── prompts.json                    # ✅ Enhanced prompts (ACTIVE)
├── prompts_v2_enhanced.json        # Version 2 with 10 prompts/class
└── prompts_v3_distinguished.json   # Version 3 (not recommended)

notebooks/
└── evaluation.ipynb                # Full evaluation with comparisons

model/
└── clip_model.py                   # Updated with variants and prompt loading
```

## Testing and Validation

Run the evaluation notebook to test improvements:

```bash
jupyter notebook notebooks/evaluation.ipynb
```

Cells to run:
1. **Baseline test**: Original model with default prompts
2. **Enhanced prompts test**: Base model + V2 prompts → 58.3%
3. **Large model test**: ViT-L/14 model
4. **Failure analysis**: Confusion matrix and per-class breakdown
5. **Comparison**: Side-by-side accuracy metrics

## Key Learnings

1. ✅ **Prompt engineering matters**: +4.2% improvement from better prompts alone
2. ❌ **Larger models don't always win**: ViT-L/14 was actually slower and less accurate
3. ❌ **Negative descriptors hurt CLIP**: Avoid "NOT X" in prompts
4. ✅ **Ensemble prompts work**: Multiple descriptions for same class → better robustness
5. ✅ **Architecture keywords help**: Gopuram, mandapas, arches are key features

## Next Steps

1. **Deploy**: Update app.py to use enhanced prompts by default ✅
2. **Test**: Run Streamlit app with new prompts
3. **Gather feedback**: Collect user corrections for hard cases
4. **Iterate**: Use feedback to improve prompts further
5. **Fine-tune** (if sufficient data): Create labeled dataset for fine-tuning

---

**Last Updated**: April 2026  
**Current Best Configuration**: Base model (ViT-B/32) + Enhanced prompts V2  
**Baseline Accuracy**: 54.2% → **58.3%** (+4.2%)
