# Hampi Zero-Shot CLIP Dataset

A curated dataset of historical monuments in Hampi, Karnataka, India for zero-shot image classification using CLIP models.

## Dataset Overview

This dataset contains **120 images** across **10 distinct monument classes** from the UNESCO World Heritage Site of Hampi. Each class has exactly 12 images.

### Monument Classes

1. **Lotus Mahal** - A beautiful two-story palace with Indo-Islamic architecture
2. **Virupaksha Temple** - Ancient 7th-century temple dedicated to Lord Shiva
3. **Vittala Temple** - Famous for its stone chariot and musical pillars
4. **Elephant Stables** - Historic stables that housed royal elephants
5. **Hampi Bazaar** - Ancient marketplace street
6. **Zenana Enclosure** - Fortified area reserved for royal women, includes watch towers
7. **Royal Centre** - Area with courtly and military structures
8. **Queen's Bath** - Ornate royal bathing complex
9. **Hemakuta Temple Hill Complex** - Temple complex on Hemakuta Hill
10. **Monolithic Bull** - Large stone Nandi (bull) sculpture

## Directory Structure

```
hampi_zero_shot/
├── README.md                    # This file
├── classes.json                 # List of 10 class names
├── prompts.json                 # Text prompts for each class (5 prompts per class)
├── metadata.json                # Wikipedia summaries for each monument
├── manifest.jsonl               # Complete image metadata (one JSON object per line)
└── images/                      # Image directory
    ├── Lotus_Mahal/            # 12 images
    ├── Virupaksha_Temple/      # 12 images
    ├── Vittala_Temple/         # 12 images
    ├── Elephant_Stables/       # 12 images
    ├── Hampi_Bazaar/           # 12 images
    ├── Zenana_Enclosure/       # 12 images
    ├── Royal_Centre/           # 12 images
    ├── Queen_s_Bath/           # 12 images
    ├── Hemakuta_temple_hill_complex/  # 12 images
    └── Monolithic_Bull/        # 12 images
```

## File Descriptions

### classes.json
A simple JSON array containing the 10 class names:
```json
[
  "Lotus Mahal",
  "Virupaksha Temple",
  ...
]
```

### prompts.json
Contains 5 text prompts for each class to use with CLIP:
```json
{
  "Lotus Mahal": [
    "a photo of Lotus Mahal",
    "a clear daylight photo of Lotus Mahal",
    "a tourist photo of Lotus Mahal in Hampi",
    "an architectural photo of Lotus Mahal",
    "a heritage monument photo of Lotus Mahal"
  ],
  ...
}
```

### metadata.json
Wikipedia summary for each monument:
```json
{
  "Lotus Mahal": {
    "wiki_title": "Lotus Mahal",
    "summary": "The Lotus Mahal is a palace in the Zenana Enclosure..."
  },
  ...
}
```

### manifest.jsonl
Complete image metadata, one JSON object per line:
```json
{"class_name": "Lotus Mahal", "file_title": "000_...", "local_path": "...", "relative_path": "images/Lotus_Mahal/000_..."}
...
```

## Usage with CLIP

### Zero-Shot Classification Example

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load class names
with open("classes.json", "r") as f:
    classes = json.load(f)

# Load prompts
with open("prompts.json", "r") as f:
    prompts = json.load(f)

# Create text inputs from prompts
text_inputs = []
for class_name in classes:
    text_inputs.extend(prompts[class_name])

# Load and classify an image
image = Image.open("images/Lotus_Mahal/000_lotus_mahal.jpg")

# Prepare inputs
inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

# Average probabilities across prompts for each class
class_probs = []
for i in range(len(classes)):
    start_idx = i * 5  # 5 prompts per class
    end_idx = start_idx + 5
    avg_prob = probs[0, start_idx:end_idx].mean().item()
    class_probs.append(avg_prob)

# Get predicted class
predicted_class = classes[class_probs.index(max(class_probs))]
print(f"Predicted: {predicted_class}")
print(f"Confidence: {max(class_probs):.2%}")
```

### Batch Evaluation Example

```python
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Load dataset
with open("classes.json", "r") as f:
    classes = json.load(f)

with open("prompts.json", "r") as f:
    prompts = json.load(f)

with open("manifest.jsonl", "r") as f:
    manifest = [json.loads(line) for line in f]

# Initialize model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare text inputs
text_inputs = []
for class_name in classes:
    text_inputs.extend(prompts[class_name])

# Evaluate all images
correct = 0
total = 0

for item in manifest:
    image_path = item["local_path"]
    true_class = item["class_name"]
    
    # Load and classify
    image = Image.open(image_path)
    inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    # Average probabilities across prompts
    class_probs = []
    for i in range(len(classes)):
        start_idx = i * 5
        end_idx = start_idx + 5
        avg_prob = probs[0, start_idx:end_idx].mean().item()
        class_probs.append(avg_prob)
    
    predicted_class = classes[class_probs.index(max(class_probs))]
    
    if predicted_class == true_class:
        correct += 1
    total += 1
    
    if total % 10 == 0:
        print(f"Processed {total}/{len(manifest)} images...")

accuracy = correct / total
print(f"\nZero-shot Accuracy: {accuracy:.2%} ({correct}/{total})")
```

## Dataset Statistics

- **Total Images**: 120
- **Number of Classes**: 10
- **Images per Class**: 12
- **Image Format**: JPEG (640px thumbnails from Wikimedia Commons)
- **Text Prompts per Class**: 5
- **License**: All images are from Wikimedia Commons (CC-BY-SA and similar licenses)

## Image Sources

All images are sourced from Wikimedia Commons, primarily from the following categories:
- Category:Lotus Mahal (Hampi)
- Category:Virupaksha_Temple
- Category:Vittala_Temple
- Category:Elephant stables at Hampi
- Category:Hampi Bazaar
- Category:Zanana Enclosure
- Category:Royal Centre (Hampi)
- Category:Queen's_Bath
- Category:Hemakuta temple hill complex
- Category:Monolithic Bull (Hampi)

## Notes for CLIP Zero-Shot Classification

1. **Use Multiple Prompts**: The dataset includes 5 different prompts per class. For best results, average the probabilities across all prompts for each class.

2. **Ensemble Predictions**: You can experiment with different prompt templates or add more prompts to improve accuracy.

3. **Class Similarity**: Some monuments (like Lotus Mahal and Zenana Enclosure) are physically close and share architectural styles, which may affect classification accuracy.

4. **Image Quality**: All images are 640px thumbnails from Wikimedia Commons, suitable for CLIP's default input size.

## Citation

If you use this dataset, please cite:
```
Hampi Zero-Shot CLIP Dataset
UNESCO World Heritage Site monuments from Hampi, Karnataka, India
Images sourced from Wikimedia Commons
2026
```

## License

This dataset compilation is provided for educational and research purposes. Individual images retain their original Wikimedia Commons licenses (primarily CC-BY-SA). Please refer to the manifest.jsonl file for specific image attribution where available.
