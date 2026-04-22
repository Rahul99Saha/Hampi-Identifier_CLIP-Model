#!/usr/bin/env python3
"""
Example: Zero-Shot Classification with CLIP on Hampi Dataset

This script demonstrates how to use the Hampi dataset for zero-shot
image classification using OpenAI's CLIP model.

Requirements:
    pip install torch torchvision transformers pillow
"""

import json
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict

def load_dataset():
    """Load the Hampi dataset metadata"""
    base_path = Path(__file__).parent
    
    with open(base_path / "classes.json", "r") as f:
        classes = json.load(f)
    
    with open(base_path / "prompts.json", "r") as f:
        prompts = json.load(f)
    
    with open(base_path / "manifest.jsonl", "r") as f:
        manifest = [json.loads(line) for line in f]
    
    return classes, prompts, manifest

def classify_image(image_path, model, processor, text_inputs, classes):
    """Classify a single image using CLIP"""
    image = Image.open(image_path).convert("RGB")
    
    # Prepare inputs
    inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    # Average probabilities across prompts for each class (5 prompts per class)
    class_probs = []
    num_prompts_per_class = len(text_inputs) // len(classes)
    
    for i in range(len(classes)):
        start_idx = i * num_prompts_per_class
        end_idx = start_idx + num_prompts_per_class
        avg_prob = probs[0, start_idx:end_idx].mean().item()
        class_probs.append(avg_prob)
    
    # Get predicted class
    predicted_idx = class_probs.index(max(class_probs))
    predicted_class = classes[predicted_idx]
    confidence = class_probs[predicted_idx]
    
    return predicted_class, confidence, class_probs

def evaluate_dataset():
    """Evaluate CLIP zero-shot performance on the entire Hampi dataset"""
    print("Loading Hampi Zero-Shot Dataset...")
    classes, prompts, manifest = load_dataset()
    
    print(f"Dataset loaded: {len(classes)} classes, {len(manifest)} images")
    print(f"Classes: {', '.join(classes)}\n")
    
    # Initialize CLIP model
    print("Loading CLIP model (openai/clip-vit-base-patch32)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Prepare text inputs from prompts
    text_inputs = []
    for class_name in classes:
        text_inputs.extend(prompts[class_name])
    
    print(f"Using {len(text_inputs)} text prompts ({len(text_inputs)//len(classes)} per class)\n")
    
    # Evaluate
    print("Evaluating...")
    correct = 0
    total = 0
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for item in manifest:
        image_path = Path(item["local_path"])
        true_class = item["class_name"]
        
        if not image_path.exists():
            # Try relative path
            image_path = Path(__file__).parent / item["relative_path"]
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        predicted_class, confidence, _ = classify_image(
            image_path, model, processor, text_inputs, classes
        )
        
        confusion_matrix[true_class][predicted_class] += 1
        
        if predicted_class == true_class:
            correct += 1
        
        total += 1
        
        if total % 10 == 0:
            print(f"  Processed {total}/{len(manifest)} images... "
                  f"Current accuracy: {correct/total:.1%}")
    
    # Results
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Zero-shot Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"\nPer-class Accuracy:")
    
    for class_name in classes:
        class_total = sum(confusion_matrix[class_name].values())
        class_correct = confusion_matrix[class_name][class_name]
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {class_name:30s}: {class_acc:.1%} ({class_correct}/{class_total})")
    
    print(f"\n{'='*60}")

def demo_single_image():
    """Demo: Classify a single image"""
    print("Demo: Single Image Classification\n")
    
    classes, prompts, manifest = load_dataset()
    
    # Load model
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Prepare text inputs
    text_inputs = []
    for class_name in classes:
        text_inputs.extend(prompts[class_name])
    
    # Pick a random image
    sample_item = manifest[0]
    image_path = Path(sample_item["local_path"])
    if not image_path.exists():
        image_path = Path(__file__).parent / sample_item["relative_path"]
    
    true_class = sample_item["class_name"]
    
    print(f"Classifying: {image_path.name}")
    print(f"True class: {true_class}\n")
    
    predicted_class, confidence, class_probs = classify_image(
        image_path, model, processor, text_inputs, classes
    )
    
    print(f"Predicted: {predicted_class}")
    print(f"Confidence: {confidence:.2%}\n")
    
    print("All class probabilities:")
    sorted_results = sorted(zip(classes, class_probs), key=lambda x: x[1], reverse=True)
    for class_name, prob in sorted_results:
        bar = '█' * int(prob * 50)
        print(f"  {class_name:30s}: {prob:.2%} {bar}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_single_image()
    else:
        evaluate_dataset()
