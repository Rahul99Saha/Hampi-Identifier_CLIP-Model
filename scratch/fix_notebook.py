"""
Fix the evaluation notebook to use FOLDER_TO_CLASS mapping.

The 0% accuracy bug is caused by comparing folder names (e.g. "Elephant_Stables")
with class labels (e.g. "Elephant Stables"). This script patches the notebook
to import and use the FOLDER_TO_CLASS mapping.
"""

import json
import os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "evaluation.ipynb")

with open(NOTEBOOK_PATH, "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue

    source = "".join(cell["source"])

    # Fix Cell 1: Add FOLDER_TO_CLASS to imports
    if "from model.clip_model import HampiCLIPModel, MONUMENT_NAMES, MONUMENT_PROMPTS" in source:
        new_source = source.replace(
            "from model.clip_model import HampiCLIPModel, MONUMENT_NAMES, MONUMENT_PROMPTS",
            "from model.clip_model import HampiCLIPModel, MONUMENT_NAMES, MONUMENT_PROMPTS, FOLDER_TO_CLASS"
        )
        cell["source"] = new_source.split("\n")
        cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
        print(f"Patched cell {i}: added FOLDER_TO_CLASS import")

    # Fix Cell 7: Use FOLDER_TO_CLASS to convert folder names to class labels
    if "top1_correct = preds[0]['name'] == monument_name" in source:
        # Replace the comparison logic to use the mapping
        new_source = source.replace(
            "for monument_name in os.listdir(TEST_DIR):",
            "for folder_name in os.listdir(TEST_DIR):"
        ).replace(
            "folder = os.path.join(TEST_DIR, monument_name)",
            "folder = os.path.join(TEST_DIR, folder_name)"
        ).replace(
            "top1_correct = preds[0]['name'] == monument_name",
            "# Map folder name to CLIP class label (e.g. 'Elephant_Stables' -> 'Elephant Stables')\n"
            "            class_name = FOLDER_TO_CLASS.get(folder_name, folder_name)\n"
            "            top1_correct = preds[0]['name'] == class_name"
        ).replace(
            "top3_correct = any(p['name'] == monument_name for p in preds)",
            "top3_correct = any(p['name'] == class_name for p in preds)"
        ).replace(
            "'true': monument_name,",
            "'true': class_name,"
        )
        cell["source"] = new_source.split("\n")
        cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
        # Clear old outputs
        cell["outputs"] = []
        print(f"Patched cell {i}: fixed folder name -> class name comparison")

with open(NOTEBOOK_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("\nNotebook patched successfully!")
