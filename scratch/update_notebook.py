import json
import os

notebook_path = 'notebooks/evaluation.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Cell 3 fix (Single image prediction)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'IMAGE_PATH = \'data/test_images/Virupaksha Temple\'' in ''.join(cell['source']):
        new_source = []
        for line in cell['source']:
            line = line.replace("IMAGE_PATH = 'data/test_images/Virupaksha Temple'", "IMAGE_PATH = '../data/test_images/virupaksha.jpg'")
            line = line.replace("— add a test image first.", "— please verify the path.")
            new_source.append(line)
        cell['source'] = new_source

# Cell 4 fix (Batch evaluation)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'TEST_DIR = \'../data/test_images\'' in ''.join(cell['source']):
        new_source = []
        for line in cell['source']:
            if "img = Image.open(os.path.join(folder, fname)).convert('RGB')" in line:
                # Add try-except
                new_source.append("            try:\n")
                new_source.append("                img = Image.open(os.path.join(folder, fname)).convert('RGB')\n")
                new_source.append("                preds, lat = model.predict(img, top_k=3)\n")
                new_source.append("            except Exception as e:\n")
                new_source.append(f"                print(f'Skipping corrupted image {{fname}}: {{e}}')\n")
                new_source.append("                continue\n")
            elif "preds, lat = model.predict(img, top_k=3)" in line:
                # Already handled in the try-except block above
                continue
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
