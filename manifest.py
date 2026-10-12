import os
import re
import pandas as pd

root_dir = "SAMPLE 1 TO 8"
rows = []

for dirpath, _, filenames in os.walk(root_dir):
    sample_match = re.search(r"SAMPLE\s+(\d+)(?!\s+TO)", dirpath)
    if not sample_match:
        continue
    sample_id = int(sample_match.group(1))

    for fname in filenames:
        if not fname.lower().endswith(".csv"):
            continue
        layer_match = re.search(
            r"Layer_(\d+)(?:_(?:and|to)_(\d+))?",
            fname,
            re.IGNORECASE
        )
        if not layer_match:
            print("SKIPPED:", dirpath, fname)
            continue

        layer1 = int(layer_match.group(1))
        layer2 = int(layer_match.group(2)) if layer_match.group(2) else layer1

        # "25_and_27" / "29_to_32" hold every layer in between
        layers = ",".join(str(l) for l in range(layer1, layer2 + 1))

        rows.append({
            "filepath": os.path.join(dirpath, fname),
            "sample_id": sample_id,
            "layers_in_file": layers
        })

manifest = pd.DataFrame(
    rows,
    columns=["filepath", "sample_id", "layers_in_file"]
)
if not manifest.empty:
    manifest = manifest.sort_values(
        ["sample_id", "filepath"]
    )
    manifest.to_csv("manifest.csv", index=False)

    print("Manifest created successfully!")
    print("Shape:", manifest.shape)
    print(manifest.head())
else:
    print("No CSV files were found.")

with open('manifest.csv', 'rb') as f:
    print(f.readline())