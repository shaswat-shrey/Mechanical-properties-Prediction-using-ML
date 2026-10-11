import pandas as pd
import os
PATH = "/Users/shaswatshrey/Development/MajorProject/SAMPLE 1 TO 8/SAMPLE 1"
OUTPUT = "/Users/shaswatshrey/Development/MajorProject/peak.csv"

result = []
for root, dir, files in os.walk(PATH):
    print(dir, files, root)
    for file in files:
        if not file.lower().endswith(".csv"):
            continue
        file_path = os.path.join(root, file)

        df = pd.read_csv(file_path, sep=',', encoding='latin1')

        df.columns = df.columns.str.strip()

        energy = pd.to_numeric(df["ABS-ENERGY"], errors="coerce")

        result.append({
            "Layers": file,
            "Maximum": round(energy.max(), 2),
            "95th_Percentile": round(energy.quantile(0.95), 2),
            "Mean": round(energy.mean(), 2),
            "Median": round(energy.median(), 2)
        })

result_df = pd.DataFrame(result)
result_df.to_csv(OUTPUT, index=False)

print(result_df)
print(f"\nSaved to: {OUTPUT}")