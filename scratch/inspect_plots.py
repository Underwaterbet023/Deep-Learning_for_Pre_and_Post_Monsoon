import json

nb = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb", encoding="utf-8"))
cell8 = "".join(nb['cells'][8]['source']).splitlines()

for i, l in enumerate(cell8):
    if "def plot_residuals" in l or "def plot_observed_timeline" in l:
        print(f"--- Line {i}: {l} ---")
        for j in range(i, min(len(cell8), i+45)):
            print(f"  {j}: {cell8[j]}")
