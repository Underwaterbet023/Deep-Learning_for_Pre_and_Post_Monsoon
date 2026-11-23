import json

nb = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb", encoding="utf-8"))
cell8 = "".join(nb['cells'][8]['source']).splitlines()

for i, l in enumerate(cell8):
    if "def plot_time_series" in l:
        print(f"Cell 8 Line {i}: {l}")
        for j in range(max(0, i-2), min(len(cell8), i+100)):
            print(f"  {j}: {cell8[j]}")
        print("="*60)
