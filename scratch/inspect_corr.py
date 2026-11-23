import json

nb = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb", encoding="utf-8"))

cell8 = "".join(nb['cells'][8]['source']).splitlines()
print(f"Cell 8 total lines: {len(cell8)}")

for i, l in enumerate(cell8):
    if "def plot_correlation_matrix" in l:
        print(f"Line {i}: {l}")
        for j in range(max(0, i-2), min(len(cell8), i+30)):
            print(f"  {j}: {cell8[j]}")

cell9 = "".join(nb['cells'][9]['source']).splitlines()
print(f"\nCell 9 total lines: {len(cell9)}")
for i, l in enumerate(cell9):
    if "plot_correlation_matrix" in l:
        print(f"Cell 9 Line {i}: {l}")
        for j in range(max(0, i-5), min(len(cell9), i+15)):
            print(f"  {j}: {cell9[j]}")
