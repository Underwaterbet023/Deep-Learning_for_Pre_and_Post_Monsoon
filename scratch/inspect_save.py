import json

nb = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb", encoding="utf-8"))
cell8 = "".join(nb['cells'][8]['source']).splitlines()

for i, l in enumerate(cell8):
    if "def save_and_show" in l or "save_and_show" in l:
        print(f"Line {i}: {l}")
        for j in range(max(0, i-2), min(len(cell8), i+15)):
            print(f"  {j}: {cell8[j]}")
        print("---")
