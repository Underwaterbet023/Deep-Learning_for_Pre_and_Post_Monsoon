import json

nb = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb", encoding="utf-8"))
cell9 = "".join(nb['cells'][9]['source']).splitlines()

for i in range(40, 100):
    if i < len(cell9):
        print(f"{i}: {cell9[i]}")
