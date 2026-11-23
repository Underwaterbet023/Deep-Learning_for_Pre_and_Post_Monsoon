import json

nb = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb", encoding="utf-8"))
cell8 = "".join(nb['cells'][8]['source']).splitlines()
for i in range(915, len(cell8)):
    print(f"{i}: {cell8[i]}")
