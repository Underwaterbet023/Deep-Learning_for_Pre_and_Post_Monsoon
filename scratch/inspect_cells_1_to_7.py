import json

nb = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb", encoding="utf-8"))

for i in [1, 2, 4, 5, 6, 7]:
    print(f"=== CELL {i} ===")
    print("".join(nb['cells'][i]['source'])[:500])
