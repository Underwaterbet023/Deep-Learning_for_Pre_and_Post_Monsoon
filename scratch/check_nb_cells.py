import json

nb = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb", encoding="utf-8"))

print(f"Total cells: {len(nb['cells'])}")
print("Cell 8 lines:", len(nb['cells'][8]['source']))
print("Cell 9 lines:", len(nb['cells'][9]['source']))
print("Cell 10 lines:", len(nb['cells'][10]['source']))
print("Cell 12 lines:", len(nb['cells'][12]['source']))
