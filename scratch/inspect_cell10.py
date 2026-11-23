import json

nb = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb", encoding="utf-8"))

print("=== CELL 10 CODE ===")
print("".join(nb['cells'][10]['source']))

print("=== CELL 10 ERROR FULL ===")
for out in nb['cells'][10].get('outputs', []):
    if out.get('output_type') == 'error':
        print("\n".join(out.get('traceback', [])))
