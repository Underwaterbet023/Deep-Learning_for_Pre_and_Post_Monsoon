import json
import ast

nb = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb", encoding="utf-8"))
cell9 = "".join(nb['cells'][9]['source'])

tree = ast.parse(cell9)
calls = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)

print("Calls in Cell 9:")
for c in sorted(calls):
    print(" ", c)
