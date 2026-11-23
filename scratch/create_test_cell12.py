import json

nb = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb", encoding="utf-8"))

cell1 = "".join(nb['cells'][1]['source'])
cell2 = "".join(nb['cells'][2]['source'])
cell8 = "".join(nb['cells'][8]['source'])
cell12 = "".join(nb['cells'][12]['source'])

test_code = cell1 + "\n" + cell2 + "\n" + cell8 + "\n" + cell12

with open("scratch/run_test_cell12.py", "w", encoding="utf-8") as f:
    f.write(test_code)

print("Created scratch/run_test_cell12.py")
