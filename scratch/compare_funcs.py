import json

nb_bak = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb.bak", encoding="utf-8"))
cell8_bak = "".join(nb_bak['cells'][8]['source'])

funcs_bak = [line.strip() for line in cell8_bak.splitlines() if line.strip().startswith("def ")]
print("Functions in BAK cell 8:")
for f in funcs_bak:
    print(" ", f)

nb_cur = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb", encoding="utf-8"))
cell8_cur = "".join(nb_cur['cells'][8]['source'])
funcs_cur = [line.strip() for line in cell8_cur.splitlines() if line.strip().startswith("def ")]
print("\nFunctions in CURRENT cell 8:")
for f in funcs_cur:
    print(" ", f)

print("\nMissing in current:")
for f in funcs_bak:
    name = f.split("(")[0]
    if not any(name in fc for fc in funcs_cur):
        print("  MISSING:", f)
