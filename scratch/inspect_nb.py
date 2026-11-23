import json
import sys

nb = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb", encoding="utf-8"))

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    cell_type = c['cell_type']
    source = "".join(c['source'])
    outputs = c.get('outputs', [])
    err_count = 0
    for out in outputs:
        if out.get('output_type') == 'error':
            err_count += 1
            print(f"!!! CELL {i} HAS ERROR: {out.get('ename')}: {out.get('evalue')}")
            for tb in out.get('traceback', [])[-3:]:
                print("   ", tb)
    print(f"Cell {i} ({cell_type}, {len(source.splitlines())} lines, {len(outputs)} outputs, {err_count} errors): {source.strip().splitlines()[0] if source.strip() else 'EMPTY'}")
