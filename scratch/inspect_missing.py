import json

nb_bak = json.load(open("HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb.bak", encoding="utf-8"))
cell8_bak = "".join(nb_bak['cells'][8]['source'])

for name in ["def make_taylor_dataframe", "def regional_daily_summary", "def aggregate_predictions", "def save_figure"]:
    idx = cell8_bak.find(name)
    if idx != -1:
        print("="*60)
        print(f"=== {name} ===")
        # find end of function (next def at col 0)
        next_def = cell8_bak.find("\ndef ", idx + 1)
        if next_def == -1:
            next_def = idx + 1000
        print(cell8_bak[idx:next_def])
