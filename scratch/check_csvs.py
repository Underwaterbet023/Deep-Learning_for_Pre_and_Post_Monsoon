import pandas as pd

files = [
    "post-monsoon_wl_1994-2023.csv",
    "pre-monsoon_2014-2024.csv",
    "pre-monsoon_2004-2013.csv",
    "pre-monsoon_1994-2003.csv"
]

for f in files:
    df = pd.read_csv(f, nrows=5)
    print(f"=== {f} ===")
    print("Columns:", list(df.columns))
    print("Shape:", df.shape)
