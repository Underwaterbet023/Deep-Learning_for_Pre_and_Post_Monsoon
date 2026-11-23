import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

SHOW_FIGURES = False

MODEL_COLORS = {
    "Observed": "#000000",       # BLACK
    "LSTM": "#e74c3c",           # RED
    "GRU": "#27ae60",            # GREEN
    "CNN-LSTM": "#00cec9",       # CYAN
    "Hybrid LSTM-GRU": "#0984e3",# BLUE
}

DATASET_FILES = {
    "post_monsoon_1994_2023": "post-monsoon_wl_1994-2023.csv",
    "pre_monsoon_2014_2024": "pre-monsoon_2014-2024.csv",
    "pre_monsoon_2004_2013": "pre-monsoon_2004-2013.csv",
    "pre_monsoon_1994_2003": "pre-monsoon_1994-2003.csv",
}

DATASET_TITLES = {
    "post_monsoon_1994_2023": "Post-Monsoon Season (1994–2023)",
    "pre_monsoon_2014_2024": "Pre-Monsoon Season (2014–2024)",
    "pre_monsoon_2004_2013": "Pre-Monsoon Season (2004–2013)",
    "pre_monsoon_1994_2003": "Pre-Monsoon Season (1994–2003)",
}

OUTPUT_ROOT = Path("outputs") / "HIGDSS_DL_4_SEPARATE_DATASETS"

def is_notebook():
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except Exception:
        pass
    return False

def save_and_show_figure(fig, path, dpi=600):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if SHOW_FIGURES and is_notebook():
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            pass
    plt.close(fig)

def save_and_show_plotly(fig, path, width=1200, height=520):
    html_path = path.with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    if SHOW_FIGURES and is_notebook():
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            pass

def plot_time_series_publication(pred_df, split_date, out_path, dataset_title):
    models = [m for m in ["LSTM", "GRU", "CNN-LSTM", "Hybrid LSTM-GRU"] if m in pred_df.columns]
    df = pred_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # 1. Real data aggregated by Date (regional mean only for visualization)
    daily = df.groupby("Date")[["Observed"] + models].mean().reset_index().sort_values("Date")
    if len(daily) == 0:
        return

    # 2. Common standardization for visualization across all available series
    all_vals = np.concatenate([daily[col].dropna().values for col in ["Observed"] + models])
    common_mean = float(np.mean(all_vals))
    common_std = float(np.std(all_vals))
    if common_std == 0 or np.isnan(common_std):
        common_std = 1.0

    daily_scaled = pd.DataFrame({"Date": daily["Date"]})
    daily_scaled["Observed"] = (daily["Observed"] - common_mean) / common_std
    for m in models:
        daily_scaled[m] = (daily[m] - common_mean) / common_std

    # 3. Static Research-paper Style Plot
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=600)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Thin gray frame
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.9)

    ax.grid(False)

    # Horizontal black dashed zero reference line
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=3)

    # Train/Test boundary
    if split_date is not None:
        split_ts = pd.Timestamp(split_date)
        date_span_days = max(1, (daily["Date"].max() - daily["Date"].min()).days)
        half_width = max(pd.Timedelta(days=15), pd.Timedelta(days=int(date_span_days * 0.012)))
        split_start = split_ts - half_width
        split_end = split_ts + half_width

        ax.axvspan(split_start, split_end, color="#f28e8e", alpha=0.55, zorder=2, label="Train/Test Split")
        ax.axvline(split_ts, color="black", linestyle="--", linewidth=1.0, zorder=4)

    # Model lines: LSTM (Red), GRU (Green), CNN-LSTM (Cyan), Hybrid LSTM-GRU (Blue)
    model_color_map = {
        "LSTM": "#e74c3c",           # RED
        "GRU": "#27ae60",            # GREEN
        "CNN-LSTM": "#00cec9",       # CYAN
        "Hybrid LSTM-GRU": "#0984e3",# BLUE
    }

    for m in models:
        ax.plot(
            daily_scaled["Date"], daily_scaled[m],
            color=model_color_map.get(m, "#e67e22"),
            linewidth=1.3, label=m, zorder=6
        )

    # Black observed line on top
    ax.plot(
        daily_scaled["Date"], daily_scaled["Observed"],
        color="#000000", linewidth=1.6, label="Observed", zorder=8
    )

    ax.set_xlabel("Time (Years)", fontsize=9.5, fontweight="bold", color="#222222")
    ax.set_ylabel("Standardized Groundwater Level", fontsize=9.5, fontweight="bold", color="#222222")

    year_span = (daily["Date"].max() - daily["Date"].min()).days / 365.25
    ax.xaxis.set_major_locator(mdates.YearLocator(2 if year_span > 10 else 1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))

    y_min = float(daily_scaled[["Observed"] + models].min().min())
    y_max = float(daily_scaled[["Observed"] + models].max().max())
    y_pad = max(0.2, (y_max - y_min) * 0.08)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.tick_params(direction="out", length=4, width=0.8, colors="#333333", labelsize=8.5)

    ax.legend(
        loc="upper right", frameon=True, facecolor="white",
        edgecolor="#777777", framealpha=0.95, fontsize=8.0, ncol=max(2, len(models) + 1)
    )

    fig.tight_layout()
    save_and_show_figure(fig, out_path, dpi=600)

def plot_time_series_interactive(pred_df, split_date, out_path, dataset_title):
    models = [m for m in ["LSTM", "GRU", "CNN-LSTM", "Hybrid LSTM-GRU"] if m in pred_df.columns]
    df = pred_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    daily = df.groupby("Date")[["Observed"] + models].mean().reset_index().sort_values("Date")
    if len(daily) == 0:
        return

    all_vals = np.concatenate([daily[col].dropna().values for col in ["Observed"] + models])
    common_mean = float(np.mean(all_vals))
    common_std = float(np.std(all_vals))
    if common_std == 0 or np.isnan(common_std):
        common_std = 1.0

    daily_scaled = pd.DataFrame({"Date": daily["Date"]})
    daily_scaled["Observed"] = (daily["Observed"] - common_mean) / common_std
    for m in models:
        daily_scaled[m] = (daily[m] - common_mean) / common_std

    fig = go.Figure()

    fig.add_hline(y=0.0, line_dash="dash", line_color="black", line_width=1.0)

    if split_date is not None:
        split_ts = pd.Timestamp(split_date)
        date_span_days = max(1, (daily["Date"].max() - daily["Date"].min()).days)
        half_width = max(pd.Timedelta(days=15), pd.Timedelta(days=int(date_span_days * 0.012)))
        split_start = split_ts - half_width
        split_end = split_ts + half_width

        fig.add_vrect(
            x0=str(split_start.date()), x1=str(split_end.date()),
            fillcolor="rgba(242, 142, 142, 0.55)", line_width=0,
            annotation_text="Train/Test Split", annotation_position="top left",
            annotation_font=dict(size=10, color="#d63031")
        )
        fig.add_vline(x=str(split_ts.date()), line_dash="dash", line_color="black", line_width=1.0)

    model_color_map = {
        "LSTM": "#e74c3c",           # RED
        "GRU": "#27ae60",            # GREEN
        "CNN-LSTM": "#00cec9",       # CYAN
        "Hybrid LSTM-GRU": "#0984e3",# BLUE
    }

    for m in models:
        fig.add_trace(go.Scatter(
            x=daily_scaled["Date"], y=daily_scaled[m], mode="lines", name=m,
            line=dict(color=model_color_map.get(m, "#e67e22"), width=1.5),
            customdata=daily[m].values,
            hovertemplate=f"<b>{m}</b><br>Standardized: %{{y:.3f}}<br>Physical Level: %{{customdata:.2f}} mbgl<extra></extra>"
        ))

    fig.add_trace(go.Scatter(
        x=daily_scaled["Date"], y=daily_scaled["Observed"], mode="lines", name="Observed",
        line=dict(color="#000000", width=2.0),
        customdata=daily["Observed"].values,
        hovertemplate="<b>Observed</b><br>Standardized: %{y:.3f}<br>Physical Level: %{customdata:.2f} mbgl<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Time Series: Observed vs Deep Learning Predictions — {dataset_title}</b>",
            font=dict(size=14, family="Arial", color="#111111"),
            x=0.5, xanchor="center"
        ),
        xaxis=dict(
            title="<b>Time (Years)</b>",
            showgrid=False,
            linecolor="#777777", linewidth=0.9, mirror=True,
            tickfont=dict(size=10),
            rangeselector=dict(
                buttons=[
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(count=10, label="10Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ],
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#777777",
                borderwidth=0.8,
            ),
            rangeslider=dict(visible=True, thickness=0.08),
            type="date"
        ),
        yaxis=dict(
            title="<b>Standardized Groundwater Level</b>",
            showgrid=False,
            linecolor="#777777", linewidth=0.9, mirror=True,
            tickfont=dict(size=10),
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        width=1200, height=520,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
            bgcolor="rgba(255,255,255,0.95)", bordercolor="#777777", borderwidth=0.8, font=dict(size=10)
        ),
        margin=dict(l=65, r=40, t=80, b=60),
        hovermode="x unified",
    )
    save_and_show_plotly(fig, out_path, width=1200, height=520)

def plot_time_series(df, split_date, out_path, dataset_title):
    plot_time_series_publication(df, split_date, out_path, dataset_title)
    try:
        plot_time_series_interactive(df, split_date, out_path.with_name(out_path.stem + "_interactive"), dataset_title)
    except Exception as e:
        print("Interactive time series error:", e)

if __name__ == "__main__":
    print("Testing exact user specification for time series on all 4 datasets...")
    sys.stdout.flush()

    for key, title in DATASET_TITLES.items():
        dataset_dir = OUTPUT_ROOT / key
        table_dir = dataset_dir / "tables"
        fig_dir = dataset_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        tr_path = table_dir / "Training_Predictions_All_Models.csv"
        te_path = table_dir / "Testing_Predictions_All_Models.csv"

        if tr_path.exists() and te_path.exists():
            tr = pd.read_csv(tr_path)
            te = pd.read_csv(te_path)
            tr["Date"] = pd.to_datetime(tr["Date"])
            te["Date"] = pd.to_datetime(te["Date"])
            comb = pd.concat([tr, te], ignore_index=True)
            split_d = te["Date"].min()
            plot_time_series(comb, split_d, fig_dir / "02_Time_Series_Observed_vs_Predicted.png", title)
            print(f"  [OK] Time Series (Models + Observed) for {key} at 600 DPI")
            sys.stdout.flush()
        else:
            # Generate observed time series from raw CSV
            csv_file = DATASET_FILES.get(key)
            if csv_file and Path(csv_file).exists():
                raw = pd.read_csv(csv_file, low_memory=False)
                date_col = "Date" if "Date" in raw.columns else "date"
                gw_col = "DTWL" if "DTWL" in raw.columns else ("WL_in_mbgl" if "WL_in_mbgl" in raw.columns else "groundwater_level")
                raw["Date"] = pd.to_datetime(raw[date_col], errors="coerce")
                raw = raw.dropna(subset=["Date", gw_col]).rename(columns={gw_col: "Observed"})
                daily_raw = raw.groupby("Date")["Observed"].mean().reset_index()
                split_d = daily_raw["Date"].iloc[int(len(daily_raw)*0.7)]
                plot_time_series(daily_raw, split_d, fig_dir / "02_Time_Series_Observed_vs_Predicted.png", title)
                print(f"  [OK] Time Series (Observed Baseline) for {key} at 600 DPI")
                sys.stdout.flush()

    print("\nALL 4 TIME SERIES COMPLETED SUCCESSFULLY!")
    sys.stdout.flush()
