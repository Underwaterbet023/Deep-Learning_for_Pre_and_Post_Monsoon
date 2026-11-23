import os
import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

PLOT_FACE = "#ffffff"
GRID_COLOR = "#e5e7eb"
BORDER_COLOR = "#000000"

MODEL_COLORS = {
    "LSTM": "#e74c3c",            # Red
    "GRU": "#27ae60",             # Green
    "CNN-LSTM": "#00cec9",        # Cyan
    "Hybrid LSTM-GRU": "#0984e3", # Blue
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

def style_axis(ax, facecolor=PLOT_FACE):
    ax.set_facecolor(facecolor)
    ax.grid(True, color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1.5)
    ax.tick_params(direction="out", width=1.2, length=5, colors=BORDER_COLOR)

def save_and_show_figure(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if is_notebook():
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            pass
    plt.close(fig)

def save_and_show_plotly(fig, path, width=1500, height=720):
    html_path = path.with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    if is_notebook():
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            pass

# ------------------------------------------------------------------------------
# 02. NEW TIME SERIES PLOT (MATCHING USER REFERENCE IMAGE EXACTLY)
# ------------------------------------------------------------------------------
def plot_time_series_publication(pred_df, split_date, out_path, dataset_title):
    models = [m for m in MODEL_COLORS if m in pred_df.columns]
    df = pred_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    
    daily = df.groupby("Date").mean(numeric_only=True).reset_index().sort_values("Date")
    if len(daily) == 0:
        return

    cols_to_scale = ["Observed"] + models
    all_vals = daily[cols_to_scale].values.flatten()
    val_min = np.nanmin(all_vals)
    val_max = np.nanmax(all_vals)
    val_range = val_max - val_min if val_max > val_min else 1.0

    def scale_to_unit(v):
        return 2.0 * (v - val_min) / val_range - 1.0

    fig, ax = plt.subplots(figsize=(14.0, 5.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.3)

    # Horizontal zero baseline
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5, zorder=3)

    date_nums = mdates.date2num(daily["Date"])
    split_ts = pd.Timestamp(split_date) if split_date is not None else daily["Date"].iloc[int(len(daily)*0.7)]

    mask_train = daily["Date"] < split_ts
    mask_test = daily["Date"] >= split_ts

    # Vertical salmon/pink shaded band
    if mask_train.any() and mask_test.any():
        span_start = split_ts - pd.DateOffset(months=4)
        span_end = split_ts + pd.DateOffset(months=4)
        ax.axvspan(span_start, span_end, color="#ff7675", alpha=0.38, zorder=2, label="Transition / Anomaly Zone")

    obs_scaled = scale_to_unit(daily["Observed"].values)

    # Piecewise linear dashed trendlines
    if mask_train.sum() > 2:
        x_tr = date_nums[mask_train]
        y_tr = obs_scaled[mask_train]
        slope_tr, intercept_tr = np.polyfit(x_tr, y_tr, 1)
        ax.plot(daily["Date"][mask_train], slope_tr * x_tr + intercept_tr,
                color="black", linestyle="--", linewidth=1.6, zorder=4, label="Trendline (Train)")

    if mask_test.sum() > 2:
        x_te = date_nums[mask_test]
        y_te = obs_scaled[mask_test]
        slope_te, intercept_te = np.polyfit(x_te, y_te, 1)
        ax.plot(daily["Date"][mask_test], slope_te * x_te + intercept_te,
                color="black", linestyle="--", linewidth=1.6, zorder=4, label="Trendline (Test)")

    # Plot Model Lines
    for m in models:
        y_sc = scale_to_unit(daily[m].values)
        ax.plot(daily["Date"], y_sc, color=MODEL_COLORS[m],
                linewidth=1.7, label=m, zorder=6)

    # Plot Observed Line on top in bold black
    ax.plot(daily["Date"], obs_scaled, color="#000000",
            linewidth=2.0, label="Observed", zorder=8)

    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.minorticks_on()
    ax.tick_params(which="both", direction="out", top=False, right=False)
    ax.tick_params(which="major", length=6, width=1.2, labelsize=10, colors="#222222")
    ax.tick_params(which="minor", length=3.5, width=0.8, colors="#444444")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    year_span = (daily["Date"].max() - daily["Date"].min()).days / 365.25
    ax.xaxis.set_major_locator(mdates.YearLocator(2 if year_span > 10 else 1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))

    ax.set_title(f"Time Series Groundwater Dynamics & Deep Learning Predictions — {dataset_title}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Standardized / Normalized Index", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (Years)", fontsize=11, fontweight="bold")

    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#333333",
              framealpha=0.92, fontsize=8.5, ncol=3)

    fig.tight_layout()
    save_and_show_figure(fig, out_path, dpi=300)

def plot_time_series_interactive(pred_df, split_date, out_path, dataset_title):
    models = [m for m in MODEL_COLORS if m in pred_df.columns]
    df = pred_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    daily = df.groupby("Date").mean(numeric_only=True).reset_index().sort_values("Date")
    if len(daily) == 0:
        return

    cols_to_scale = ["Observed"] + models
    all_vals = daily[cols_to_scale].values.flatten()
    val_min = np.nanmin(all_vals)
    val_max = np.nanmax(all_vals)
    val_range = val_max - val_min if val_max > val_min else 1.0

    def scale_to_unit(v):
        return 2.0 * (v - val_min) / val_range - 1.0

    fig = go.Figure()

    # Zero line
    fig.add_hline(y=0.0, line_dash="dash", line_color="black", line_width=1.5)

    split_ts = pd.Timestamp(split_date) if split_date is not None else daily["Date"].iloc[int(len(daily)*0.7)]
    span_start = split_ts - pd.DateOffset(months=4)
    span_end = split_ts + pd.DateOffset(months=4)
    fig.add_vrect(
        x0=str(span_start.date()), x1=str(span_end.date()),
        fillcolor="rgba(255, 118, 117, 0.38)", line_width=0,
        annotation_text="Transition / Anomaly Zone", annotation_position="top left",
        annotation_font=dict(size=10, color="#d63031")
    )

    date_nums = mdates.date2num(daily["Date"])
    mask_train = daily["Date"] < split_ts
    mask_test = daily["Date"] >= split_ts
    obs_scaled = scale_to_unit(daily["Observed"].values)

    if mask_train.sum() > 2:
        x_tr = date_nums[mask_train]
        y_tr = obs_scaled[mask_train]
        slope_tr, intercept_tr = np.polyfit(x_tr, y_tr, 1)
        fig.add_trace(go.Scatter(
            x=daily["Date"][mask_train], y=slope_tr * x_tr + intercept_tr,
            mode="lines", name="Trendline (Train)",
            line=dict(color="black", width=1.6, dash="dash"),
            hoverinfo="skip"
        ))

    if mask_test.sum() > 2:
        x_te = date_nums[mask_test]
        y_te = obs_scaled[mask_test]
        slope_te, intercept_te = np.polyfit(x_te, y_te, 1)
        fig.add_trace(go.Scatter(
            x=daily["Date"][mask_test], y=slope_te * x_te + intercept_te,
            mode="lines", name="Trendline (Test)",
            line=dict(color="black", width=1.6, dash="dash"),
            hoverinfo="skip"
        ))

    for m in models:
        y_sc = scale_to_unit(daily[m].values)
        fig.add_trace(go.Scatter(
            x=daily["Date"], y=y_sc, mode="lines", name=m,
            line=dict(color=MODEL_COLORS[m], width=2.0),
            customdata=daily[m].values,
            hovertemplate=f"<b>{m}</b><br>Date: %{{x|%b %Y}}<br>Normalized Index: %{{y:.3f}}<br>Physical Level: %{{customdata:.2f}} mbgl<extra></extra>"
        ))

    fig.add_trace(go.Scatter(
        x=daily["Date"], y=obs_scaled, mode="lines", name="Observed",
        line=dict(color="#000000", width=2.5),
        customdata=daily["Observed"].values,
        hovertemplate="<b>Observed</b><br>Date: %{x|%b %Y}<br>Normalized Index: %{y:.3f}<br>Physical Level: %{customdata:.2f}} mbgl<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Interactive Time Series — {dataset_title}</b>",
            font=dict(size=16, family="Arial Black", color="#0f172a"),
            x=0.5, xanchor="center"
        ),
        xaxis=dict(
            title="<b>Time (Years)</b>", showgrid=True, gridcolor="#e2e8f0",
            linecolor="black", linewidth=1.5, mirror=True, tickfont=dict(size=11)
        ),
        yaxis=dict(
            title="<b>Standardized / Normalized Index [-1.0, 1.0]</b>", range=[-1.05, 1.05],
            tickmode="array", tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
            showgrid=True, gridcolor="#e2e8f0", linecolor="black", linewidth=1.5, mirror=True, tickfont=dict(size=11)
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        width=1500, height=650,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
            bgcolor="rgba(255,255,255,0.95)", bordercolor="black", borderwidth=1.2, font=dict(size=11)
        ),
        margin=dict(l=70, r=50, t=100, b=70),
        hovermode="x unified",
    )
    save_and_show_plotly(fig, out_path, width=1500, height=650)

def plot_time_series(df, split_date, out_path, dataset_title):
    plot_time_series_publication(df, split_date, out_path, dataset_title)
    try:
        plot_time_series_interactive(df, split_date, out_path.with_name(out_path.stem + "_interactive"), dataset_title)
    except Exception as e:
        print("Time series interactive error:", e)

# ------------------------------------------------------------------------------
# 05. CHRONOLOGICAL PREDICTION RESIDUALS (FIXED ACCORDING TO USER DATA)
# ------------------------------------------------------------------------------
def plot_residuals(pred_df, out_path, dataset_title):
    models = [m for m in MODEL_COLORS if m in pred_df.columns]
    df = pred_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    for m in models:
        df[f"{m}_res"] = df[m] - df["Observed"]

    res_summary = df.groupby("Date").agg(
        **{f"{m}_mean": (f"{m}_res", "mean") for m in models},
        **{f"{m}_q25": (f"{m}_res", lambda x: x.quantile(0.25)) for m in models},
        **{f"{m}_q75": (f"{m}_res", lambda x: x.quantile(0.75)) for m in models},
    ).reset_index().sort_values("Date")

    fig, ax = plt.subplots(figsize=(14.5, 6.0), dpi=300)
    fig.patch.set_facecolor("white")
    style_axis(ax, facecolor="white")

    ax.axhline(0, color="black", linestyle="--", linewidth=1.6, label="Zero Error (Residual = 0)", zorder=12)

    for m in models:
        ax.fill_between(
            res_summary["Date"], res_summary[f"{m}_q25"], res_summary[f"{m}_q75"],
            color=MODEL_COLORS[m], alpha=0.12, zorder=5
        )
        ax.plot(
            res_summary["Date"], res_summary[f"{m}_mean"], color=MODEL_COLORS[m],
            linewidth=1.9, label=f"{m} (Mean Residual)", zorder=8
        )

    ax.set_title(f"Chronological Prediction Residuals (Predicted − Observed) — {dataset_title}",
                 fontweight="bold", fontsize=13, pad=12)
    ax.set_xlabel("Time (Years)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Prediction Residual Error (m)", fontweight="bold", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator(2 if len(res_summary) > 200 else 1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.legend(loc="upper right", fontsize=8.5, ncol=2, frameon=True, facecolor="white", edgecolor="#333333", framealpha=0.95)

    fig.tight_layout()
    save_and_show_figure(fig, out_path, dpi=300)

    fig_i = go.Figure()
    fig_i.add_hline(y=0, line_dash="dash", line_color="black", line_width=1.6, annotation_text="Zero Error Baseline")
    for m in models:
        fig_i.add_trace(go.Scatter(
            x=res_summary["Date"], y=res_summary[f"{m}_mean"], mode="lines", name=f"{m} (Mean)",
            line=dict(color=MODEL_COLORS[m], width=2.2),
            hovertemplate=f"<b>{m}</b><br>Date: %{{x|%b %Y}}<br>Mean Residual: %{{y:+.3f}} m<extra></extra>"
        ))
    fig_i.update_layout(
        title=dict(text=f"<b>Interactive Chronological Residuals — {dataset_title}</b>", font=dict(size=16, family="Arial Black")),
        xaxis=dict(title="<b>Time (Years)</b>", showgrid=True, gridcolor="#e2e8f0", linecolor="black", linewidth=1.5, mirror=True),
        yaxis=dict(title="<b>Prediction Residual Error (m)</b>", showgrid=True, gridcolor="#e2e8f0", linecolor="black", linewidth=1.5, mirror=True),
        plot_bgcolor="white", paper_bgcolor="white",
        width=1500, height=650,
        margin=dict(l=70, r=50, t=100, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.95)", bordercolor="black", borderwidth=1),
        hovermode="x unified",
    )
    save_and_show_plotly(fig_i, out_path.with_name(out_path.stem + "_interactive"), width=1500, height=650)

# ------------------------------------------------------------------------------
# 03. ROBUST CORRELATION MATRIX (FIXES THE KEYERROR)
# ------------------------------------------------------------------------------
def plot_correlation_matrix_publication(corr, out_path, dataset_title):
    valid_cols = [c for c in corr.columns if not corr[c].isna().all()]
    corr = corr.loc[valid_cols, valid_cols]
    cols = list(corr.columns)
    data = corr.values
    n = len(cols)
    
    fig, ax = plt.subplots(figsize=(max(11, n * 0.78), max(9.5, n * 0.68)), dpi=300)
    im = ax.imshow(data, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(cols, rotation=90, ha="center", fontsize=9.0, fontweight="bold")
    ax.set_yticklabels(cols, fontsize=9.0, fontweight="bold")
    
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            text_color = "white" if abs(val) > 0.45 else "black"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.0,
                color=text_color, fontweight="normal" if abs(val) < 0.70 else "bold"
            )
            
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_ticks(np.linspace(-1, 1, 9))
    cbar.set_label("Pearson Correlation Coefficient", fontweight="bold", fontsize=10.5)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1.2)
    
    ax.set_title(f"Correlation Matrix — {dataset_title}", fontweight="bold", fontsize=13.5, pad=14)
    fig.patch.set_linewidth(1.5)
    fig.patch.set_edgecolor("black")
    fig.tight_layout()
    save_and_show_figure(fig, out_path, dpi=300)

def plot_correlation_matrix_interactive(corr, out_path, dataset_title):
    valid_cols = [c for c in corr.columns if not corr[c].isna().all()]
    corr = corr.loc[valid_cols, valid_cols]
    cols = list(corr.columns)
    z = np.round(corr.values, 2)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=cols, y=cols,
        colorscale="RdBu_r", zmin=-1.0, zmax=1.0,
        text=z, texttemplate="%{text:.2f}", textfont={"size": 11},
        colorbar=dict(title=dict(text="Correlation", font=dict(family="Arial", size=12, color="black"))),
        hovertemplate="<b>Feature X:</b> %{x}<br><b>Feature Y:</b> %{y}<br><b>Correlation:</b> %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=f"<b>Interactive Correlation Matrix — {dataset_title}</b>", font=dict(size=16, family="Arial Black")),
        xaxis=dict(tickangle=-90, tickfont=dict(size=10.5, family="Arial"), showgrid=False),
        yaxis=dict(tickfont=dict(size=10.5, family="Arial"), showgrid=False, autorange="reversed"),
        width=950, height=900,
        margin=dict(l=130, r=60, t=80, b=130),
        paper_bgcolor="white", plot_bgcolor="white"
    )
    save_and_show_plotly(fig, out_path, width=950, height=900)

def plot_correlation_matrix(corr_or_df, out_path, dataset_title):
    if isinstance(corr_or_df, pd.DataFrame):
        if not (corr_or_df.index.equals(corr_or_df.columns)):
            numeric_df = corr_or_df.select_dtypes(include=[np.number])
            cols = [c for c in numeric_df.columns if c not in ["well_id", "date", "Date"]]
            corr = numeric_df[cols].corr()
        else:
            corr = corr_or_df
    else:
        corr = corr_or_df
    plot_correlation_matrix_publication(corr, out_path, dataset_title)
    try:
        plot_correlation_matrix_interactive(corr, out_path.with_name(out_path.stem + "_interactive"), dataset_title)
    except Exception as e:
        print("Correlation interactive failed:", e)

if __name__ == "__main__":
    print("Generating figures across datasets...")
    sys.stdout.flush()
    for dataset_key, title in DATASET_TITLES.items():
        dataset_dir = OUTPUT_ROOT / dataset_key
        table_dir = dataset_dir / "tables"
        fig_dir = dataset_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # 1. Correlation Matrix
        if (table_dir / "03_Correlation_Matrix.csv").exists():
            corr = pd.read_csv(table_dir / "03_Correlation_Matrix.csv", index_col=0)
            plot_correlation_matrix(corr, fig_dir / "03_Correlation_Matrix.png", title)
            print(f"  [OK] 03_Correlation_Matrix for {dataset_key}")
            sys.stdout.flush()
        
        # 2. Time Series & Residuals
        tr_path = table_dir / "Training_Predictions_All_Models.csv"
        te_path = table_dir / "Testing_Predictions_All_Models.csv"
        if tr_path.exists() and te_path.exists():
            tr = pd.read_csv(tr_path)
            te = pd.read_csv(te_path)
            tr["Date"] = pd.to_datetime(tr["Date"])
            te["Date"] = pd.to_datetime(te["Date"])
            comb = pd.concat([tr, te], ignore_index=True)
            split_d = te["Date"].min()

            # Plot Time Series
            plot_time_series(comb, split_d, fig_dir / "02_Time_Series_Observed_vs_Predicted.png", title)
            print(f"  [OK] 02_Time_Series_Observed_vs_Predicted for {dataset_key}")
            sys.stdout.flush()

            # Plot Residuals
            plot_residuals(te, fig_dir / "05_Test_Residuals_vs_Time.png", title)
            print(f"  [OK] 05_Test_Residuals_vs_Time for {dataset_key}")
            sys.stdout.flush()
        else:
            # If no model predictions yet, generate the Observed time series matching the new design!
            csv_file = DATASET_FILES.get(dataset_key)
            if csv_file and Path(csv_file).exists():
                raw = pd.read_csv(csv_file, low_memory=False)
                date_col = "Date" if "Date" in raw.columns else "date"
                gw_col = "DTWL" if "DTWL" in raw.columns else ("WL_in_mbgl" if "WL_in_mbgl" in raw.columns else "groundwater_level")
                raw["Date"] = pd.to_datetime(raw[date_col], errors="coerce")
                raw = raw.dropna(subset=["Date", gw_col]).rename(columns={gw_col: "Observed"})
                daily_raw = raw.groupby("Date")["Observed"].mean().reset_index()
                split_d = daily_raw["Date"].iloc[int(len(daily_raw)*0.7)]
                plot_time_series(daily_raw, split_d, fig_dir / "02_Time_Series_Observed_vs_Predicted.png", title)
                print(f"  [OK] 02_Time_Series_Observed (Baseline) for {dataset_key}")
                sys.stdout.flush()

    print("\nALL AVAILABLE FIGURES GENERATED SUCCESSFULLY!")
    sys.stdout.flush()
