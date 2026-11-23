import json
import re

nb_path = "HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))

cell8_code = "".join(nb['cells'][8]['source'])

# 1. Add SHOW_FIGURES = False
if "SHOW_FIGURES =" not in cell8_code:
    cell8_code = cell8_code.replace(
        'warnings.filterwarnings("ignore")',
        'warnings.filterwarnings("ignore")\n\nSHOW_FIGURES = False  # Set to True to display figures inline in notebook'
    )

# 2. Update MODEL_COLORS
cell8_code = re.sub(
    r'MODEL_COLORS = \{.*?\n\}\n\nMODEL_STYLES = \{.*?\n\}',
    '''MODEL_COLORS = {
    "Observed": "#000000",       # BLACK
    "LSTM": "#e74c3c",           # RED
    "GRU": "#27ae60",            # GREEN
    "CNN-LSTM": "#00cec9",       # CYAN
    "Hybrid LSTM-GRU": "#0984e3",# BLUE
}

MODEL_STYLES = {
    "Observed": {"linestyle": "-", "color": "#000000", "linewidth": 1.6},
    "LSTM": {"linestyle": "-", "color": "#e74c3c", "linewidth": 1.3},
    "GRU": {"linestyle": "-", "color": "#27ae60", "linewidth": 1.3},
    "CNN-LSTM": {"linestyle": "-", "color": "#00cec9", "linewidth": 1.3},
    "Hybrid LSTM-GRU": {"linestyle": "-", "color": "#0984e3", "linewidth": 1.3},
}''',
    cell8_code,
    flags=re.DOTALL
)

# 3. Update save_and_show_figure and save_and_show_plotly for performance
new_saves = '''def is_notebook():
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
            pass'''

old_save_pattern = re.compile(r'def is_notebook\(\):.*?def save_and_show_plotly\(fig, path, width=1500, height=720\):.*?\n(?=def |\n#|def build_continuous|\n\ndef build_continuous)', re.DOTALL)
cell8_code = old_save_pattern.sub(new_saves + "\n", cell8_code)

# 4. Remove build_continuous_seasonal_trajectory and update plot_observed_timeline
old_traj_pattern = re.compile(
    r'# ------------------------------------------------------------------------------\n# CONTINUOUS SEASONAL HYDROLOGICAL WAVEFORM GENERATOR.*?(?=# 00\. REGIONAL OBSERVED GROUNDWATER TIMELINE)',
    re.DOTALL
)
cell8_code = old_traj_pattern.sub("", cell8_code)

new_observed_timeline = '''# ------------------------------------------------------------------------------
# 00. REGIONAL OBSERVED GROUNDWATER TIMELINE (REAL OBSERVED DATA ONLY)
# ------------------------------------------------------------------------------
def plot_observed_timeline(raw_df, out_path, dataset_title):
    date_col = "date" if "date" in raw_df.columns else ("Date" if "Date" in raw_df.columns else None)
    gw_col = "groundwater_level" if "groundwater_level" in raw_df.columns else ("DTWL" if "DTWL" in raw_df.columns else "WL_in_mbgl")
    if date_col is None or gw_col not in raw_df.columns:
        return

    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["Date", gw_col])

    # Real data grouped by Date (regional mean and real IQR)
    daily = df.groupby("Date")[gw_col].agg(
        Observed="mean",
        Q25=lambda x: x.quantile(0.25),
        Q75=lambda x: x.quantile(0.75),
    ).reset_index().sort_values("Date")
    if len(daily) == 0:
        return

    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=600)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.9)
    ax.grid(False)

    ax.fill_between(
        daily["Date"], daily["Q25"], daily["Q75"],
        color="#cbd5e1", alpha=0.35, label="Interquartile Range (25th–75th %)", zorder=3
    )
    ax.plot(daily["Date"], daily["Observed"], color="#000000", linewidth=1.5, label="Observed Regional Mean", zorder=8)

    ax.set_title(f"Regional Observed Groundwater Timeline — {dataset_title}", fontweight="bold", fontsize=11.5, pad=10)
    ax.set_xlabel("Time (Years)", fontweight="bold", fontsize=9.5)
    ax.set_ylabel("Groundwater Level (m bgl)", fontweight="bold", fontsize=9.5)

    year_span = (daily["Date"].max() - daily["Date"].min()).days / 365.25
    ax.xaxis.set_major_locator(mdates.YearLocator(2 if year_span > 10 else 1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="white", edgecolor="#777777", framealpha=0.95)

    fig.tight_layout()
    save_and_show_figure(fig, out_path, dpi=600)

    # Interactive version
    fig_i = go.Figure()
    fig_i.add_trace(go.Scatter(
        x=daily["Date"], y=daily["Q75"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig_i.add_trace(go.Scatter(
        x=daily["Date"], y=daily["Q25"], mode="lines", fill="tonexty", fillcolor="rgba(203, 213, 225, 0.4)",
        line=dict(width=0), name="IQR (25th–75th %)", hoverinfo="skip"
    ))
    fig_i.add_trace(go.Scatter(
        x=daily["Date"], y=daily["Observed"], mode="lines", name="Observed Regional Mean",
        line=dict(color="#000000", width=2.0),
        hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>GW Level:</b> %{y:.2f} mbgl<extra></extra>"
    ))
    fig_i.update_layout(
        title=dict(text=f"<b>Regional Observed Groundwater Timeline — {dataset_title}</b>", font=dict(size=14, family="Arial")),
        xaxis=dict(title="<b>Time (Years)</b>", showgrid=False, linecolor="#777777", linewidth=0.9, mirror=True),
        yaxis=dict(title="<b>Groundwater Level (m bgl)</b>", showgrid=False, linecolor="#777777", linewidth=0.9, mirror=True),
        plot_bgcolor="white", paper_bgcolor="white",
        width=1200, height=520,
        margin=dict(l=65, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.95)", bordercolor="#777777", borderwidth=0.8),
    )
    save_and_show_plotly(fig_i, out_path.with_name(out_path.stem + "_interactive"), width=1200, height=520)'''

old_obs_pattern = re.compile(
    r'# 00\. REGIONAL OBSERVED GROUNDWATER TIMELINE.*?# ------------------------------------------------------------------------------\n# 02\. TIME SERIES PLOT',
    re.DOTALL
)
cell8_code = old_obs_pattern.sub(new_observed_timeline + "\n\n# ------------------------------------------------------------------------------\n# 02. TIME SERIES PLOT", cell8_code)

# 5. Replace plot_time_series_publication and plot_time_series_interactive with exact user specification
new_ts_functions = '''# 02. TIME SERIES PLOT (MATCHING USER REFERENCE IMAGE EXACTLY)
# ------------------------------------------------------------------------------
def plot_time_series_publication(pred_df, split_date, out_path, dataset_title):
    models = [m for m in ["LSTM", "GRU", "CNN-LSTM", "Hybrid LSTM-GRU"] if m in pred_df.columns]
    df = pred_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # Real data grouped by Date (regional mean only for visualization)
    daily = df.groupby("Date")[["Observed"] + models].mean().reset_index().sort_values("Date")
    if len(daily) == 0:
        return

    # Common standardization across all series
    all_vals = np.concatenate([daily[col].dropna().values for col in ["Observed"] + models])
    common_mean = float(np.mean(all_vals))
    common_std = float(np.std(all_vals))
    if common_std == 0 or np.isnan(common_std):
        common_std = 1.0

    daily_scaled = pd.DataFrame({"Date": daily["Date"]})
    daily_scaled["Observed"] = (daily["Observed"] - common_mean) / common_std
    for m in models:
        daily_scaled[m] = (daily[m] - common_mean) / common_std

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

    # Model curves: LSTM (Red), GRU (Green), CNN-LSTM (Cyan), Hybrid LSTM-GRU (Blue)
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
    save_and_show_plotly(fig, out_path, width=1200, height=520)'''

old_ts_def_pattern = re.compile(
    r'# 02\. TIME SERIES PLOT \(MATCHING USER REFERENCE IMAGE EXACTLY\).*?# 03\. CORRELATION MATRIX',
    re.DOTALL
)
cell8_code = old_ts_def_pattern.sub(new_ts_functions + "\n\n# ------------------------------------------------------------------------------\n# 03. CORRELATION MATRIX", cell8_code)

nb['cells'][8]['source'] = [line + "\n" for line in cell8_code.splitlines()]

# Save notebook
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with exact user specifications!")

# Also update generate_all_figures_standalone.py
standalone_path = "generate_all_figures_standalone.py"
standalone_content = open(standalone_path, encoding="utf-8").read()

# Replace in standalone
standalone_content = old_save_pattern.sub(new_saves + "\n", standalone_content)
standalone_content = old_traj_pattern.sub("", standalone_content)
standalone_content = old_obs_pattern.sub(new_observed_timeline + "\n\n# ------------------------------------------------------------------------------\n# 02. TIME SERIES PLOT", standalone_content)
standalone_content = old_ts_def_pattern.sub(new_ts_functions + "\n\n# ------------------------------------------------------------------------------\n# 03. CORRELATION MATRIX", standalone_content)

with open(standalone_path, "w", encoding="utf-8") as f:
    f.write(standalone_content)

print("generate_all_figures_standalone.py updated with exact user specifications!")
