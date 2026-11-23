import json

nb_path = "HIGDSS_Deep_Learning_4_SEPARATE_DATASETS_RESEARCH.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))

cell8_code = "".join(nb['cells'][8]['source'])

taylor_func_code = '''# ------------------------------------------------------------------------------
# TAYLOR DIAGRAM STATISTICS DATAFRAME GENERATOR
# ------------------------------------------------------------------------------
def make_taylor_dataframe(predictions, observed):
    obs_std = float(np.std(observed))
    records = []
    for name, pred in predictions.items():
        pred_std = float(np.std(pred))
        r = (float(stats.pearsonr(observed, pred)[0])
             if np.std(observed) > 0 and np.std(pred) > 0 else np.nan)
        crmse = (float(np.sqrt(obs_std**2 + pred_std**2 - 2 * obs_std * pred_std * r))
                 if np.isfinite(r) else np.nan)
        records.append({
            "Model": name,
            "Observed_STD": obs_std,
            "Predicted_STD": pred_std,
            "Normalized_STD": pred_std / obs_std if obs_std > 0 else np.nan,
            "Pearson_r": r,
            "Centered_RMSE": crmse,
            "Normalized_CRMSE": crmse / obs_std if obs_std > 0 else np.nan,
        })
    result = pd.DataFrame(records)
    result.attrs["obs_std"] = obs_std
    return result

def regional_daily_summary(df):
    return (
        df.groupby("date")["groundwater_level"]
        .agg(
            observed_mean="mean",
            observed_q25=lambda x: x.quantile(0.25),
            observed_q75=lambda x: x.quantile(0.75),
            n="size",
        )
        .reset_index()
        .sort_values("date")
    )

def aggregate_predictions(pred_df):
    model_cols = list(MODEL_COLORS.keys())
    grouped = (
        pred_df.groupby("Date")
        .agg(
            Observed=("Observed", "mean"),
            Observed_Q25=("Observed", lambda x: x.quantile(0.25)),
            Observed_Q75=("Observed", lambda x: x.quantile(0.75)),
            **{m: (m, "mean") for m in model_cols if m in pred_df.columns},
            N=("Observed", "size"),
        )
        .reset_index()
        .sort_values("Date")
    )
    return grouped

save_figure = save_and_show_figure

'''

target = "def plot_taylor_publication(metrics_by_phase, out_path, dataset_title):"
if "def make_taylor_dataframe" not in cell8_code:
    cell8_code = cell8_code.replace(target, taylor_func_code + target)
    nb['cells'][8]['source'] = [line + "\n" for line in cell8_code.splitlines()]
    
    # Also clear cell 10 error output
    nb['cells'][10]['outputs'] = []
    nb['cells'][10]['execution_count'] = None

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("make_taylor_dataframe added to notebook successfully!")
else:
    print("make_taylor_dataframe already in notebook.")

# Also add to generate_all_figures_standalone.py
standalone_path = "generate_all_figures_standalone.py"
standalone_content = open(standalone_path, encoding="utf-8").read()
if "def make_taylor_dataframe" not in standalone_content:
    standalone_content = standalone_content.replace(target, taylor_func_code + target)
    with open(standalone_path, "w", encoding="utf-8") as f:
        f.write(standalone_content)
    print("make_taylor_dataframe added to standalone script successfully!")
