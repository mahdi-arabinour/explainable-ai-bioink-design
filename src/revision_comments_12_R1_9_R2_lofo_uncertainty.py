from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "results" / "revision_comments_4_R1_16_R2"
OUT_DIR = ROOT / "results" / "revision_comments_12_R1_9_R2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_FILE = IN_DIR / "Nested_LOFO_Group_Metrics_Comments_4-R1_16-R2.csv"
SUMMARY_FILE = IN_DIR / "Nested_LOFO_Summary_Comments_4-R1_16-R2.csv"
PRED_FILE = IN_DIR / "Nested_LOFO_Predictions_Comments_4-R1_16-R2.csv"

BOOT_REPS = 10000
BOOT_SEED = 42

groups = pd.read_csv(GROUP_FILE)
summary = pd.read_csv(SUMMARY_FILE)
preds = pd.read_csv(PRED_FILE)

sr = groups.loc[groups["Target"] == "SR"].copy().reset_index(drop=True)
sr = sr[[
    "Held_out_formulation", "N_train", "N_test", "Selected_feature_set", "Selected_model",
    "R2", "MAE", "RMSE", "Spearman"
]]

pooled = summary.loc[summary["Target"] == "SR"].iloc[0]

sr_summary = pd.DataFrame([{
    "Target": "SR",
    "N_outer_formulations": int(pooled["N_outer_formulations"]),
    "Pooled_LOFO_R2": float(pooled["LOFO_R2"]),
    "Pooled_LOFO_MAE": float(pooled["LOFO_MAE"]),
    "Pooled_LOFO_RMSE": float(pooled["LOFO_RMSE"]),
    "Pooled_LOFO_Spearman": float(pooled["LOFO_Spearman"]),
    "Fold_R2_mean": float(sr["R2"].mean()),
    "Fold_R2_SD": float(sr["R2"].std(ddof=1)),
    "Fold_R2_median": float(sr["R2"].median()),
    "Fold_R2_min": float(sr["R2"].min()),
    "Fold_R2_max": float(sr["R2"].max()),
    "Positive_R2_folds": int((sr["R2"] > 0).sum()),
    "Negative_R2_folds": int((sr["R2"] < 0).sum()),
    "Fold_MAE_mean": float(sr["MAE"].mean()),
    "Fold_MAE_SD": float(sr["MAE"].std(ddof=1)),
    "Fold_RMSE_mean": float(sr["RMSE"].mean()),
    "Fold_RMSE_SD": float(sr["RMSE"].std(ddof=1)),
    "Fold_Spearman_median": float(sr["Spearman"].median()),
}])

# Compact all-target distribution summary to avoid selectively reporting SR only.
all_summary_rows = []
for target, d in groups.groupby("Target", sort=True):
    p = summary.loc[summary["Target"] == target].iloc[0]
    all_summary_rows.append({
        "Target": target,
        "N_outer_formulations": int(p["N_outer_formulations"]),
        "Pooled_LOFO_R2": float(p["LOFO_R2"]),
        "Fold_R2_median": float(d["R2"].median()),
        "Fold_R2_min": float(d["R2"].min()),
        "Fold_R2_max": float(d["R2"].max()),
        "Positive_R2_folds": int((d["R2"] > 0).sum()),
        "Fold_MAE_mean": float(d["MAE"].mean()),
        "Fold_MAE_SD": float(d["MAE"].std(ddof=1)),
        "Fold_RMSE_mean": float(d["RMSE"].mean()),
        "Fold_RMSE_SD": float(d["RMSE"].std(ddof=1)),
    })
all_summary = pd.DataFrame(all_summary_rows)


def pooled_metrics(df):
    y = df["Observed"].to_numpy(dtype=float)
    yp = df["Predicted"].to_numpy(dtype=float)
    err = y - yp
    sse = float(np.sum(err ** 2))
    ybar = float(np.mean(y))
    sst = float(np.sum((y - ybar) ** 2))
    r2 = np.nan if sst <= 0 else 1.0 - sse / sst
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return r2, mae, rmse


def block_bootstrap_for_target(df, rng):
    """Evaluation-stage cluster bootstrap of already-held-out nested-LOFO predictions.

    The resampling unit is the outer held-out formulation. All observations belonging to a
    sampled formulation are carried together. Models are not refit in this bootstrap; it
    quantifies variation in the pooled evaluation metric across the eight observed formulation
    blocks, conditional on the nested-LOFO outer predictions already produced.
    """
    formulations = sorted(df["Held_out_formulation"].astype(str).unique())
    blocks = []
    for g in formulations:
        z = df.loc[df["Held_out_formulation"].astype(str) == g]
        y = z["Observed"].to_numpy(dtype=float)
        yp = z["Predicted"].to_numpy(dtype=float)
        e = y - yp
        blocks.append({
            "formulation": g,
            "n": len(z),
            "sum_y": float(np.sum(y)),
            "sum_y2": float(np.sum(y ** 2)),
            "sse": float(np.sum(e ** 2)),
            "sae": float(np.sum(np.abs(e))),
        })
    stats = pd.DataFrame(blocks)
    arr = stats[["n", "sum_y", "sum_y2", "sse", "sae"]].to_numpy(dtype=float)
    draw_idx = rng.integers(0, len(formulations), size=(BOOT_REPS, len(formulations)))
    agg = arr[draw_idx].sum(axis=1)
    n = agg[:, 0]
    sy = agg[:, 1]
    sy2 = agg[:, 2]
    sse = agg[:, 3]
    sae = agg[:, 4]
    sst = sy2 - (sy ** 2) / n
    r2 = 1.0 - sse / sst
    mae = sae / n
    rmse = np.sqrt(sse / n)

    # Delete-one-formulation evaluation sensitivity using the same fixed outer predictions.
    delete_one = []
    for g in formulations:
        z = df.loc[df["Held_out_formulation"].astype(str) != g]
        r2_g, mae_g, rmse_g = pooled_metrics(z)
        delete_one.append({
            "Omitted_formulation": g,
            "R2": r2_g,
            "MAE": mae_g,
            "RMSE": rmse_g,
        })
    delete_one = pd.DataFrame(delete_one)

    base_r2, base_mae, base_rmse = pooled_metrics(df)
    row = {
        "N_outer_formulations": len(formulations),
        "Bootstrap_replicates": BOOT_REPS,
        "Pooled_LOFO_R2": base_r2,
        "Bootstrap_R2_median": float(np.median(r2)),
        "Bootstrap_R2_2p5": float(np.quantile(r2, 0.025)),
        "Bootstrap_R2_97p5": float(np.quantile(r2, 0.975)),
        "Bootstrap_P_R2_gt_0": float(np.mean(r2 > 0)),
        "Pooled_LOFO_MAE": base_mae,
        "Bootstrap_MAE_2p5": float(np.quantile(mae, 0.025)),
        "Bootstrap_MAE_97p5": float(np.quantile(mae, 0.975)),
        "Pooled_LOFO_RMSE": base_rmse,
        "Bootstrap_RMSE_2p5": float(np.quantile(rmse, 0.025)),
        "Bootstrap_RMSE_97p5": float(np.quantile(rmse, 0.975)),
        "Delete1_R2_min": float(delete_one["R2"].min()),
        "Delete1_R2_max": float(delete_one["R2"].max()),
    }
    return row, delete_one, r2

rng = np.random.default_rng(BOOT_SEED)
bootstrap_rows = []
delete_rows = []
sr_draws = None
for target in sorted(preds["Target"].unique()):
    d = preds.loc[preds["Target"] == target].copy().reset_index(drop=True)
    row, delete_one, r2_draws = block_bootstrap_for_target(d, rng)
    row["Target"] = target
    bootstrap_rows.append(row)
    delete_one.insert(0, "Target", target)
    delete_rows.append(delete_one)
    if target == "SR":
        sr_draws = r2_draws.copy()

bootstrap_summary = pd.DataFrame(bootstrap_rows)[[
    "Target", "N_outer_formulations", "Bootstrap_replicates", "Pooled_LOFO_R2",
    "Bootstrap_R2_median", "Bootstrap_R2_2p5", "Bootstrap_R2_97p5", "Bootstrap_P_R2_gt_0",
    "Pooled_LOFO_MAE", "Bootstrap_MAE_2p5", "Bootstrap_MAE_97p5",
    "Pooled_LOFO_RMSE", "Bootstrap_RMSE_2p5", "Bootstrap_RMSE_97p5",
    "Delete1_R2_min", "Delete1_R2_max"
]]
delete_summary = pd.concat(delete_rows, ignore_index=True)

sr_boot = bootstrap_summary.loc[bootstrap_summary["Target"] == "SR"].copy()

sr.to_csv(OUT_DIR / "SR_Nested_LOFO_Per_Formulation_Comments_12-R1_9-R2.csv", index=False)
sr_summary.to_csv(OUT_DIR / "SR_Nested_LOFO_Summary_Comments_12-R1_9-R2.csv", index=False)
groups.to_csv(OUT_DIR / "All_Targets_Nested_LOFO_Per_Formulation_Comments_12-R1_9-R2.csv", index=False)
all_summary.to_csv(OUT_DIR / "All_Targets_Nested_LOFO_Distribution_Summary_Comments_12-R1_9-R2.csv", index=False)
bootstrap_summary.to_csv(OUT_DIR / "All_Targets_LOFO_Formulation_Block_Bootstrap_Comments_12-R1_9-R2.csv", index=False)
sr_boot.to_csv(OUT_DIR / "SR_LOFO_Formulation_Block_Bootstrap_Comments_12-R1_9-R2.csv", index=False)
delete_summary.to_csv(OUT_DIR / "All_Targets_LOFO_Delete1_Evaluation_Sensitivity_Comments_12-R1_9-R2.csv", index=False)
np.savez_compressed(OUT_DIR / "SR_LOFO_Formulation_Block_Bootstrap_R2_Draws_Comments_12-R1_9-R2.npz", r2=sr_draws)

# Dedicated SI workbook generated from the same canonical uncertainty analysis.
with pd.ExcelWriter(OUT_DIR / "LOFO_Analysis_Comments_12-R1_9-R2.xlsx", engine="openpyxl") as writer:
    sr.to_excel(writer, sheet_name="SR_Per_Formulation", index=False)
    sr_summary.to_excel(writer, sheet_name="SR_Summary", index=False)
    groups.to_excel(writer, sheet_name="All_Targets_Per_Form", index=False)
    all_summary.to_excel(writer, sheet_name="All_Targets_Summary", index=False)
    bootstrap_summary.to_excel(writer, sheet_name="Block_Bootstrap", index=False)
    delete_summary.to_excel(writer, sheet_name="Delete1_Sensitivity", index=False)

metadata = {
    "comments_addressed": ["12-R1", "9-R2"],
    "analysis_type": "formulation-level uncertainty analysis of existing nested LOFO outer predictions",
    "source_group_metrics": str(GROUP_FILE.relative_to(ROOT)),
    "source_pooled_summary": str(SUMMARY_FILE.relative_to(ROOT)),
    "source_outer_predictions": str(PRED_FILE.relative_to(ROOT)),
    "new_model_fits_required": False,
    "bootstrap_replicates": BOOT_REPS,
    "bootstrap_seed": BOOT_SEED,
    "bootstrap_unit": "outer held-out formulation block",
    "bootstrap_interval": "percentile 95% interval",
    "statistical_note": (
        "Per-formulation R2 values are reported as fold-level diagnostics and are not averaged to replace pooled LOFO R2. "
        "The formulation-block bootstrap resamples the eight already-held-out formulation blocks with replacement while preserving "
        "all observations within each block. It quantifies evaluation-sample uncertainty conditional on the fixed nested-LOFO outer "
        "predictions; it does not refit model-selection/training procedures within each bootstrap replicate and therefore is not a full "
        "training-procedure bootstrap. With only eight formulation blocks, the interval is a sensitivity analysis rather than a precise "
        "population-level confidence statement."
    ),
}
(OUT_DIR / "LOFO_Metadata_Comments_12-R1_9-R2.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("SR nested LOFO per-formulation diagnostics")
print(sr.to_string(index=False))
print("\nSR descriptive summary")
print(sr_summary.to_string(index=False))
print("\nAll-target formulation-block bootstrap summary")
print(bootstrap_summary.to_string(index=False))
print("\nSR formulation-block bootstrap")
print(sr_boot.to_string(index=False))
print("\nMetadata")
print(json.dumps(metadata, indent=2))
