from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "revision_comments_2_R1_6_7_R2"
OUT.mkdir(parents=True, exist_ok=True)

printability = pd.read_csv(DATA / "printability_table_S2.csv")
angle = pd.read_csv(DATA / "angle_fidelity_table_S3.csv")

# Reviewer-responsive screening uses only jointly observed formulation-pressure
# conditions. This removes the weak Pr model from the gate and avoids ranking
# candidates more finely than the model error supports.
key = ["Alg_mg_ml", "HA_mg_ml", "Pressure_kPa"]

angle = angle.copy()
angle["AF_error_from_ideal_1"] = (angle["AF_mean"] - 1.0).abs()
angle["AF_within_1p00_1p25"] = angle["AF_mean"].between(1.00, 1.25, inclusive="both")

# For each observed formulation-pressure condition, retain the measured angle
# whose measured AF is closest to 1. This is a descriptive summary, not a model ranking.
best_measured_af = (
    angle.sort_values(["AF_error_from_ideal_1", "Angle_deg"], ascending=[True, True])
    .groupby(key, as_index=False)
    .first()
)
best_measured_af = best_measured_af[
    key + ["Angle_deg", "AF_mean", "AF_std", "AF_error_from_ideal_1", "AF_within_1p00_1p25"]
].rename(
    columns={
        "Angle_deg": "Best_measured_angle_deg",
        "AF_mean": "Best_measured_AF",
        "AF_std": "Best_measured_AF_std",
        "AF_error_from_ideal_1": "Best_measured_AF_error_from_1",
        "AF_within_1p00_1p25": "Measured_AF_criterion_1p00_1p25",
    }
)

overlap = printability.merge(best_measured_af, on=key, how="inner", validate="one_to_one")
overlap["Formulation_label"] = (
    (overlap["Alg_mg_ml"] / 10).astype(int).astype(str)
    + "ALG"
    + (overlap["HA_mg_ml"] / 10).astype(int).astype(str)
    + "HA"
)

overlap["Measured_Pr_criterion_0p95_1p05"] = overlap["Pr_mean"].between(0.95, 1.05, inclusive="both")
overlap["Measured_Pr_sensitivity_0p90_1p10"] = overlap["Pr_mean"].between(0.90, 1.10, inclusive="both")
overlap["Source_Qm_SR_feasibility"] = (overlap["Qm_mean_mg_s"] > 0) & (overlap["SR_mean"] < 6)

overlap["Candidate_set"] = (
    overlap["Measured_Pr_criterion_0p95_1p05"]
    & overlap["Measured_AF_criterion_1p00_1p25"]
)
overlap["Relaxed_sensitivity_set"] = (
    overlap["Measured_Pr_sensitivity_0p90_1p10"]
    & overlap["Measured_AF_criterion_1p00_1p25"]
)

# No combined error and no rank are calculated. Rows are sorted only for display.
overlap = overlap.sort_values(["Alg_mg_ml", "HA_mg_ml", "Pressure_kPa"]).reset_index(drop=True)
candidates = overlap[overlap["Candidate_set"]].copy().reset_index(drop=True)
candidates["Display_order_only"] = np.arange(1, len(candidates) + 1)

candidate_cols = [
    "Display_order_only", "Formulation_label", "Alg_mg_ml", "HA_mg_ml", "Pressure_kPa",
    "Pr_mean", "Pr_std", "SR_mean", "SR_std", "Qm_mean_mg_s", "Qm_std_mg_s",
    "Best_measured_angle_deg", "Best_measured_AF", "Best_measured_AF_std",
    "Measured_Pr_criterion_0p95_1p05", "Measured_AF_criterion_1p00_1p25",
    "Source_Qm_SR_feasibility"
]
candidate_table = candidates[candidate_cols].copy()

# Summary by formulation for a region-level description without within-set ranking.
by_form = (
    candidates.groupby(["Formulation_label", "Alg_mg_ml", "HA_mg_ml"], as_index=False)
    .agg(
        N_candidate_conditions=("Pressure_kPa", "size"),
        Pressure_min_kPa=("Pressure_kPa", "min"),
        Pressure_max_kPa=("Pressure_kPa", "max"),
    )
    .sort_values(["Alg_mg_ml", "HA_mg_ml"])
)

summary = pd.DataFrame([
    ["Jointly observed formulation-pressure conditions", len(overlap)],
    ["Conditions with measured Pr available", int(overlap["Pr_mean"].notna().sum())],
    ["Candidate-set conditions (measured Pr 0.95-1.05 AND measured AF 1.00-1.25)", len(candidates)],
    ["Relaxed sensitivity-set conditions (measured Pr 0.90-1.10 AND measured AF 1.00-1.25)", int(overlap["Relaxed_sensitivity_set"].sum())],
    ["Candidate-set conditions also satisfying source Qm>0 and SR<6", int(candidates["Source_Qm_SR_feasibility"].sum())],
    ["Within-set rank calculated", "No"],
    ["Predicted Pr used as a screening gate", "No"],
    ["Predicted AF used as a screening gate", "No"],
], columns=["Item", "Value"])

# Save CSVs.
overlap.to_csv(OUT / "All_Observed_Overlap_Comments_2-R1_6-7-R2.csv", index=False)
candidate_table.to_csv(OUT / "Table4_Unordered_Candidate_Set_Comments_2-R1_6-7-R2.csv", index=False)
by_form.to_csv(OUT / "Candidate_Regions_By_Formulation_Comments_2-R1_6-7-R2.csv", index=False)
summary.to_csv(OUT / "Candidate_Screening_Audit_Comments_2-R1_6-7-R2.csv", index=False)

# Create a single workbook with transparent audit trail.
xlsx = OUT / "Candidate_Screening_Comments_2-R1_6-7-R2.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Audit_Summary", index=False)
    overlap.to_excel(writer, sheet_name="All_Observed_Overlap", index=False)
    candidate_table.to_excel(writer, sheet_name="Candidate_Set", index=False)
    by_form.to_excel(writer, sheet_name="By_Formulation", index=False)

# Style workbook for readability.
wb = load_workbook(xlsx)
header_fill = PatternFill("solid", fgColor="D9EAF7")
header_font = Font(bold=True)
thin_gray = Side(style="thin", color="D9D9D9")
for ws in wb.worksheets:
    ws.freeze_panes = "A2"
    ws.sheet_view.showGridLines = False
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=thin_gray)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="center", wrap_text=True)
    for col_idx, col_cells in enumerate(ws.columns, 1):
        max_len = 0
        for c in col_cells:
            v = "" if c.value is None else str(c.value)
            max_len = max(max_len, min(len(v), 55))
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max(max_len + 2, 10), 38)

# Number formats in candidate sheet.
if "Candidate_Set" in wb.sheetnames:
    ws = wb["Candidate_Set"]
    headers = {c.value: c.column for c in ws[1]}
    for h in ["Pr_mean", "Pr_std", "SR_mean", "SR_std", "Qm_mean_mg_s", "Qm_std_mg_s", "Best_measured_AF", "Best_measured_AF_std"]:
        if h in headers:
            for r in range(2, ws.max_row + 1):
                ws.cell(r, headers[h]).number_format = "0.000"
wb.save(xlsx)

# Candidate map: all jointly observed conditions + unordered measured candidate set.
form_order = (
    overlap[["Formulation_label", "Alg_mg_ml", "HA_mg_ml"]]
    .drop_duplicates()
    .sort_values(["Alg_mg_ml", "HA_mg_ml"])["Formulation_label"]
    .tolist()
)
y_map = {f: i for i, f in enumerate(form_order)}
overlap_plot = overlap.copy()
overlap_plot["Y"] = overlap_plot["Formulation_label"].map(y_map)
cand_plot = overlap_plot[overlap_plot["Candidate_set"]].copy()

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(overlap_plot["Pressure_kPa"], overlap_plot["Y"], s=42, alpha=0.35, label="Jointly observed conditions")
ax.scatter(cand_plot["Pressure_kPa"], cand_plot["Y"], s=95, marker="s", alpha=0.9, label="Unordered candidate set")
ax.set_yticks(np.arange(len(form_order)))
ax.set_yticklabels(form_order)
ax.set_xlabel("Pressure (kPa)")
ax.set_ylabel("Formulation")
ax.set_title("Measured-domain candidate set (no within-set ranking)")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(OUT / "Figure3_Unordered_Candidate_Set_Comments_2-R1_6-7-R2.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "Figure3_Unordered_Candidate_Set_Comments_2-R1_6-7-R2.pdf", bbox_inches="tight")
plt.close(fig)

meta = {
    "comments": ["2-R1", "6-R2", "7-R2"],
    "decision": "Remove model-error-based within-set ranking and remove predicted Pr from screening gate.",
    "candidate_definition": "Jointly observed formulation-pressure conditions with measured Pr in [0.95, 1.05] and at least one measured AF in [1.00, 1.25].",
    "n_jointly_observed": int(len(overlap)),
    "n_candidate_set": int(len(candidates)),
    "n_relaxed_sensitivity_set": int(overlap["Relaxed_sensitivity_set"].sum()),
    "all_candidate_set_meet_source_Qm_gt0_SR_lt6": bool(candidates["Source_Qm_SR_feasibility"].all()),
    "ranking_removed": True,
    "predicted_Pr_gate_removed": True,
    "predicted_AF_gate_removed": True,
    "display_sort": "Formulation and pressure only; Display_order_only is not a scientific rank."
}
(OUT / "Candidate_Screening_Metadata_Comments_2-R1_6-7-R2.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print(summary.to_string(index=False))
print("\nCandidate set:\n")
print(candidate_table.to_string(index=False))
print("\nBy formulation:\n")
print(by_form.to_string(index=False))
print("\nOutputs written to", OUT)
