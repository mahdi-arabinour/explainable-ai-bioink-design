#!/usr/bin/env python3
"""Normalize the publication-facing SI table package after all scientific audits.

Legacy results/ filenames are intentionally unchanged. This script operates only
on supplementary/final_revision/tables after a successful clean master run.
"""
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "supplementary" / "final_revision" / "tables"

# Old S19-S31 -> S16-S28; Tables S6-S8 are redundant with old S21.
RENAMES = {
    "Table_S19_Ablation.xlsx": "Table_S16_Ablation.xlsx",
    "Table_S20_Baselines.xlsx": "Table_S17_Baselines.xlsx",
    "Table_S22_XAI.xlsx": "Table_S19_XAI.xlsx",
    "Table_S23_LOFO_Diagnostics.xlsx": "Table_S20_LOFO_Diagnostics.xlsx",
    "Table_S25_Bootstrap_Ranking_Stability.csv": "Table_S22_Bootstrap_Ranking_Stability.csv",
    "Table_S26_R2_Aggregation_Summary_Comments_4-R1_16-R2.csv": "Table_S23_R2_Aggregation.csv",
    "Table_S28_Dimensionless_Stress_Ratio_Sensitivity.csv": "Table_S25_Dimensionless_Stress_Ratio_Sensitivity.csv",
    "Table_S29_LOFO_Formulation_Block_Uncertainty.csv": "Table_S26_LOFO_Formulation_Block_Uncertainty.csv",
    "Table_S30_Baseline_vs_Nested_Apples_to_Apples_Comment_3-R2.csv": "Table_S27_Baseline_Comparison.csv",
    "Table_S31_Relaxed_Pr_Only_Conditions_Comment_7_R2.csv": "Table_S28_Pr_Threshold_Sensitivity.csv",
}

RECORD_COLUMNS = [
    ("Target", "Target"),
    ("Nested repeated R2 mean", "Nested_repeated_R2_mean"),
    ("Nested repeated R2 SD", "Nested_repeated_R2_SD"),
    ("Nested repeated MAE mean", "Repeated_MAE"),
    ("Nested repeated RMSE mean", "Repeated_RMSE"),
    ("Nested repeated Spearman mean", "Repeated_Spearman_rho"),
    ("Nested repeated selection mode", "Most_frequent_repeated_selection"),
    ("Repeated selection frequency", "Repeated_selection_frequency"),
    ("Nested LOFO R2", "Nested_LOFO_R2"),
    ("Nested LOFO MAE", "LOFO_MAE"),
    ("Nested LOFO RMSE", "LOFO_RMSE"),
    ("Nested LOFO Spearman", "LOFO_Spearman_rho"),
    ("Nested LOFO selection mode", "Most_frequent_LOFO_selection"),
    ("LOFO selection frequency", "LOFO_selection_frequency"),
]

def run():
    rep = pd.read_csv(
        ROOT / "results" / "revision_comments_4_R1_16_R2"
        / "Reportable_Nested_Performance_Comments_4-R1_16-R2.csv"
    )
    historical = TAB / "Table_S6-S8_Nested_Performance.xlsx"
    old_s21 = TAB / "Table_S21_Nested_CV.xlsx"
    if not historical.exists() or not old_s21.exists():
        raise FileNotFoundError("Expected old SI consolidation files for the renumbering guard")
    s6 = pd.read_excel(historical, sheet_name="Table S6")
    if set(s6.Target) != set(rep.Target) or len(s6) != 4 or len(rep) != 4:
        raise AssertionError("Unexpected repeated-validation target set")
    for _, row in rep.iterrows():
        sd = float(s6.loc[s6.Target.eq(row.Target), "R2 SD"].iloc[0])
        if abs(sd - float(row["Nested repeated R2 SD"])) > 1e-12:
            raise AssertionError("Unique R2 SD values cannot be reconciled")
    tab18 = rep[[old for old, new in RECORD_COLUMNS]].copy()
    tab18.columns = [new for old, new in RECORD_COLUMNS]
    tab18.to_csv(TAB / "Table_S18_Nested_CV_With_R2_SD.csv", index=False)

    old_recon = TAB / "Table_S27_Table2_vs_TableS19_Reconciliation.csv"
    if not old_recon.exists():
        raise FileNotFoundError(old_recon)
    content = old_recon.read_text(encoding="utf-8")
    (TAB / "Table_S24_Table2_vs_TableS16_Reconciliation.csv").write_text(
        content.replace("TableS19_", "TableS16_"), encoding="utf-8"
    )

    for previous, current in RENAMES.items():
        src = TAB / previous
        dst = TAB / current
        if not src.exists():
            raise FileNotFoundError(src)
        src.replace(dst)
        if dst.suffix == ".xlsx":
            # Update presentation-facing sheet labels only; numeric data is untouched.
            previous_number = int(previous.split("_")[1][1:])
            current_number = int(current.split("_")[1][1:])
            wb = load_workbook(dst)
            for ws in wb.worksheets:
                ws.title = ws.title.replace(
                    f"Table S{previous_number}", f"Table S{current_number}"
                ).replace(
                    f"Table_S{previous_number}", f"Table_S{current_number}"
                )
            wb.save(dst)

    # Old S21 workbook is superseded by complete S18 CSV (including the four SDs).
    old_s21.unlink()
    old_recon.unlink()
    historical.unlink()

    old_prov = TAB / "Table_S24_Provenance.csv"
    if old_prov.exists():
        old_prov.replace(TAB / "Table_S21_Provenance.csv")

    expected = {p for p in [*RENAMES.values(),
                            "Table_S18_Nested_CV_With_R2_SD.csv",
                            "Table_S24_Table2_vs_TableS16_Reconciliation.csv"]}
    existing = {p.name for p in TAB.iterdir() if p.is_file()}
    if not expected.issubset(existing):
        raise AssertionError("Published SI table files not complete")
    if historical.exists() or old_s21.exists() or old_recon.exists():
        raise AssertionError("Superseded SI table packages remain active")
    print("PASS: SI table range S16-S28, with S18 preserving all unique SD values")


if __name__ == "__main__":
    run()
