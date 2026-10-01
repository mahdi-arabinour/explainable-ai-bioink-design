from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results' / 'revision_comments_2_R1_6_7_R2'
OUT = ROOT / 'results' / 'revision_comment_7_R2_pr_gate_sensitivity'
OUT.mkdir(parents=True, exist_ok=True)

src = BASE / 'All_Observed_Overlap_Comments_2-R1_6-7-R2.csv'
if not src.exists():
    raise FileNotFoundError(
        f'{src} not found. Run src/revision_comments_2_R1_6_7_R2_candidate_set.py first.'
    )

df = pd.read_csv(src)
required = {
    'Candidate_set', 'Relaxed_sensitivity_set', 'Formulation_label', 'Pressure_kPa',
    'Pr_mean', 'Pr_std', 'Best_measured_angle_deg', 'Best_measured_AF',
    'Best_measured_AF_std', 'Source_Qm_SR_feasibility'
}
missing = required.difference(df.columns)
if missing:
    raise ValueError(f'Missing expected columns: {sorted(missing)}')

strict = df[df['Candidate_set']].copy()
relaxed = df[df['Relaxed_sensitivity_set']].copy()
relaxed_only = relaxed[~relaxed['Candidate_set']].copy()

summary = pd.DataFrame([
    {
        'Screening_definition': 'Main strict measured-domain set',
        'Measured_Pr_interval': '0.95-1.05',
        'Measured_AF_requirement': 'At least one measured AF in 1.00-1.25',
        'N_conditions': int(len(strict)),
        'Predicted_Pr_used': 'No',
        'Interpretation': 'Primary conservative operational set; not a validated acceptance boundary.'
    },
    {
        'Screening_definition': 'Relaxed measured-Pr sensitivity set',
        'Measured_Pr_interval': '0.90-1.10',
        'Measured_AF_requirement': 'At least one measured AF in 1.00-1.25',
        'N_conditions': int(len(relaxed)),
        'Predicted_Pr_used': 'No',
        'Interpretation': 'Threshold-sensitivity check only; not used to redefine the main set.'
    },
])
summary.to_csv(OUT / 'Pr_Gate_Sensitivity_Summary_Comment_7_R2.csv', index=False)

cols = [
    'Formulation_label', 'Pressure_kPa', 'Pr_mean', 'Pr_std',
    'Best_measured_angle_deg', 'Best_measured_AF', 'Best_measured_AF_std',
    'Source_Qm_SR_feasibility'
]
relaxed_only = relaxed_only[cols].sort_values(['Formulation_label','Pressure_kPa']).reset_index(drop=True)
relaxed_only.to_csv(OUT / 'Table_S31_Relaxed_Pr_Only_Conditions_Comment_7_R2.csv', index=False)

meta = {
    'comment': '7-R2',
    'purpose': 'Demonstrate that predicted Pr is not used for candidate eligibility and quantify sensitivity of measured-domain membership to the operational Pr tolerance.',
    'n_jointly_observed': int(len(df)),
    'n_strict_0p95_1p05': int(len(strict)),
    'n_relaxed_0p90_1p10': int(len(relaxed)),
    'n_added_by_relaxation': int(len(relaxed_only)),
    'predicted_Pr_used_in_final_gate': False,
    'predicted_AF_used_in_final_gate': False,
    'main_set_redefined_by_sensitivity': False,
    'note': 'The strict 0.95-1.05 measured-Pr interval is treated as a conservative operational screening definition, not as a model-derived or externally validated acceptance boundary.'
}
(OUT / 'Pr_Gate_Sensitivity_Metadata_Comment_7_R2.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

print(summary.to_string(index=False))
print('\nRelaxed-only conditions (added by 0.90-1.10 Pr sensitivity):')
print(relaxed_only.to_string(index=False))
print('\nOutputs written to', OUT)
