from pathlib import Path
import json
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEST = ROOT / 'results' / 'revision_comments_4_R1_16_R2' / 'Reportable_Nested_Performance_Comments_4-R1_16-R2.csv'
S19 = ROOT / 'results' / 'revision_comments_6_7_R1_8_R2' / 'Table_S19_Ablation_Comments_6-7_R1_8-R2.csv'
# Backward-compatible filename used in the archived package.
if not S19.exists():
    S19 = ROOT / 'results' / 'revision_comments_6_7_R1_8_R2' / 'Table_S19_Ablation_Comments_6-7-R1_8-R2.csv'
S27 = ROOT / 'results' / 'revision_comment_6_R1_reconciliation' / 'Table_S27_Table2_vs_TableS19_Reconciliation.csv'
OUT = ROOT / 'results' / 'revision_comment_8_R2_reconciliation_audit'
OUT.mkdir(parents=True, exist_ok=True)

nested = pd.read_csv(NEST)
s19 = pd.read_csv(S19)
s27 = pd.read_csv(S27)

def expected_table2(row):
    target = row['Target']
    validation = row['Validation']
    tr = nested.loc[nested['Target'] == target].iloc[0]
    if validation == 'Repeated_KFold':
        return float(tr['Nested repeated R2 mean']), str(tr['Nested repeated selection mode'])
    if validation == 'LOFO':
        return float(tr['Nested LOFO R2']), str(tr['Nested LOFO selection mode'])
    raise ValueError(validation)

checks = []
for _, row in s27.iterrows():
    target, validation = row['Target'], row['Validation']
    sr = s19.loc[(s19['Target'] == target) & (s19['Validation'] == validation)].iloc[0]
    t2_r2, t2_sel = expected_table2(row)
    tests = {
        'table2_r2_match': abs(float(row['Table2_nested_R2']) - t2_r2) < 1e-12,
        'table2_selection_match': str(row['Table2_modal_selection']) == t2_sel,
        's19_core_model_match': str(row['TableS19_core_model']) == str(sr['Core_model']),
        's19_core_r2_match': abs(float(row['TableS19_core_R2']) - float(sr['Core_R2'])) < 1e-12,
        's19_engineered_model_match': str(row['TableS19_engineered_model']) == str(sr['Engineered_model']),
        's19_engineered_r2_match': abs(float(row['TableS19_engineered_R2']) - float(sr['Engineered_R2'])) < 1e-12,
        'same_algorithm_flag_match': bool(row['Same_algorithm_core_vs_engineered']) == (str(sr['Core_model']) == str(sr['Engineered_model'])),
    }
    checks.append({'Target': target, 'Validation': validation, **tests, 'row_pass': all(tests.values())})

check_df = pd.DataFrame(checks)
check_df.to_csv(OUT / 'Comment8_R2_Rowwise_Reconciliation_Audit.csv', index=False)
summary = {
    'comment': '8-R2',
    'purpose': 'Consistency audit of the already-generated Table S27 reconciliation; no model fitting is performed.',
    'rows_checked': int(len(check_df)),
    'rows_passed': int(check_df['row_pass'].sum()),
    'mismatches': int((~check_df['row_pass']).sum()),
    'same_algorithm_rows': int(s27['Same_algorithm_core_vs_engineered'].sum()),
    'different_algorithm_rows': int((~s27['Same_algorithm_core_vs_engineered']).sum()),
    'status': 'PASS' if bool(check_df['row_pass'].all()) else 'FAIL',
    'table_s27_sha256': hashlib.sha256(S27.read_bytes()).hexdigest(),
    'interpretation': 'Table 2 and Table S19 are non-equivalent analyses. Six S19 rows hold the algorithm constant; Pr-LOFO and SR-repeated do not and are not interpreted as pure feature effects.',
    'new_model_fitting': False,
}
(OUT / 'Comment8_R2_Consistency_Audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(check_df.to_string(index=False))
print(json.dumps(summary, indent=2))
if summary['status'] != 'PASS':
    raise SystemExit(1)
