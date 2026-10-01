from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / 'results' / 'revision_comments_6_7_R1_8_R2' / 'Table_S19_Ablation_Comments_6-7-R1_8-R2.csv'
NEST = ROOT / 'results' / 'revision_comments_4_R1_16_R2' / 'Reportable_Nested_Performance_Comments_4-R1_16-R2.csv'
OUT = ROOT / 'results' / 'revision_comment_6_R1_reconciliation'
OUT.mkdir(parents=True, exist_ok=True)

s19 = pd.read_csv(ABL)
t2 = pd.read_csv(NEST)

rows = []
for _, r in s19.iterrows():
    target = r['Target']
    validation = r['Validation']
    tr = t2.loc[t2['Target'] == target].iloc[0]
    if validation == 'Repeated_KFold':
        table2_r2 = tr['Nested repeated R2 mean']
        table2_selection = tr['Nested repeated selection mode']
    elif validation == 'LOFO':
        table2_r2 = tr['Nested LOFO R2']
        table2_selection = tr['Nested LOFO selection mode']
    else:
        raise ValueError(validation)

    same_algorithm = str(r['Core_model']) == str(r['Engineered_model'])
    rows.append({
        'Target': target,
        'Validation': validation,
        'Table2_nested_R2': table2_r2,
        'Table2_modal_selection': table2_selection,
        'TableS19_core_model': r['Core_model'],
        'TableS19_core_R2': r['Core_R2'],
        'TableS19_engineered_model': r['Engineered_model'],
        'TableS19_engineered_R2': r['Engineered_R2'],
        'Same_algorithm_core_vs_engineered': same_algorithm,
        'Interpretation': (
            'Same algorithm; core-versus-engineered difference can be interpreted as feature-representation sensitivity conditional on this algorithm.'
            if same_algorithm else
            'Different algorithms; the core-versus-engineered difference reflects a combined model-identity and feature-representation change and is not a pure feature-ablation effect.'
        ),
    })

out = pd.DataFrame(rows).sort_values(['Target', 'Validation']).reset_index(drop=True)
out_path = OUT / 'Table_S27_Table2_vs_TableS19_Reconciliation.csv'
out.to_csv(out_path, index=False)

metadata = {
    'comment': '6-R1',
    'purpose': 'Explicitly reconcile Table 2 nested performance with Table S19 representation sensitivity without treating non-equivalent analyses as duplicate estimates.',
    'table2_definition': 'Selection-aware nested outer-test performance; algorithm and feature set are selected only within training data.',
    'tableS19_definition': 'Continuity analysis preserving the submitted S19 core/engineered model identities after duplicate-feature cleanup.',
    'same_algorithm_rows': int(out['Same_algorithm_core_vs_engineered'].sum()),
    'different_algorithm_rows': int((~out['Same_algorithm_core_vs_engineered']).sum()),
    'different_algorithm_comparisons': out.loc[~out['Same_algorithm_core_vs_engineered'], ['Target','Validation','TableS19_core_model','TableS19_engineered_model']].to_dict(orient='records'),
    'interpretation_rule': 'Only same-algorithm S19 rows are interpreted as feature-representation sensitivity. Rows with different algorithms are descriptive continuity comparisons and are not used to attribute performance changes solely to engineered descriptors.',
    'new_model_fitting': False,
}
(OUT / 'Table_S27_Reconciliation_Metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

print(out.to_string(index=False))
print(json.dumps(metadata, indent=2))
