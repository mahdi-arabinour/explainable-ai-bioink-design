from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'revision_comments_4_R1_16_R2'
SUP = ROOT / 'supplementary' / 'final_revision' / 'tables'
SUP.mkdir(parents=True, exist_ok=True)

fold_path = OUT / 'Repeated_Outer_Fold_Metrics_Comments_4-R1_16-R2.csv'
repeat_path = OUT / 'Repeated_Repeat_Metrics_Comments_4-R1_16-R2.csv'

fold = pd.read_csv(fold_path)
repeat = pd.read_csv(repeat_path)

rows = []
for target in sorted(fold['Target'].unique()):
    fg = fold.loc[fold['Target'] == target].copy()
    rg = repeat.loc[repeat['Target'] == target].copy()
    r2 = fg['R2'].dropna()
    rows.append({
        'Target': target,
        'N_outer_folds': int(r2.shape[0]),
        'Mean_outer_fold_R2': float(r2.mean()),
        'Median_outer_fold_R2': float(r2.median()),
        'Positive_outer_fold_fraction': float((r2 > 0).mean()),
        'Outer_fold_R2_min': float(r2.min()),
        'Outer_fold_R2_max': float(r2.max()),
        'N_outer_repetitions': int(rg.shape[0]),
        'Repeat_pooled_OOF_R2_mean': float(rg['R2'].mean()),
        'Repeat_pooled_OOF_R2_SD': float(rg['R2'].std(ddof=1)),
        'Repeat_pooled_OOF_R2_min': float(rg['R2'].min()),
        'Repeat_pooled_OOF_R2_max': float(rg['R2'].max()),
    })

summary = pd.DataFrame(rows)
order = pd.Categorical(summary['Target'], categories=['Pr','SR','Qm','AF'], ordered=True)
summary = summary.assign(_order=order).sort_values('_order').drop(columns=['_order'])

out_csv = OUT / 'R2_Aggregation_Summary_Comments_4-R1_16-R2.csv'
sup_csv = SUP / 'Table_S26_R2_Aggregation_Summary_Comments_4-R1_16-R2.csv'
summary.to_csv(out_csv, index=False)
summary.to_csv(sup_csv, index=False)

print(summary.to_string(index=False))
print(f'\nWrote: {out_csv}')
print(f'Wrote: {sup_csv}')
