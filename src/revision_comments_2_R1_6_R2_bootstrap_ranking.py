from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'results' / 'revision_comments_2_R1_6_R2_bootstrap_ranking'
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926
N_BOOT = 1000
N_JOBS = 4

CORE = ['Alg_mg_ml','HA_mg_ml','Pressure_kPa']
ENG = ['TP','HF','Alg_HA_ratio_safe','P_over_TP','P_over_Alg','P_over_HA','Alg_x_HA','Alg_x_P','HA_x_P','TP_x_P','HF_x_P']
ANGLE = ['Angle_deg','P_x_Angle','Alg_x_Angle','HA_x_Angle','TP_x_Angle']
AF_FEATURES = CORE + ENG + ANGLE


def add_features(df):
    d = df.copy()
    d['TP'] = d['Alg_mg_ml'] + d['HA_mg_ml']
    d['HF'] = d['HA_mg_ml'] / d['TP']
    d['Alg_HA_ratio_safe'] = d['Alg_mg_ml'] / (d['HA_mg_ml'] + 1e-9)
    d['P_over_TP'] = d['Pressure_kPa'] / d['TP']
    d['P_over_Alg'] = d['Pressure_kPa'] / (d['Alg_mg_ml'] + 1e-9)
    d['P_over_HA'] = d['Pressure_kPa'] / (d['HA_mg_ml'] + 1e-9)
    d['Alg_x_HA'] = d['Alg_mg_ml'] * d['HA_mg_ml']
    d['Alg_x_P'] = d['Alg_mg_ml'] * d['Pressure_kPa']
    d['HA_x_P'] = d['HA_mg_ml'] * d['Pressure_kPa']
    d['TP_x_P'] = d['TP'] * d['Pressure_kPa']
    d['HF_x_P'] = d['HF'] * d['Pressure_kPa']
    d['P_x_Angle'] = d['Pressure_kPa'] * d['Angle_deg']
    d['Alg_x_Angle'] = d['Alg_mg_ml'] * d['Angle_deg']
    d['HA_x_Angle'] = d['HA_mg_ml'] * d['Angle_deg']
    d['TP_x_Angle'] = d['TP'] * d['Angle_deg']
    return d


def af_model(seed):
    return XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=2,
        subsample=0.9, colsample_bytree=0.9,
        objective='reg:squarederror', random_state=seed, n_jobs=1, verbosity=0
    )

pr = pd.read_csv(DATA / 'modeling_dataset_Pr.csv')
af = pd.read_csv(DATA / 'modeling_dataset_AF.csv')
legacy = pd.read_csv(ROOT / 'archive' / 'legacy_pre_revision_candidate_ranking' / 'tables' / '029_tables_final_top_strict_design_candidates.csv')
cand = add_features(legacy.rename(columns={'Best_angle_deg':'Angle_deg'}))

# Use the duplicate-reduced fixed reference models retained for downstream interpretation.
# Performance claims remain based on the separately reported nested validation.
pr_ref = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
af_ref = af_model(42)
pr_ref.fit(pr[CORE], pr['Pr_mean'])
af_ref.fit(af[AF_FEATURES], af['AF_mean'])

cand['Reference_pred_Pr'] = pr_ref.predict(cand[CORE])
cand['Reference_pred_AF'] = af_ref.predict(cand[AF_FEATURES])
cand['Reference_combined_error'] = (
    (cand['Reference_pred_Pr'] - 1.0).abs() +
    (cand['Reference_pred_AF'] - 1.0).abs()
)

forms = sorted(pr['Formulation_label'].astype(str).unique())
assert forms == sorted(af['Formulation_label'].astype(str).unique())


def one_bootstrap(b):
    rng = np.random.default_rng(SEED + b)
    sampled = rng.choice(forms, size=len(forms), replace=True)
    pr_b = pd.concat([pr[pr['Formulation_label'].astype(str) == f] for f in sampled], ignore_index=True)
    af_b = pd.concat([af[af['Formulation_label'].astype(str) == f] for f in sampled], ignore_index=True)
    m_pr = clone(pr_ref)
    m_af = af_model(SEED + b)
    m_pr.fit(pr_b[CORE], pr_b['Pr_mean'])
    m_af.fit(af_b[AF_FEATURES], af_b['AF_mean'])
    p_pr = m_pr.predict(cand[CORE])
    p_af = m_af.predict(cand[AF_FEATURES])
    e = np.abs(p_pr - 1.0) + np.abs(p_af - 1.0)
    return e, p_pr, p_af

res = Parallel(n_jobs=N_JOBS, backend='loky')(
    delayed(one_bootstrap)(b) for b in range(N_BOOT)
)
E = np.stack([r[0] for r in res])
P_PR = np.stack([r[1] for r in res])
P_AF = np.stack([r[2] for r in res])

order = np.argsort(E, axis=1)
ranks = np.empty_like(order)
ranks[np.arange(N_BOOT)[:, None], order] = np.arange(1, E.shape[1] + 1)
q = np.quantile(E, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975], axis=0)

summary = cand[['Formulation_label','Alg_mg_ml','HA_mg_ml','Pressure_kPa','Angle_deg',
                'Reference_pred_Pr','Reference_pred_AF','Reference_combined_error']].copy()
for name, arr in zip(
    ['E_q2p5','E_q5','E_q25','E_median','E_q75','E_q95','E_q97p5'], q
):
    summary[name] = arr
summary['Bootstrap_P_top'] = (ranks == 1).mean(axis=0)
summary['Bootstrap_P_top3'] = (ranks <= 3).mean(axis=0)
summary['Bootstrap_P_top5'] = (ranks <= 5).mean(axis=0)
summary['Bootstrap_mean_rank'] = ranks.mean(axis=0)
summary = summary.sort_values(['E_median','Reference_combined_error']).reset_index(drop=True)

full_best_idx = int(np.argmin(cand['Reference_combined_error'].to_numpy()))
pair_rows = []
for j in range(len(cand)):
    if j == full_best_idx:
        continue
    diff = E[:, j] - E[:, full_best_idx]  # positive means legacy top has lower score
    lo, med, hi = np.quantile(diff, [0.025, 0.5, 0.975])
    pair_rows.append({
        'Reference_best_formulation': cand.iloc[full_best_idx]['Formulation_label'],
        'Reference_best_pressure_kPa': cand.iloc[full_best_idx]['Pressure_kPa'],
        'Comparator_formulation': cand.iloc[j]['Formulation_label'],
        'Comparator_pressure_kPa': cand.iloc[j]['Pressure_kPa'],
        'DeltaE_comparator_minus_best_q2p5': lo,
        'DeltaE_comparator_minus_best_median': med,
        'DeltaE_comparator_minus_best_q97p5': hi,
        'P_reference_best_lower_error': float((diff > 0).mean()),
        'Statistically_separated_at_95pct': bool(lo > 0 or hi < 0),
    })
pairwise = pd.DataFrame(pair_rows)

summary.to_csv(OUT / 'Bootstrap_Candidate_Ranking_Summary.csv', index=False)
pairwise.to_csv(OUT / 'Bootstrap_Pairwise_Differences_vs_Original_Top.csv', index=False)
np.savez_compressed(OUT / 'Bootstrap_Candidate_Ranking_Draws.npz', E=E, Pr=P_PR, AF=P_AF)

best_row = summary.loc[summary['Formulation_label'].eq('1ALG8HA') & summary['Pressure_kPa'].eq(65)].iloc[0]
meta = {
    'purpose': 'Reviewer-suggested uncertainty analysis of the originally submitted within-set ranking.',
    'bootstrap_unit': 'polymer formulation cluster',
    'n_formulations_per_bootstrap': len(forms),
    'n_bootstrap_replicates': N_BOOT,
    'reference_Pr_model': 'LinearRegression / core (duplicate-reduced fixed reference model)',
    'reference_AF_model': 'XGBoost / angle_engineered (duplicate-reduced fixed reference model)',
    'ranking_score': '|Pr_pred - 1| + |AF_pred - 1|',
    'original_top': '1ALG8HA / 65 kPa / 120 degrees',
    'original_top_bootstrap_P_top': float(best_row['Bootstrap_P_top']),
    'original_top_bootstrap_P_top3': float(best_row['Bootstrap_P_top3']),
    'n_pairwise_95pct_separations_vs_original_top': int(pairwise['Statistically_separated_at_95pct'].sum()),
    'interpretation': 'The original top candidate is the modal bootstrap leader, but no pairwise 95% bootstrap difference interval excludes zero; a unique top rank is therefore not statistically resolved.'
}
(OUT / 'Bootstrap_Candidate_Ranking_Metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

print(summary.to_string(index=False))
print('\nPairwise 95% separations:', pairwise['Statistically_separated_at_95pct'].sum(), 'of', len(pairwise))
print(json.dumps(meta, indent=2))
