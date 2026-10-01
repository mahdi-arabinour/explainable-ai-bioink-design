from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'results' / 'revision_comment_10_R1_dimensionless_stress_ratio'
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(np.unique(y_true)) <= 1 or len(np.unique(y_pred)) <= 1:
        return np.nan
    return float(spearmanr(y_true, y_pred).correlation)


def calc_metrics(y_true, y_pred):
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(mean_squared_error(y_true, y_pred) ** 0.5),
        'Spearman': safe_spearman(y_true, y_pred),
    }


class PowerLawRegressor(BaseEstimator, RegressorMixin):
    """Positive power-law baseline y = a*x^b fit by OLS in log-log space."""
    def fit(self, X, y):
        x = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError('PowerLawRegressor requires strictly positive x and y.')
        self.model_ = LinearRegression().fit(np.log(x).reshape(-1, 1), np.log(y))
        self.a_ = float(np.exp(self.model_.intercept_))
        self.b_ = float(self.model_.coef_[0])
        return self

    def predict(self, X):
        x = np.asarray(X, dtype=float).reshape(-1)
        return self.a_ * np.power(x, self.b_)


def repeated_cv(df, feature, target, estimator, label, n_splits=5, n_repeats=20):
    X = df[[feature]].copy()
    y = df[target].astype(float).copy()
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    rows = []
    for split_id, (tr, te) in enumerate(cv.split(X), start=1):
        m = clone(estimator)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        met = calc_metrics(y.iloc[te], pred)
        met.update({'Baseline': label, 'Split': split_id, 'N_test': len(te)})
        rows.append(met)
    split_df = pd.DataFrame(rows)
    summary = {
        'Baseline': label,
        'Validation': 'Repeated 5-fold CV (20 repeats)',
        'N': len(df),
        'Predictor': feature,
        'R2': split_df['R2'].mean(),
        'R2_SD': split_df['R2'].std(ddof=1),
        'MAE': split_df['MAE'].mean(),
        'RMSE': split_df['RMSE'].mean(),
        'Spearman': split_df['Spearman'].mean(),
    }
    return summary, split_df


def lofo(df, feature, target, estimator, label):
    X = df[[feature]].copy()
    y = df[target].astype(float).copy()
    groups = df['Formulation_label'].astype(str)
    pred_rows, group_rows = [], []
    for group in sorted(groups.unique()):
        train = groups != group
        test = groups == group
        m = clone(estimator)
        m.fit(X.loc[train], y.loc[train])
        pred = m.predict(X.loc[test])
        yt = y.loc[test]
        for idx, obs, pr in zip(yt.index, yt.values, pred):
            pred_rows.append({
                'Baseline': label,
                'Held_out_formulation': group,
                'Index': int(idx),
                'Observed': float(obs),
                'Predicted': float(pr),
            })
        met = calc_metrics(yt, pred)
        met.update({'Baseline': label, 'Held_out_formulation': group, 'N_test': int(test.sum())})
        group_rows.append(met)
    pred_df = pd.DataFrame(pred_rows)
    group_df = pd.DataFrame(group_rows)
    met = calc_metrics(pred_df['Observed'], pred_df['Predicted'])
    summary = {
        'Baseline': label,
        'Validation': 'LOFO (pooled predictions)',
        'N': len(df),
        'Predictor': feature,
        'R2': met['R2'],
        'R2_SD': np.nan,
        'MAE': met['MAE'],
        'RMSE': met['RMSE'],
        'Spearman': met['Spearman'],
    }
    return summary, pred_df, group_df


def add_dimensionless_stress_ratio(df):
    d = df.copy()
    # Source printing pressure is in kPa and the source oscillatory yield stress is in Pa.
    # Pressure/yield-stress is therefore dimensionless. Under the fixed nozzle geometry
    # used throughout the source experiments, a nominal capillary wall-stress/yield-stress
    # ratio differs only by a constant geometry factor, so this variable preserves the same
    # ordering without introducing undocumented assumptions about the internal conical profile.
    d['P_over_Yield_dimless'] = d['Pressure_kPa'] * 1000.0 / d['Yield_stress_mean_Pa']
    return d

qm = add_dimensionless_stress_ratio(
    pd.read_csv(DATA / 'modeling_dataset_Qm.csv').dropna(
        subset=['Qm_mean_mg_s','P_over_TP','Pressure_kPa','Yield_stress_mean_Pa']
    ).reset_index(drop=True)
)
sr = add_dimensionless_stress_ratio(
    pd.read_csv(DATA / 'modeling_dataset_SR.csv').dropna(
        subset=['SR_mean','P_over_TP','Pressure_kPa','Yield_stress_mean_Pa']
    ).reset_index(drop=True)
)

jobs = []
for target_name, df, target_col in [('Qm', qm, 'Qm_mean_mg_s'), ('SR', sr, 'SR_mean')]:
    for feature, descriptor_label in [
        ('P_over_TP', 'P/TP'),
        ('P_over_Yield_dimless', 'P/yield-stress dimensionless ratio'),
    ]:
        jobs.extend([
            (target_name, df, feature, target_col, LinearRegression(), f'{target_name}: {descriptor_label} linear'),
            (target_name, df, feature, target_col, PowerLawRegressor(), f'{target_name}: {descriptor_label} power law'),
        ])

summary_rows, repeated_parts, lofo_pred_parts, lofo_group_parts = [], [], [], []
for target_name, df, feature, target, estimator, label in jobs:
    rep_sum, rep_splits = repeated_cv(df, feature, target, estimator, label)
    rep_sum['Target'] = target_name
    summary_rows.append(rep_sum)
    rep_splits['Target'] = target_name
    rep_splits['Predictor'] = feature
    repeated_parts.append(rep_splits)

    lofo_sum, lofo_pred, lofo_group = lofo(df, feature, target, estimator, label)
    lofo_sum['Target'] = target_name
    summary_rows.append(lofo_sum)
    lofo_pred['Target'] = target_name
    lofo_pred['Predictor'] = feature
    lofo_group['Target'] = target_name
    lofo_group['Predictor'] = feature
    lofo_pred_parts.append(lofo_pred)
    lofo_group_parts.append(lofo_group)

summary = pd.DataFrame(summary_rows)[
    ['Target','Baseline','Validation','Predictor','N','R2','R2_SD','MAE','RMSE','Spearman']
]
repeated_splits = pd.concat(repeated_parts, ignore_index=True)
lofo_predictions = pd.concat(lofo_pred_parts, ignore_index=True)
lofo_groups = pd.concat(lofo_group_parts, ignore_index=True)

corr_rows=[]
for target_name, df, target_col in [('Qm', qm, 'Qm_mean_mg_s'), ('SR', sr, 'SR_mean')]:
    for feature in ['P_over_TP','P_over_Yield_dimless']:
        rho,p=spearmanr(df[feature],df[target_col])
        corr_rows.append({'Target':target_name,'Predictor':feature,'Spearman_rho':rho,'p_value':p,'N':len(df)})
corr=pd.DataFrame(corr_rows)

condition_cols=['Formulation_label','Pressure_kPa','P_over_TP','Yield_stress_mean_Pa','P_over_Yield_dimless']
cond=pd.concat([
    qm[condition_cols].assign(Target_dataset='Qm'),
    sr[condition_cols].assign(Target_dataset='SR')
],ignore_index=True)

summary.to_csv(OUT/'Dimensionless_Stress_Ratio_Baseline_Summary.csv',index=False)
repeated_splits.to_csv(OUT/'Dimensionless_Stress_Ratio_Repeated_Fold_Metrics.csv',index=False)
lofo_predictions.to_csv(OUT/'Dimensionless_Stress_Ratio_LOFO_Predictions.csv',index=False)
lofo_groups.to_csv(OUT/'Dimensionless_Stress_Ratio_LOFO_Group_Metrics.csv',index=False)
corr.to_csv(OUT/'Dimensionless_Stress_Ratio_Correlation_Summary.csv',index=False)
cond.to_csv(OUT/'Dimensionless_Stress_Ratio_Condition_Table.csv',index=False)

metadata={
    'comment':'Reviewer 1 Comment 10',
    'purpose':'Test a defensible dimensionless stress-scale alternative before demoting P/TP.',
    'dimensionless_descriptor':'P_over_Yield_dimless = Pressure_kPa*1000 / Yield_stress_mean_Pa',
    'physical_interpretation':'Driving-pressure/yield-stress ratio. With fixed nozzle geometry across source experiments, a nominal capillary wall-stress/yield-stress ratio is a constant multiple of this descriptor.',
    'caveat':'Not labeled as an exact in-nozzle wall shear stress because the archived dataset does not specify the internal conical radius-versus-length profile or the fraction of pneumatic pressure lost upstream of the nozzle.',
    'cross_model_note':'Cross parameters are archived, but a defensible shear-rate-based Cross number would require additional flow/geometry assumptions not present in the curated source tables; no such assumptions were introduced.',
    'validation':'Repeated 5-fold CV (20 repeats) and pooled LOFO, matching the simple-baseline framework used elsewhere in the revision.',
    'random_seed':RANDOM_STATE,
}
(OUT/'Dimensionless_Stress_Ratio_Metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

print('\nSUMMARY')
print(summary.to_string(index=False))
print('\nCORRELATIONS')
print(corr.to_string(index=False))

# Targeted multivariable substitution sensitivity.
# This uses the same duplicate-reduced engineered feature representation and SVR-RBF
# model used in the S19 engineered-side Qm and SR comparisons. P/TP is replaced one-for-one
# by the dimensionless P/yield-stress descriptor; all other predictors and validation splits
# are unchanged. It is a representation-sensitivity control, not a nested performance estimate.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneGroupOut, cross_validate, cross_val_predict
from sklearn.metrics import make_scorer

CORE = ['Alg_mg_ml','HA_mg_ml','Pressure_kPa']
GEN = ['TP','HF','Alg_HA_ratio_safe','P_over_TP','P_over_Alg','P_over_HA','Alg_x_P','HA_x_P','TP_x_P','HF_x_P','Alg_x_HA']

def svr_model():
    return Pipeline([('scaler',StandardScaler()),('model',SVR(kernel='rbf',C=10.0,epsilon=0.05))])

def sp_score(y_true,y_pred):
    v=spearmanr(y_true,y_pred).correlation
    return 0.0 if pd.isna(v) else float(v)

SCORING={'r2':'r2','mae':'neg_mean_absolute_error','rmse':'neg_root_mean_squared_error','spearman':make_scorer(sp_score)}
sub_rows=[]
for target_name, df, target_col in [('Qm',qm,'Qm_mean_mg_s'),('SR',sr,'SR_mean')]:
    for descriptor in ['P_over_TP','P_over_Yield_dimless']:
        features = CORE + [descriptor if f=='P_over_TP' else f for f in GEN]
        X=df[features]; y=df[target_col]; groups=df['Formulation_label'].astype(str)
        z=cross_validate(
            svr_model(),X,y,
            cv=RepeatedKFold(n_splits=5,n_repeats=20,random_state=RANDOM_STATE),
            scoring=SCORING,n_jobs=-1,pre_dispatch='2*n_jobs'
        )
        sub_rows.append({
            'Target':target_name,'Descriptor':descriptor,'Model':'SVR_RBF','Validation':'Repeated_KFold',
            'R2':float(z['test_r2'].mean()),'MAE':float(-z['test_mae'].mean()),
            'RMSE':float(-z['test_rmse'].mean()),'Spearman':float(z['test_spearman'].mean()),
            'N_features':len(features),'Feature_list':', '.join(features)
        })
        pred=cross_val_predict(
            svr_model(),X,y,
            cv=LeaveOneGroupOut().split(X,y,groups),n_jobs=-1,pre_dispatch='2*n_jobs'
        )
        met=calc_metrics(y,pred)
        sub_rows.append({
            'Target':target_name,'Descriptor':descriptor,'Model':'SVR_RBF','Validation':'LOFO',
            'R2':met['R2'],'MAE':met['MAE'],'RMSE':met['RMSE'],'Spearman':met['Spearman'],
            'N_features':len(features),'Feature_list':', '.join(features)
        })
sub=pd.DataFrame(sub_rows)
sub.to_csv(OUT/'Dimensionless_Stress_Ratio_Multivariable_Substitution.csv',index=False)
print('\nMULTIVARIABLE SUBSTITUTION')
print(sub[['Target','Descriptor','Validation','R2','MAE','RMSE','Spearman']].to_string(index=False))

# Compact manuscript/SI table: one row per target/descriptor/model with repeated and LOFO R2/RMSE.
def compact_pair(summary_df, target, predictor, model_phrase):
    r=summary_df[(summary_df.Target==target)&(summary_df.Predictor==predictor)&(summary_df.Baseline.str.contains(model_phrase,regex=False))]
    rep=r[r.Validation.str.startswith('Repeated')].iloc[0]
    lo=r[r.Validation.str.startswith('LOFO')].iloc[0]
    return {
        'Target':target,
        'Descriptor':'P/TP' if predictor=='P_over_TP' else 'P_applied / yield stress',
        'Model':'Linear' if model_phrase==' linear' else 'Power law',
        'Repeated_R2':rep.R2,'Repeated_RMSE':rep.RMSE,
        'LOFO_R2':lo.R2,'LOFO_RMSE':lo.RMSE,
    }
compact=[]
for target in ['Qm','SR']:
    for predictor in ['P_over_TP','P_over_Yield_dimless']:
        compact.append(compact_pair(summary,target,predictor,' linear'))
        compact.append(compact_pair(summary,target,predictor,' power law'))
compact_df=pd.DataFrame(compact)
compact_df.to_csv(OUT/'Table_S28_Single_Descriptor_Summary.csv',index=False)
final_table_dir=ROOT/'supplementary'/'final_revision'/'tables'
final_table_dir.mkdir(parents=True,exist_ok=True)
compact_df.to_csv(final_table_dir/'Table_S28_Dimensionless_Stress_Ratio_Sensitivity.csv',index=False)

metadata_path=OUT/'Dimensionless_Stress_Ratio_Metadata.json'
meta=json.loads(metadata_path.read_text(encoding='utf-8'))
meta.update({
    'multivariable_substitution':'Fixed SVR-RBF engineered representation sensitivity: P/TP replaced one-for-one by P/yield-stress; all other features and validation schemes unchanged.',
    'key_Qm_single_descriptor':'Power-law P/yield R2 = -0.254 repeated and -0.179 LOFO versus P/TP = -0.490 and -0.455; improvement remained below zero.',
    'key_Qm_multivariable_substitution':'P/TP engineered SVR R2 = 0.697 repeated and -0.558 LOFO versus P/yield = 0.674 and -0.880.',
    'key_SR_multivariable_substitution':'P/TP engineered SVR R2 = 0.055 repeated and 0.390 LOFO versus P/yield = 0.027 and 0.196.',
    'conclusion':'The dimensionless stress-scale descriptor does not support a transferable one-parameter physical rule and does not outperform P/TP when substituted into the duplicate-reduced engineered representation. P/TP is retained as a conditional empirical multivariable descriptor, not a physical law.'
})
metadata_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
