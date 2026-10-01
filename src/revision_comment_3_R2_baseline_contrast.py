from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
BASE_OUT = ROOT / 'results' / 'revision_comment_3_R2'
NEST_OUT = ROOT / 'results' / 'revision_comments_4_R1_16_R2'
OUT = ROOT / 'results' / 'revision_comment_3_R2_baseline_contrast'
SUP = ROOT / 'supplementary' / 'final_revision' / 'tables'
OUT.mkdir(parents=True, exist_ok=True)
SUP.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(np.unique(y_true)) <= 1 or len(np.unique(y_pred)) <= 1:
        return np.nan
    return float(spearmanr(y_true, y_pred).correlation)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(mean_squared_error(y_true, y_pred) ** 0.5),
        'Spearman': safe_spearman(y_true, y_pred),
    }


class AngleGroupMeanRegressor(BaseEstimator, RegressorMixin):
    """Predict the training-set mean AF within each sampled angle level."""
    def fit(self, X, y):
        x = np.asarray(X).reshape(-1)
        y = np.asarray(y, dtype=float)
        self.global_mean_ = float(np.mean(y))
        self.angle_means_ = {float(a): float(np.mean(y[x == a])) for a in np.unique(x)}
        return self

    def predict(self, X):
        x = np.asarray(X).reshape(-1)
        return np.array([self.angle_means_.get(float(a), self.global_mean_) for a in x], dtype=float)


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


def repeated_predictions(df, feature, target, estimator, label, nested_target_pred):
    """Fit the simple baseline on the exact outer folds used by nested CV.

    Table S30 is a matched comparison.  We therefore reuse the authoritative
    (Repeat, Outer_fold, Row_index) membership written by the nested procedure
    instead of creating a second cross-validation splitter.
    """
    X = df[[feature]].copy()
    y = df[target].astype(float).copy()
    all_idx = np.arange(len(df), dtype=int)
    rows = []

    nested_target_pred = nested_target_pred.copy()
    nested_target_pred['Repeat'] = nested_target_pred['Repeat'].astype(int)
    nested_target_pred['Outer_fold'] = nested_target_pred['Outer_fold'].astype(int)
    nested_target_pred['Row_index'] = nested_target_pred['Row_index'].astype(int)

    repeats = sorted(nested_target_pred['Repeat'].unique().tolist())
    if repeats != list(range(1, 21)):
        raise RuntimeError(f'Expected nested repeats 1..20 for {label}; found {repeats}')

    for repeat in repeats:
        rep = nested_target_pred[nested_target_pred['Repeat'] == repeat]
        folds = sorted(rep['Outer_fold'].unique().tolist())
        if folds != [1, 2, 3, 4, 5]:
            raise RuntimeError(f'Expected folds 1..5 for {label}, repeat {repeat}; found {folds}')
        seen = []
        for outer_fold in folds:
            g = rep[rep['Outer_fold'] == outer_fold].sort_values('Row_index')
            te = g['Row_index'].to_numpy(dtype=int)
            if len(np.unique(te)) != len(te):
                raise RuntimeError(f'Duplicate test row indices for {label}, repeat {repeat}, fold {outer_fold}')
            tr = np.setdiff1d(all_idx, te, assume_unique=True)

            observed_nested = g['Observed'].to_numpy(dtype=float)
            observed_here = y.iloc[te].to_numpy(dtype=float)
            if not np.allclose(observed_nested, observed_here, rtol=0.0, atol=1e-12):
                raise RuntimeError(f'Observed-value mismatch for {label}, repeat {repeat}, fold {outer_fold}')

            m = clone(estimator)
            m.fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[te])
            for idx, obs, pr in zip(te, observed_here, pred):
                rows.append({
                    'Baseline': label,
                    'Repeat': int(repeat),
                    'Outer_fold': int(outer_fold),
                    'Row_index': int(idx),
                    'Observed': float(obs),
                    'Predicted': float(pr),
                })
            seen.extend(te.tolist())

        if sorted(seen) != all_idx.tolist():
            raise RuntimeError(f'Nested fold membership does not cover each row exactly once for {label}, repeat {repeat}')

    return pd.DataFrame(rows)


def repeat_metrics(pred_df, prediction_col='Predicted'):
    rows = []
    for repeat, g in pred_df.groupby('Repeat'):
        met = metrics(g['Observed'], g[prediction_col])
        met['Repeat'] = int(repeat)
        rows.append(met)
    return pd.DataFrame(rows)


af = pd.read_csv(DATA / 'modeling_dataset_AF.csv').dropna(subset=['AF_mean', 'Angle_deg']).reset_index(drop=True)
qm = pd.read_csv(DATA / 'modeling_dataset_Qm.csv').dropna(subset=['Qm_mean_mg_s', 'P_over_TP']).reset_index(drop=True)

jobs = [
    ('AF', af, 'Angle_deg', 'AF_mean', LinearRegression(), 'AF: angle-only linear'),
    ('AF', af, 'Angle_deg', 'AF_mean', AngleGroupMeanRegressor(), 'AF: angle-only categorical mean'),
    ('Qm', qm, 'P_over_TP', 'Qm_mean_mg_s', LinearRegression(), 'Qm: P/TP linear'),
    ('Qm', qm, 'P_over_TP', 'Qm_mean_mg_s', PowerLawRegressor(), 'Qm: P/TP power law'),
]

nested_pred = pd.read_csv(NEST_OUT / 'Repeated_Predictions_Comments_4-R1_16-R2.csv')
nested_lofo = pd.read_csv(NEST_OUT / 'Nested_LOFO_Summary_Comments_4-R1_16-R2.csv')
base_summary = pd.read_csv(BASE_OUT / 'baseline_summary_Comment_3-R2.csv')

prediction_parts = []
repeat_detail_parts = []
summary_rows = []
for target_name, df, feature, target, estimator, label in jobs:
    npred = nested_pred[nested_pred['Target'] == target_name].copy()
    bp = repeated_predictions(df, feature, target, estimator, label, npred)
    bp['Target'] = target_name
    prediction_parts.append(bp)

    bm = repeat_metrics(bp)
    bm['Target'] = target_name
    bm['Baseline'] = label

    nm = repeat_metrics(npred)
    nm = nm.rename(columns={c: f'Nested_{c}' for c in ['R2','MAE','RMSE','Spearman']})
    paired = bm.merge(nm, on='Repeat', how='inner')
    paired['Delta_R2_nested_minus_baseline'] = paired['Nested_R2'] - paired['R2']
    paired['Delta_MAE_baseline_minus_nested'] = paired['MAE'] - paired['Nested_MAE']
    paired['Delta_RMSE_baseline_minus_nested'] = paired['RMSE'] - paired['Nested_RMSE']
    repeat_detail_parts.append(paired)

    summary_rows.append({
        'Target': target_name,
        'Baseline': label,
        'Validation': 'Repeated 5-fold CV; repeat-pooled OOF',
        'N_matched_repeats': int(len(paired)),
        'Baseline_R2_mean': float(paired['R2'].mean()),
        'Nested_multivariable_R2_mean': float(paired['Nested_R2'].mean()),
        'Delta_R2_nested_minus_baseline': float(paired['Delta_R2_nested_minus_baseline'].mean()),
        'Baseline_MAE_mean': float(paired['MAE'].mean()),
        'Nested_multivariable_MAE_mean': float(paired['Nested_MAE'].mean()),
        'Delta_MAE_baseline_minus_nested': float(paired['Delta_MAE_baseline_minus_nested'].mean()),
        'Baseline_RMSE_mean': float(paired['RMSE'].mean()),
        'Nested_multivariable_RMSE_mean': float(paired['Nested_RMSE'].mean()),
        'Delta_RMSE_baseline_minus_nested': float(paired['Delta_RMSE_baseline_minus_nested'].mean()),
        'Fraction_repeats_nested_better_R2': float((paired['Delta_R2_nested_minus_baseline'] > 0).mean()),
        'Fraction_repeats_nested_better_MAE': float((paired['Delta_MAE_baseline_minus_nested'] > 0).mean()),
        'Fraction_repeats_nested_better_RMSE': float((paired['Delta_RMSE_baseline_minus_nested'] > 0).mean()),
        'Interpretation': 'Descriptive paired comparison on identical held-out repeat partitions; repeated partitions are not treated as independent inferential replicates.'
    })

    b_lofo = base_summary[(base_summary['Target'] == target_name) & (base_summary['Baseline'] == label) & base_summary['Validation'].str.startswith('LOFO')].iloc[0]
    n_lofo = nested_lofo[nested_lofo['Target'] == target_name].iloc[0]
    summary_rows.append({
        'Target': target_name,
        'Baseline': label,
        'Validation': 'LOFO; pooled outer predictions',
        'N_matched_repeats': np.nan,
        'Baseline_R2_mean': float(b_lofo['R2']),
        'Nested_multivariable_R2_mean': float(n_lofo['LOFO_R2']),
        'Delta_R2_nested_minus_baseline': float(n_lofo['LOFO_R2'] - b_lofo['R2']),
        'Baseline_MAE_mean': float(b_lofo['MAE']),
        'Nested_multivariable_MAE_mean': float(n_lofo['LOFO_MAE']),
        'Delta_MAE_baseline_minus_nested': float(b_lofo['MAE'] - n_lofo['LOFO_MAE']),
        'Baseline_RMSE_mean': float(b_lofo['RMSE']),
        'Nested_multivariable_RMSE_mean': float(n_lofo['LOFO_RMSE']),
        'Delta_RMSE_baseline_minus_nested': float(b_lofo['RMSE'] - n_lofo['LOFO_RMSE']),
        'Fraction_repeats_nested_better_R2': np.nan,
        'Fraction_repeats_nested_better_MAE': np.nan,
        'Fraction_repeats_nested_better_RMSE': np.nan,
        'Interpretation': 'Pooled formulation-level comparison; positive delta R2 or error delta favors nested multivariable modeling.'
    })

pred_all = pd.concat(prediction_parts, ignore_index=True)
repeat_detail = pd.concat(repeat_detail_parts, ignore_index=True)
summary = pd.DataFrame(summary_rows)

pred_all.to_csv(OUT / 'Baseline_Repeated_Predictions_Comment_3-R2.csv', index=False)
repeat_detail.to_csv(OUT / 'Paired_Repeat_Detail_Comment_3-R2.csv', index=False)
summary.to_csv(OUT / 'Baseline_vs_Nested_Apples_to_Apples_Comment_3-R2.csv', index=False)
summary.to_csv(SUP / 'Table_S30_Baseline_vs_Nested_Apples_to_Apples_Comment_3-R2.csv', index=False)

print(summary.to_string(index=False))
