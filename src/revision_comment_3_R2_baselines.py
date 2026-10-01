from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'results' / 'revision_comment_3_R2'
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


class AngleGroupMeanRegressor(BaseEstimator, RegressorMixin):
    """One-way-ANOVA-like baseline: predict the training mean AF for each angle level."""
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


def repeated_cv(df, feature, target, estimator, label, n_splits=5, n_repeats=20):
    X = df[[feature]].copy()
    y = df[target].astype(float).copy()
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    split_rows = []
    for split_id, (tr, te) in enumerate(cv.split(X), start=1):
        m = clone(estimator)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        met = calc_metrics(y.iloc[te], pred)
        met.update({'Baseline': label, 'Split': split_id, 'N_test': len(te)})
        split_rows.append(met)
    split_df = pd.DataFrame(split_rows)
    summary = {
        'Baseline': label,
        'Validation': 'Repeated 5-fold CV (20 repeats)',
        'N': len(df),
        'Predictor': feature,
        'R2': split_df['R2'].mean(),
        'R2_SD': split_df['R2'].std(ddof=1),
        'MAE': split_df['MAE'].mean(),
        'MAE_SD': split_df['MAE'].std(ddof=1),
        'RMSE': split_df['RMSE'].mean(),
        'RMSE_SD': split_df['RMSE'].std(ddof=1),
        'Spearman': split_df['Spearman'].mean(),
        'Spearman_SD': split_df['Spearman'].std(ddof=1),
    }
    return summary, split_df


def lofo(df, feature, target, estimator, label):
    X = df[[feature]].copy()
    y = df[target].astype(float).copy()
    groups = df['Formulation_label'].astype(str)
    pred_rows = []
    group_rows = []
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
                'Error': float(pr - obs),
                'Abs_error': float(abs(pr - obs)),
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
        'MAE_SD': group_df['MAE'].std(ddof=1),
        'RMSE': met['RMSE'],
        'RMSE_SD': group_df['RMSE'].std(ddof=1),
        'Spearman': met['Spearman'],
        'Spearman_SD': group_df['Spearman'].std(ddof=1),
    }
    return summary, pred_df, group_df


af = pd.read_csv(DATA / 'modeling_dataset_AF.csv').dropna(subset=['AF_mean', 'Angle_deg']).reset_index(drop=True)
qm = pd.read_csv(DATA / 'modeling_dataset_Qm.csv').dropna(subset=['Qm_mean_mg_s', 'P_over_TP']).reset_index(drop=True)

jobs = [
    ('AF', af, 'Angle_deg', 'AF_mean', LinearRegression(), 'AF: angle-only linear'),
    ('AF', af, 'Angle_deg', 'AF_mean', AngleGroupMeanRegressor(), 'AF: angle-only categorical mean'),
    ('Qm', qm, 'P_over_TP', 'Qm_mean_mg_s', LinearRegression(), 'Qm: P/TP linear'),
    ('Qm', qm, 'P_over_TP', 'Qm_mean_mg_s', PowerLawRegressor(), 'Qm: P/TP power law'),
]

summary_rows = []
repeated_parts = []
lofo_pred_parts = []
lofo_group_parts = []
for target_name, df, feature, target, estimator, label in jobs:
    rep_sum, rep_splits = repeated_cv(df, feature, target, estimator, label)
    rep_sum['Target'] = target_name
    summary_rows.append(rep_sum)
    rep_splits['Target'] = target_name
    repeated_parts.append(rep_splits)

    lofo_sum, lofo_pred, lofo_group = lofo(df, feature, target, estimator, label)
    lofo_sum['Target'] = target_name
    summary_rows.append(lofo_sum)
    lofo_pred['Target'] = target_name
    lofo_group['Target'] = target_name
    lofo_pred_parts.append(lofo_pred)
    lofo_group_parts.append(lofo_group)

summary = pd.DataFrame(summary_rows)[['Target','Baseline','Validation','Predictor','N','R2','R2_SD','MAE','MAE_SD','RMSE','RMSE_SD','Spearman','Spearman_SD']]
repeated_splits = pd.concat(repeated_parts, ignore_index=True)
lofo_predictions = pd.concat(lofo_pred_parts, ignore_index=True)
lofo_groups = pd.concat(lofo_group_parts, ignore_index=True)

# Full-data parameter fits are descriptive only, not validation metrics.
full_fit_rows=[]
for target_name, df, feature, target, estimator, label in jobs:
    m=clone(estimator).fit(df[[feature]],df[target])
    row={'Target':target_name,'Baseline':label,'Predictor':feature}
    if isinstance(m, LinearRegression):
        row.update({'Intercept_or_a':float(m.intercept_), 'Slope_or_b':float(m.coef_[0]), 'Equation_form':'y = intercept + slope*x'})
    elif isinstance(m, PowerLawRegressor):
        row.update({'Intercept_or_a':m.a_, 'Slope_or_b':m.b_, 'Equation_form':'y = a*x^b'})
    else:
        means='; '.join(f'{k:g} deg: {v:.6f}' for k,v in sorted(m.angle_means_.items()))
        row.update({'Intercept_or_a':np.nan,'Slope_or_b':np.nan,'Equation_form':'training mean by angle level','Angle_means_full_data':means})
    full_fit_rows.append(row)
full_fit=pd.DataFrame(full_fit_rows)

# The simple baseline analysis is intentionally self-contained.
# Comparison against the final nested multivariable workflow is performed later
# by revision_comment_3_R2_baseline_contrast.py after nested CV has been regenerated.

summary.to_csv(OUT/'baseline_summary_Comment_3-R2.csv', index=False)
repeated_splits.to_csv(OUT/'baseline_repeated_split_metrics_Comment_3-R2.csv', index=False)
lofo_predictions.to_csv(OUT/'baseline_LOFO_predictions_Comment_3-R2.csv', index=False)
lofo_groups.to_csv(OUT/'baseline_LOFO_group_metrics_Comment_3-R2.csv', index=False)
full_fit.to_csv(OUT/'baseline_full_fit_parameters_Comment_3-R2.csv', index=False)

print('\nBASELINE SUMMARY')
print(summary.to_string(index=False))
print('\nFULL-DATA DESCRIPTIVE FITS')
print(full_fit.to_string(index=False))
