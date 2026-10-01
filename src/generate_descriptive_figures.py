#!/usr/bin/env python3
"""Generate descriptive SI figures and the manuscript prediction heatmap from canonical data.

This module is intentionally downstream of prepare_canonical_inputs.py. It uses only
current canonical processed datasets and prespecified reference-model definitions. No
archived prediction table is read.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'results' / 'descriptive_figures'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
np.random.seed(SEED)

CORE = ['Alg_mg_ml','HA_mg_ml','Pressure_kPa']
ENG = ['TP','HF','Alg_HA_ratio_safe','P_over_TP','P_over_Alg','P_over_HA','Alg_x_HA','Alg_x_P','HA_x_P','TP_x_P','HF_x_P']
RHEO = ['Eta0_mean_Pa_s','Cross_m_mean','Cross_k_mean_s','G_prime_mean_Pa','G_double_prime_mean_Pa','Yield_stress_mean_Pa']
ANGLE = ['Angle_deg','P_x_Angle','Alg_x_Angle','HA_x_Angle','TP_x_Angle']
FEATURE_SETS = {
    'core': CORE,
    'engineered': CORE + ENG,
    'rheology_enhanced': CORE + ENG + RHEO,
    'angle_engineered': CORE + ENG + ANGLE,
}
TARGETS = {
    'Pr': ('modeling_dataset_Pr.csv','Pr_mean'),
    'SR': ('modeling_dataset_SR.csv','SR_mean'),
    'Qm': ('modeling_dataset_Qm.csv','Qm_mean_mg_s'),
    'AF': ('modeling_dataset_AF.csv','AF_mean'),
}
REFERENCE = {
    'Pr': ('core','LinearRegression'),
    'SR': ('rheology_enhanced','SVR_RBF'),
    'Qm': ('engineered','SVR_RBF'),
    'AF': ('angle_engineered','XGBoost'),
}

REQUIRED = [
    DATA/'master_printability_dataset.csv', DATA/'master_angle_fidelity_dataset.csv',
    DATA/'rheology_formulation_summary.csv', DATA/'modeling_dataset_Pr.csv',
    DATA/'modeling_dataset_SR.csv', DATA/'modeling_dataset_Qm.csv', DATA/'modeling_dataset_AF.csv',
]
for p in REQUIRED:
    if not p.exists():
        raise FileNotFoundError(f'Missing canonical prerequisite: {p.relative_to(ROOT)}')


def model(name):
    if name == 'LinearRegression':
        return Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    if name == 'SVR_RBF':
        return Pipeline([('scaler', StandardScaler()), ('model', SVR(kernel='rbf', C=10.0, epsilon=0.05))])
    if name == 'XGBoost':
        return XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=2,
            subsample=0.9, colsample_bytree=0.9,
            objective='reg:squarederror', random_state=SEED, n_jobs=1,
        )
    raise KeyError(name)


def add_engineered(df):
    df = df.copy()
    df['TP'] = df['Alg_mg_ml'] + df['HA_mg_ml']
    df['HF'] = df['HA_mg_ml'] / df['TP']
    df['Alg_HA_ratio_safe'] = df['Alg_mg_ml'] / df['HA_mg_ml']
    df['P_over_TP'] = df['Pressure_kPa'] / df['TP']
    df['P_over_Alg'] = df['Pressure_kPa'] / df['Alg_mg_ml']
    df['P_over_HA'] = df['Pressure_kPa'] / df['HA_mg_ml']
    df['Alg_x_HA'] = df['Alg_mg_ml'] * df['HA_mg_ml']
    df['Alg_x_P'] = df['Alg_mg_ml'] * df['Pressure_kPa']
    df['HA_x_P'] = df['HA_mg_ml'] * df['Pressure_kPa']
    df['TP_x_P'] = df['TP'] * df['Pressure_kPa']
    df['HF_x_P'] = df['HF'] * df['Pressure_kPa']
    if 'Angle_deg' in df.columns:
        df['P_x_Angle'] = df['Pressure_kPa'] * df['Angle_deg']
        df['Alg_x_Angle'] = df['Alg_mg_ml'] * df['Angle_deg']
        df['HA_x_Angle'] = df['HA_mg_ml'] * df['Angle_deg']
        df['TP_x_Angle'] = df['TP'] * df['Angle_deg']
    return df


def fit_reference(target):
    fn, ycol = TARGETS[target]
    train = pd.read_csv(DATA/fn).dropna(subset=[ycol]).reset_index(drop=True)
    feature_set, model_name = REFERENCE[target]
    feats = FEATURE_SETS[feature_set]
    if train[feats].isna().any().any():
        raise ValueError(f'Missing features in {target} training data')
    m = model(model_name)
    m.fit(train[feats], train[ycol])
    return m, feats, len(train), feature_set, model_name


master_print = pd.read_csv(DATA/'master_printability_dataset.csv')
master_angle = pd.read_csv(DATA/'master_angle_fidelity_dataset.csv')
rheo = pd.read_csv(DATA/'rheology_formulation_summary.csv')
rheo_cols = ['Alg_mg_ml','HA_mg_ml'] + RHEO

# Figure S1: target-specific distributions from the exact modeling datasets.
fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.0)); axes = axes.ravel()
for ax, target in zip(axes, ['AF','Pr','Qm','SR']):
    fn, ycol = TARGETS[target]
    df = pd.read_csv(DATA/fn)
    ax.hist(df[ycol].dropna(), bins=10)
    label = {'AF':'Angle Fidelity Factor (AF)', 'Pr':'Printability Ratio (Pr)', 'Qm':'Mass Flow Rate (Qm, mg/s)', 'SR':'Spreading Ratio (SR)'}[target]
    ax.set_xlabel(label); ax.set_ylabel('Count'); ax.set_title(f'Distribution of {target}')
fig.tight_layout()
fig.savefig(OUT/'FigureS1_Target_Output_Distributions.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure S2: exploratory source-observation scatter plots. These are descriptive only.
# Use the same 13 panel definitions stated in the SI caption.
panels = [
    ('AF','Angle_deg','Angle vs AF'),
    ('AF','HF','HA fraction vs AF'),
    ('Pr','HF','HA fraction vs Pr'),
    ('Qm','HF','HA fraction vs Qm'),
    ('SR','HF','HA fraction vs SR'),
    ('AF','Pressure_kPa','Pressure vs AF'),
    ('Pr','Pressure_kPa','Pressure vs Pr'),
    ('Qm','Pressure_kPa','Pressure vs Qm'),
    ('SR','Pressure_kPa','Pressure vs SR'),
    ('AF','TP','Total polymer vs AF'),
    ('Pr','TP','Total polymer vs Pr'),
    ('Qm','TP','Total polymer vs Qm'),
    ('SR','TP','Total polymer vs SR'),
]
fig, axes = plt.subplots(5, 3, figsize=(12.0, 13.6)); flat = axes.ravel()
letters = [chr(ord('A')+i) for i in range(len(panels))]
for i, (target, xcol, title) in enumerate(panels):
    fn, ycol = TARGETS[target]
    df = pd.read_csv(DATA/fn)
    ax = flat[i]
    ax.scatter(df[xcol], df[ycol], s=18)
    xlabels = {'Angle_deg':'Deposition angle (deg)', 'HF':'HA fraction', 'Pressure_kPa':'Pressure (kPa)', 'TP':'Total polymer (mg/mL)'}
    ylabels = {'AF':'AF', 'Pr':'Pr', 'Qm':'Qm (mg/s)', 'SR':'SR'}
    ax.set_xlabel(xlabels.get(xcol, xcol)); ax.set_ylabel(ylabels[target]); ax.set_title(title)
    ax.text(-0.15, 1.05, letters[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
for ax in flat[len(panels):]: ax.axis('off')
fig.tight_layout()
fig.savefig(OUT/'FigureS2_Exploratory_Feature_Output_Scatter.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Manuscript Figure 3: descriptive reference-model predictions on a source-bounded display grid.
# The grid is generated fresh from the observed min/max pressure for each formulation in the
# corresponding source subset, at 5-kPa display increments. It is not used for candidate eligibility/ranking.
forms_print = (master_print[['Formulation_label','Alg_mg_ml','HA_mg_ml']]
               .drop_duplicates().sort_values(['Alg_mg_ml','HA_mg_ml']).reset_index(drop=True))
print_rows = []
for _, f in forms_print.iterrows():
    sub = master_print[master_print['Formulation_label'].eq(f['Formulation_label'])]
    pmin, pmax = int(sub['Pressure_kPa'].min()), int(sub['Pressure_kPa'].max())
    for pressure in range(pmin, pmax + 1, 5):
        print_rows.append({'Formulation_label':f['Formulation_label'], 'Alg_mg_ml':f['Alg_mg_ml'], 'HA_mg_ml':f['HA_mg_ml'], 'Pressure_kPa':pressure})
print_grid = add_engineered(pd.DataFrame(print_rows)).merge(rheo[rheo_cols], on=['Alg_mg_ml','HA_mg_ml'], how='left', validate='many_to_one')

fit_meta = {}
for target in ['Pr','SR','Qm']:
    m, feats, n, fs, mn = fit_reference(target)
    if print_grid[feats].isna().any().any(): raise ValueError(f'Missing grid features for {target}')
    print_grid[f'Predicted_{target}'] = m.predict(print_grid[feats])
    fit_meta[target] = {'N_train':n,'feature_set':fs,'model':mn,'features':feats}

forms_angle = (master_angle[['Formulation_label','Alg_mg_ml','HA_mg_ml']]
               .drop_duplicates().sort_values(['Alg_mg_ml','HA_mg_ml']).reset_index(drop=True))
angles = sorted(float(v) for v in master_angle['Angle_deg'].dropna().unique())
af_rows = []
for _, f in forms_angle.iterrows():
    sub = master_angle[master_angle['Formulation_label'].eq(f['Formulation_label'])]
    pmin, pmax = int(sub['Pressure_kPa'].min()), int(sub['Pressure_kPa'].max())
    for pressure in range(pmin, pmax + 1, 5):
        for angle in angles:
            af_rows.append({'Formulation_label':f['Formulation_label'], 'Alg_mg_ml':f['Alg_mg_ml'], 'HA_mg_ml':f['HA_mg_ml'], 'Pressure_kPa':pressure, 'Angle_deg':angle})
af_grid = add_engineered(pd.DataFrame(af_rows)).merge(rheo[rheo_cols], on=['Alg_mg_ml','HA_mg_ml'], how='left', validate='many_to_one')
m, feats, n, fs, mn = fit_reference('AF')
if af_grid[feats].isna().any().any(): raise ValueError('Missing AF grid features')
af_grid['Predicted_AF'] = m.predict(af_grid[feats])
af_summary = (af_grid.groupby(['Formulation_label','Alg_mg_ml','HA_mg_ml','Pressure_kPa'], as_index=False)
              .agg(Mean_predicted_AF=('Predicted_AF','mean'), Min_predicted_AF=('Predicted_AF','min'), Max_predicted_AF=('Predicted_AF','max')))
fit_meta['AF'] = {'N_train':n,'feature_set':fs,'model':mn,'features':feats,'sampled_angles':angles}

print_grid.to_csv(OUT/'Manuscript_Figure3_Printability_Grid_Predictions.csv', index=False)
af_grid.to_csv(OUT/'Manuscript_Figure3_AF_Angle_Predictions.csv', index=False)
af_summary.to_csv(OUT/'Manuscript_Figure3_AF_Mean_Grid_Predictions.csv', index=False)

form_order = forms_print['Formulation_label'].tolist()
def heatmap(ax, df, value_col, title, cbar_label, pressures):
    pivot = df.pivot(index='Formulation_label', columns='Pressure_kPa', values=value_col).reindex(index=form_order, columns=pressures)
    arr = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    im = ax.imshow(arr, aspect='auto', interpolation='nearest')
    ax.set_yticks(np.arange(len(form_order))); ax.set_yticklabels(form_order)
    ax.set_xticks(np.arange(len(pressures))); ax.set_xticklabels([str(x) for x in pressures], rotation=90)
    ax.set_xlabel('Pressure (kPa)'); ax.set_ylabel('Formulation'); ax.set_title(title)
    cb = fig.colorbar(im, ax=ax); cb.set_label(cbar_label)

all_press_print = list(range(int(print_grid.Pressure_kPa.min()), int(print_grid.Pressure_kPa.max())+1, 5))
all_press_af = list(range(int(af_summary.Pressure_kPa.min()), int(af_summary.Pressure_kPa.max())+1, 5))
fig, axes = plt.subplots(2, 2, figsize=(16, 9.5))
heatmap(axes[0,0], print_grid, 'Predicted_Pr', 'Predicted printability ratio (Pr)', 'Predicted Pr', all_press_print)
heatmap(axes[0,1], print_grid, 'Predicted_SR', 'Predicted spreading ratio (SR)', 'Predicted SR', all_press_print)
heatmap(axes[1,0], print_grid, 'Predicted_Qm', 'Predicted mass flow rate (Qm)', 'Predicted Qm (mg/s)', all_press_print)
heatmap(axes[1,1], af_summary, 'Mean_predicted_AF', 'Mean predicted angle fidelity (AF)', 'Mean predicted AF', all_press_af)
for lab, ax in zip(['A','B','C','D'], axes.ravel()):
    ax.text(-0.17, 1.08, lab, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top')
fig.tight_layout()
fig.savefig(OUT/'Manuscript_Figure3_Prediction_Heatmaps.png', dpi=300, bbox_inches='tight')
plt.close(fig)

meta = {
    'seed':SEED,
    'figure_s1':'Generated from the four canonical target-specific modeling datasets.',
    'figure_s2':'Generated from canonical target-specific modeling rows; descriptive only.',
    'manuscript_figure3':{
        'scope':'Descriptive full-data reference-model predictions; not a performance estimate and not used for candidate eligibility or ranking.',
        'grid':'Formulation-specific source pressure min-to-max at 5-kPa display increments; generated from source-derived master tables, not archived predictions.',
        'reference_models':fit_meta,
    }
}
(OUT/'Descriptive_Figure_Metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('Generated descriptive figures and current-model manuscript Figure 3 in', OUT)
