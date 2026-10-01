#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'
FIG = RESULTS / 'figures'
TAB = RESULTS / 'tables'
for d in (DATA, RESULTS, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

INPUTS = {
    'rheology': DATA / 'rheology_table_S1.csv',
    'printability': DATA / 'printability_table_S2.csv',
    'angle': DATA / 'angle_fidelity_table_S3.csv',
}
for p in INPUTS.values():
    if not p.exists():
        raise FileNotFoundError(p)

rheology_df = pd.read_csv(INPUTS['rheology'])
printability_df = pd.read_csv(INPUTS['printability'])
angle_df = pd.read_csv(INPUTS['angle'])


def add_formulation_features(df):
    df = df.copy()
    df['Alg_pct_wv'] = df['Alg_mg_ml'] / 10
    df['HA_pct_wv'] = df['HA_mg_ml'] / 10
    df['Total_polymer_mg_ml'] = df['Alg_mg_ml'] + df['HA_mg_ml']
    df['Total_polymer_pct_wv'] = df['Total_polymer_mg_ml'] / 10
    df['HA_fraction'] = df['HA_mg_ml'] / df['Total_polymer_mg_ml']
    df['Alg_fraction'] = df['Alg_mg_ml'] / df['Total_polymer_mg_ml']
    df['Alg_HA_ratio'] = df['Alg_mg_ml'] / df['HA_mg_ml']
    df['Formulation_label'] = (
        df['Alg_pct_wv'].astype(int).astype(str) + 'ALG' +
        df['HA_pct_wv'].astype(int).astype(str) + 'HA'
    )
    return df


def add_ml_features(df, include_angle=False):
    df = df.copy()
    df['TP'] = df['Total_polymer_mg_ml']
    # Compatibility columns are retained in the distributed processed tables,
    # but are excluded from every canonical model feature set.
    df['TP_pct_wv'] = df['TP'] / 10
    df['HF'] = df['HA_fraction']
    df['Alg_fraction_feature'] = df['Alg_fraction']
    df['Alg_HA_ratio_safe'] = df['Alg_HA_ratio']
    if 'Pressure_kPa' in df.columns:
        df['P_over_TP'] = df['Pressure_kPa'] / df['TP']
        df['P_over_Alg'] = df['Pressure_kPa'] / df['Alg_mg_ml']
        df['P_over_HA'] = df['Pressure_kPa'] / df['HA_mg_ml']
        df['Alg_x_P'] = df['Alg_mg_ml'] * df['Pressure_kPa']
        df['HA_x_P'] = df['HA_mg_ml'] * df['Pressure_kPa']
        df['TP_x_P'] = df['TP'] * df['Pressure_kPa']
        df['HF_x_P'] = df['HF'] * df['Pressure_kPa']
    df['Alg_x_HA'] = df['Alg_mg_ml'] * df['HA_mg_ml']
    df['Alg_fraction_x_HA_fraction'] = df['Alg_fraction_feature'] * df['HF']
    if include_angle and 'Angle_deg' in df.columns:
        df['Angle_rad'] = np.deg2rad(df['Angle_deg'])
        df['sin_angle'] = np.sin(df['Angle_rad'])
        df['cos_angle'] = np.cos(df['Angle_rad'])
        if 'Pressure_kPa' in df.columns:
            df['P_x_Angle'] = df['Pressure_kPa'] * df['Angle_deg']
            df['Alg_x_Angle'] = df['Alg_mg_ml'] * df['Angle_deg']
            df['HA_x_Angle'] = df['HA_mg_ml'] * df['Angle_deg']
            df['TP_x_Angle'] = df['TP'] * df['Angle_deg']
    return df

rheology_std = add_formulation_features(rheology_df)
printability_std = add_formulation_features(printability_df)
angle_std = add_formulation_features(angle_df)

for name, df in {
    'rheology_standardized.csv': rheology_std,
    'printability_standardized.csv': printability_std,
    'angle_fidelity_standardized.csv': angle_std,
}.items():
    df.to_csv(DATA / name, index=False)

rheology_ml = add_ml_features(rheology_std, False)
printability_ml = add_ml_features(printability_std, False)
angle_ml = add_ml_features(angle_std, True)
for name, df in {
    'rheology_ml_features.csv': rheology_ml,
    'printability_ml_features.csv': printability_ml,
    'angle_fidelity_ml_features.csv': angle_ml,
}.items():
    df.to_csv(DATA / name, index=False)

rheo_response = ['Eta0_Pa_s','Cross_m','Cross_k_s','G_prime_LVE_Pa','G_double_prime_LVE_Pa','Yield_stress_Pa']
rheo_keys = ['Alg_mg_ml','HA_mg_ml','Alg_pct_wv','HA_pct_wv','Total_polymer_mg_ml','Total_polymer_pct_wv','HA_fraction','Alg_fraction','Alg_HA_ratio','Formulation_label']
rheology_summary = rheology_ml.groupby(rheo_keys, as_index=False)[rheo_response].agg(['mean','std'])
rheology_summary.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c for c in rheology_summary.columns]
rheology_summary = rheology_summary.rename(columns={
    'Eta0_Pa_s_mean':'Eta0_mean_Pa_s','Eta0_Pa_s_std':'Eta0_std_Pa_s',
    'Cross_m_mean':'Cross_m_mean','Cross_m_std':'Cross_m_std',
    'Cross_k_s_mean':'Cross_k_mean_s','Cross_k_s_std':'Cross_k_std_s',
    'G_prime_LVE_Pa_mean':'G_prime_mean_Pa','G_prime_LVE_Pa_std':'G_prime_std_Pa',
    'G_double_prime_LVE_Pa_mean':'G_double_prime_mean_Pa','G_double_prime_LVE_Pa_std':'G_double_prime_std_Pa',
    'Yield_stress_Pa_mean':'Yield_stress_mean_Pa','Yield_stress_Pa_std':'Yield_stress_std_Pa',
})
merge_cols = ['Alg_mg_ml','HA_mg_ml','Eta0_mean_Pa_s','Eta0_std_Pa_s','Cross_m_mean','Cross_m_std','Cross_k_mean_s','Cross_k_std_s','G_prime_mean_Pa','G_prime_std_Pa','G_double_prime_mean_Pa','G_double_prime_std_Pa','Yield_stress_mean_Pa','Yield_stress_std_Pa']
master_print = printability_ml.merge(rheology_summary[merge_cols], on=['Alg_mg_ml','HA_mg_ml'], how='left', validate='many_to_one')
master_angle = angle_ml.merge(rheology_summary[merge_cols], on=['Alg_mg_ml','HA_mg_ml'], how='left', validate='many_to_one')
rheology_summary.to_csv(DATA/'rheology_formulation_summary.csv', index=False)
master_print.to_csv(DATA/'master_printability_dataset.csv', index=False)
master_angle.to_csv(DATA/'master_angle_fidelity_dataset.csv', index=False)

TARGETS = {
    'Pr': (master_print, 'Pr_mean'),
    'SR': (master_print, 'SR_mean'),
    'Qm': (master_print, 'Qm_mean_mg_s'),
    'AF': (master_angle, 'AF_mean'),
}
modeling = {}
for t,(df,y) in TARGETS.items():
    out = df.dropna(subset=[y]).reset_index(drop=True)
    modeling[t] = out
    out.to_csv(DATA/f'modeling_dataset_{t}.csv', index=False)

expected_n = {'Pr':40,'SR':43,'Qm':40,'AF':96}
actual_n = {t:len(df) for t,df in modeling.items()}
if actual_n != expected_n:
    raise RuntimeError(f'Target-size mismatch: expected {expected_n}, got {actual_n}')

CORE=['Alg_mg_ml','HA_mg_ml','Pressure_kPa']
ENG=['TP','HF','Alg_HA_ratio_safe','P_over_TP','P_over_Alg','P_over_HA','Alg_x_HA','Alg_x_P','HA_x_P','TP_x_P','HF_x_P']
RHEO=['Eta0_mean_Pa_s','Cross_m_mean','Cross_k_mean_s','G_prime_mean_Pa','G_double_prime_mean_Pa','Yield_stress_mean_Pa']
ANGLE=['Angle_deg','P_x_Angle','Alg_x_Angle','HA_x_Angle','TP_x_Angle']
feature_sets={
    'core':CORE,
    'engineered':CORE+ENG,
    'rheology_enhanced':CORE+ENG+RHEO,
    'angle_engineered':CORE+ENG+ANGLE,
    'angle_rheology_enhanced':CORE+ENG+ANGLE+RHEO,
}
(DATA/'modeling_metadata.json').write_text(json.dumps({
    'feature_sets':feature_sets,
    'target_configs':{
        'Pr':{'target':'Pr_mean','allowed_feature_sets':['core','engineered','rheology_enhanced']},
        'SR':{'target':'SR_mean','allowed_feature_sets':['core','engineered','rheology_enhanced']},
        'Qm':{'target':'Qm_mean_mg_s','allowed_feature_sets':['core','engineered','rheology_enhanced']},
        'AF':{'target':'AF_mean','allowed_feature_sets':['core','angle_engineered','angle_rheology_enhanced']},
    },
    'compatibility_columns_not_used_for_modeling':['TP_pct_wv','Alg_fraction_feature','Alg_fraction_x_HA_fraction','Angle_rad','sin_angle','cos_angle'],
}, indent=2), encoding='utf-8')

# Compact, reproducible feature dictionary for the active feature sets.
units={
'Alg_mg_ml':'mg/mL','HA_mg_ml':'mg/mL','Pressure_kPa':'kPa','TP':'mg/mL','HF':'dimensionless',
'Alg_HA_ratio_safe':'dimensionless','P_over_TP':'kPa mL/mg','P_over_Alg':'kPa mL/mg','P_over_HA':'kPa mL/mg',
'Alg_x_HA':'(mg/mL)^2','Alg_x_P':'mg/mL x kPa','HA_x_P':'mg/mL x kPa','TP_x_P':'mg/mL x kPa','HF_x_P':'kPa',
'Eta0_mean_Pa_s':'Pa s','Cross_m_mean':'dimensionless','Cross_k_mean_s':'s','G_prime_mean_Pa':'Pa','G_double_prime_mean_Pa':'Pa','Yield_stress_mean_Pa':'Pa',
'Angle_deg':'degree','P_x_Angle':'kPa x degree','Alg_x_Angle':'mg/mL x degree','HA_x_Angle':'mg/mL x degree','TP_x_Angle':'mg/mL x degree'}
rows=[]
for fs, feats in feature_sets.items():
    for f in feats:
        rows.append({'Feature_set':fs,'Feature_name':f,'Unit':units.get(f,'')})
pd.DataFrame(rows).drop_duplicates().to_csv(TAB/'feature_dictionary.csv',index=False)
pd.DataFrame(rows).drop_duplicates().to_excel(TAB/'feature_dictionary.xlsx',index=False)

eda=[]
for t,(df,y) in TARGETS.items():
    m=modeling[t]
    eda.append({'Target':t,'Target_column':y,'N':len(m),'Mean':m[y].mean(),'Std':m[y].std(),'Min':m[y].min(),'Median':m[y].median(),'Max':m[y].max()})
    plt.figure(figsize=(6,4)); plt.hist(m[y], bins=10); plt.xlabel(y); plt.ylabel('Count'); plt.title(f'Distribution of {t}'); plt.tight_layout(); plt.savefig(FIG/f'distribution_{t}.png',dpi=300); plt.close()
pd.DataFrame(eda).to_csv(TAB/'eda_target_summary.csv',index=False)
pd.DataFrame(eda).to_excel(TAB/'eda_target_summary.xlsx',index=False)

corr=[]
for t,(df,y) in TARGETS.items():
    m=modeling[t]
    fs='angle_rheology_enhanced' if t=='AF' else 'rheology_enhanced'
    for f in [c for c in feature_sets[fs] if c in m.columns]:
        tmp=m[[f,y]].dropna()
        pear=np.nan if tmp[f].nunique()<=1 else tmp[f].corr(tmp[y],method='pearson')
        spear=np.nan if tmp[f].nunique()<=1 else tmp[f].corr(tmp[y],method='spearman')
        corr.append({'Target':t,'Target_column':y,'Feature_set':fs,'Feature':f,'Pearson_r':pear,'Spearman_r':spear,'Abs_Pearson_r':abs(pear) if pd.notna(pear) else np.nan,'Abs_Spearman_r':abs(spear) if pd.notna(spear) else np.nan,'N':len(tmp)})
corr_df=pd.DataFrame(corr)
corr_df.to_csv(TAB/'feature_target_correlations.csv',index=False)
corr_df.to_excel(TAB/'feature_target_correlations.xlsx',index=False)
for t in ['AF','Pr','Qm','SR']:
    s=corr_df[corr_df.Target==t].sort_values('Abs_Spearman_r',ascending=False).head(12).sort_values('Abs_Spearman_r')
    plt.figure(figsize=(7,5)); plt.barh(s.Feature,s.Abs_Spearman_r); plt.xlabel('Absolute Spearman correlation'); plt.title(f'Top feature-target correlations for {t}'); plt.tight_layout(); plt.savefig(FIG/f'top_correlations_{t}.png',dpi=300); plt.close()

summary={'status':'PASS','inputs':{k:str(v.relative_to(ROOT)) for k,v in INPUTS.items()},'target_sizes':actual_n,'seed_policy':'No stochastic fitting in this step; downstream core stochastic procedures use seed 42.'}
(RESULTS/'canonical_preparation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
