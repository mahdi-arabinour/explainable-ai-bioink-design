from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'results' / 'revision_comments_4_5'
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

TARGETS = {
    'Pr': ('modeling_dataset_Pr.csv', 'Pr_mean', ['core','engineered','rheology_enhanced']),
    'SR': ('modeling_dataset_SR.csv', 'SR_mean', ['core','engineered','rheology_enhanced']),
    'Qm': ('modeling_dataset_Qm.csv', 'Qm_mean_mg_s', ['core','engineered','rheology_enhanced']),
    'AF': ('modeling_dataset_AF.csv', 'AF_mean', ['core','angle_engineered','angle_rheology_enhanced']),
}

CORE = ['Alg_mg_ml','HA_mg_ml','Pressure_kPa']
RHEO = ['Eta0_mean_Pa_s','Cross_m_mean','Cross_k_mean_s','G_prime_mean_Pa','G_double_prime_mean_Pa','Yield_stress_mean_Pa']

OLD_ENG = [
    'TP','TP_pct_wv','HF','Alg_fraction_feature','Alg_HA_ratio_safe',
    'P_over_TP','P_over_Alg','P_over_HA','Alg_x_HA','Alg_x_P','HA_x_P','TP_x_P','HF_x_P',
    'Alg_fraction_x_HA_fraction'
]
OLD_ANGLE = ['Angle_deg','Angle_rad','sin_angle','cos_angle','P_x_Angle','Alg_x_Angle','HA_x_Angle','TP_x_Angle']

# Reviewer-responsive feature set: remove exact/affine aliases and redundant fraction product.
CLEAN_ENG = [
    'TP','HF','Alg_HA_ratio_safe',
    'P_over_TP','P_over_Alg','P_over_HA','Alg_x_HA','Alg_x_P','HA_x_P','TP_x_P','HF_x_P'
]
CLEAN_ANGLE = ['Angle_deg','P_x_Angle','Alg_x_Angle','HA_x_Angle','TP_x_Angle']

REMOVED = {
    'TP_pct_wv': 'exact scale duplicate of TP (TP = 10 * TP_pct_wv)',
    'Alg_fraction_feature': 'exact complement of HF (Alg_fraction = 1 - HF)',
    'Alg_fraction_x_HA_fraction': 'deterministic function HF * (1 - HF)',
    'Angle_rad': 'exact scale duplicate of Angle_deg',
    'sin_angle': 'deterministic transform of the same three sampled angle levels',
    'cos_angle': 'affine duplicate of Angle_deg for sampled 60/90/120 degree levels',
}


def get_models():
    return {
        'LinearRegression': Pipeline([('scaler',StandardScaler()),('model',LinearRegression())]),
        'SVR_RBF': Pipeline([('scaler',StandardScaler()),('model',SVR(kernel='rbf',C=10.0,epsilon=0.05))]),
        'RandomForest': RandomForestRegressor(n_estimators=300,max_depth=None,min_samples_leaf=2,random_state=RANDOM_STATE,n_jobs=1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200,learning_rate=0.05,max_depth=2,random_state=RANDOM_STATE),
        'XGBoost': XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=2,subsample=0.9,colsample_bytree=0.9,objective='reg:squarederror',random_state=RANDOM_STATE,n_jobs=1),
    }


def safe_spearman(a,b):
    if len(np.unique(a)) <= 1 or len(np.unique(b)) <= 1:
        return np.nan
    return spearmanr(a,b).correlation


def metrics(y,p):
    return dict(R2=r2_score(y,p), MAE=mean_absolute_error(y,p), RMSE=mean_squared_error(y,p)**0.5, Spearman=safe_spearman(y,p))


def repeated_cv(df, features, target, model):
    X=df[features]; y=df[target]
    cv=RepeatedKFold(n_splits=5,n_repeats=20,random_state=RANDOM_STATE)
    rows=[]
    for tr,te in cv.split(X):
        m=clone(model); m.fit(X.iloc[tr],y.iloc[tr]); p=m.predict(X.iloc[te])
        rows.append(metrics(y.iloc[te],p))
    d=pd.DataFrame(rows)
    return {f'{k}_mean':d[k].mean() for k in d.columns} | {f'{k}_std':d[k].std() for k in d.columns}


def lofo(df, features, target, model):
    X=df[features]; y=df[target]; g=df['Formulation_label'].astype(str)
    obs=[]; pred=[]
    for grp in sorted(g.unique()):
        tr=g!=grp; te=g==grp
        m=clone(model); m.fit(X.loc[tr],y.loc[tr]); p=m.predict(X.loc[te])
        obs.extend(y.loc[te].tolist()); pred.extend(p.tolist())
    return metrics(np.array(obs),np.array(pred))


def feature_sets(version):
    eng = OLD_ENG if version=='old' else CLEAN_ENG
    ang = OLD_ANGLE if version=='old' else CLEAN_ANGLE
    return {
        'core': CORE,
        'engineered': CORE + eng,
        'rheology_enhanced': CORE + eng + RHEO,
        'angle_engineered': CORE + eng + ang,
        'angle_rheology_enhanced': CORE + eng + ang + RHEO,
    }

# Audit reviewer-identified deterministic relationships.
af=pd.read_csv(DATA/'modeling_dataset_AF.csv')
qm=pd.read_csv(DATA/'modeling_dataset_Qm.csv')
audit = {
    'TP_minus_10x_TP_pct_max_abs': float(np.max(np.abs(qm['TP'] - 10*qm['TP_pct_wv']))),
    'HF_plus_Alg_fraction_minus_1_max_abs': float(np.max(np.abs(qm['HF'] + qm['Alg_fraction_feature'] - 1))),
    'fraction_product_residual_max_abs': float(np.max(np.abs(qm['Alg_fraction_x_HA_fraction'] - qm['HF']*(1-qm['HF'])))),
    'Angle_rad_minus_deg2rad_max_abs': float(np.max(np.abs(af['Angle_rad'] - np.deg2rad(af['Angle_deg'])))),
    'Angle_deg_cos_correlation': float(af[['Angle_deg','cos_angle']].corr().iloc[0,1]),
    'sampled_angles_deg': sorted([float(x) for x in af['Angle_deg'].unique()]),
    'removed_features': REMOVED,
    'retained_general_engineered_features': CLEAN_ENG,
    'retained_angle_features': CLEAN_ANGLE,
}
(OUT/'feature_cleanup_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')

rows=[]
for version in ['old','clean']:
    fs=feature_sets(version)
    for target,(fn,ycol,allowed) in TARGETS.items():
        df=pd.read_csv(DATA/fn).dropna(subset=[ycol]).reset_index(drop=True)
        for fsname in allowed:
            feats=[x for x in fs[fsname] if x in df.columns]
            if df[feats].isna().sum().sum():
                continue
            for model_name,model in get_models().items():
                r=repeated_cv(df,feats,ycol,model)
                l=lofo(df,feats,ycol,model)
                rows.append({
                    'Version':version,'Target':target,'Feature_set':fsname,'Model':model_name,
                    'N':len(df),'N_features':len(feats),'Feature_list':', '.join(feats),
                    'Repeated_R2_mean':r['R2_mean'],'Repeated_R2_std':r['R2_std'],
                    'Repeated_MAE_mean':r['MAE_mean'],'Repeated_RMSE_mean':r['RMSE_mean'],'Repeated_Spearman_mean':r['Spearman_mean'],
                    'LOFO_R2':l['R2'],'LOFO_MAE':l['MAE'],'LOFO_RMSE':l['RMSE'],'LOFO_Spearman':l['Spearman'],
                })

allres=pd.DataFrame(rows)
allres.to_csv(OUT/'old_vs_clean_all_model_results.csv',index=False)
allres.to_excel(OUT/'old_vs_clean_all_model_results.xlsx',index=False)

best_rows=[]
for version in ['old','clean']:
    for target in TARGETS:
        d=allres[(allres.Version==version)&(allres.Target==target)]
        br=d.sort_values(['Repeated_RMSE_mean','Repeated_MAE_mean']).iloc[0]
        bl=d.sort_values(['LOFO_RMSE','LOFO_MAE']).iloc[0]
        best_rows.append({
            'Version':version,'Target':target,
            'Best_repeated_feature_set':br.Feature_set,'Best_repeated_model':br.Model,'Best_repeated_N_features':int(br.N_features),
            'Repeated_R2_mean':br.Repeated_R2_mean,'Repeated_MAE_mean':br.Repeated_MAE_mean,'Repeated_RMSE_mean':br.Repeated_RMSE_mean,'Repeated_Spearman_mean':br.Repeated_Spearman_mean,
            'Best_LOFO_feature_set':bl.Feature_set,'Best_LOFO_model':bl.Model,'Best_LOFO_N_features':int(bl.N_features),
            'LOFO_R2':bl.LOFO_R2,'LOFO_MAE':bl.LOFO_MAE,'LOFO_RMSE':bl.LOFO_RMSE,'LOFO_Spearman':bl.LOFO_Spearman,
        })
best=pd.DataFrame(best_rows)
best.to_csv(OUT/'old_vs_clean_best_model_summary.csv',index=False)
best.to_excel(OUT/'old_vs_clean_best_model_summary.xlsx',index=False)

# Direct paired change table for clean vs old best-by-target.
old=best[best.Version=='old'].set_index('Target')
clean=best[best.Version=='clean'].set_index('Target')
comp=[]
for t in TARGETS:
    comp.append({
        'Target':t,
        'Old_repeated_model':old.loc[t,'Best_repeated_model'],
        'Clean_repeated_model':clean.loc[t,'Best_repeated_model'],
        'Old_repeated_feature_set':old.loc[t,'Best_repeated_feature_set'],
        'Clean_repeated_feature_set':clean.loc[t,'Best_repeated_feature_set'],
        'Old_repeated_R2':old.loc[t,'Repeated_R2_mean'],
        'Clean_repeated_R2':clean.loc[t,'Repeated_R2_mean'],
        'Delta_repeated_R2':clean.loc[t,'Repeated_R2_mean']-old.loc[t,'Repeated_R2_mean'],
        'Old_LOFO_model':old.loc[t,'Best_LOFO_model'],
        'Clean_LOFO_model':clean.loc[t,'Best_LOFO_model'],
        'Old_LOFO_feature_set':old.loc[t,'Best_LOFO_feature_set'],
        'Clean_LOFO_feature_set':clean.loc[t,'Best_LOFO_feature_set'],
        'Old_LOFO_R2':old.loc[t,'LOFO_R2'],
        'Clean_LOFO_R2':clean.loc[t,'LOFO_R2'],
        'Delta_LOFO_R2':clean.loc[t,'LOFO_R2']-old.loc[t,'LOFO_R2'],
    })
comp=pd.DataFrame(comp)
comp.to_csv(OUT/'old_vs_clean_best_model_comparison.csv',index=False)
comp.to_excel(OUT/'old_vs_clean_best_model_comparison.xlsx',index=False)

print('\nFEATURE CLEANUP AUDIT')
print(json.dumps(audit,indent=2))
print('\nBEST MODEL SUMMARY')
print(best.to_string(index=False))
print('\nCOMPARISON')
print(comp.to_string(index=False))
