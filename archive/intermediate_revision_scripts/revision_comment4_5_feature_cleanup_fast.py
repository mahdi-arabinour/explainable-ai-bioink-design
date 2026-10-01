from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'results' / 'revision_comments_4_5'
OUT.mkdir(parents=True, exist_ok=True)
SEED=42

TARGETS={
 'Pr':('modeling_dataset_Pr.csv','Pr_mean',['core','engineered','rheology_enhanced']),
 'SR':('modeling_dataset_SR.csv','SR_mean',['core','engineered','rheology_enhanced']),
 'Qm':('modeling_dataset_Qm.csv','Qm_mean_mg_s',['core','engineered','rheology_enhanced']),
 'AF':('modeling_dataset_AF.csv','AF_mean',['core','angle_engineered','angle_rheology_enhanced']),
}
CORE=['Alg_mg_ml','HA_mg_ml','Pressure_kPa']
RHEO=['Eta0_mean_Pa_s','Cross_m_mean','Cross_k_mean_s','G_prime_mean_Pa','G_double_prime_mean_Pa','Yield_stress_mean_Pa']
OLD_ENG=['TP','TP_pct_wv','HF','Alg_fraction_feature','Alg_HA_ratio_safe','P_over_TP','P_over_Alg','P_over_HA','Alg_x_HA','Alg_x_P','HA_x_P','TP_x_P','HF_x_P','Alg_fraction_x_HA_fraction']
OLD_ANGLE=['Angle_deg','Angle_rad','sin_angle','cos_angle','P_x_Angle','Alg_x_Angle','HA_x_Angle','TP_x_Angle']
CLEAN_ENG=['TP','HF','Alg_HA_ratio_safe','P_over_TP','P_over_Alg','P_over_HA','Alg_x_HA','Alg_x_P','HA_x_P','TP_x_P','HF_x_P']
CLEAN_ANGLE=['Angle_deg','P_x_Angle','Alg_x_Angle','HA_x_Angle','TP_x_Angle']
REMOVED={
 'TP_pct_wv':'exact scale duplicate of TP (TP = 10 * TP_pct_wv)',
 'Alg_fraction_feature':'exact complement of HF (Alg_fraction = 1 - HF)',
 'Alg_fraction_x_HA_fraction':'deterministic function HF * (1 - HF)',
 'Angle_rad':'exact scale duplicate of Angle_deg',
 'sin_angle':'deterministic transform of the same three sampled angle levels',
 'cos_angle':'affine duplicate of Angle_deg for sampled 60/90/120 degree levels',
}

def models():
 return {
  'LinearRegression':Pipeline([('scaler',StandardScaler()),('model',LinearRegression())]),
  'SVR_RBF':Pipeline([('scaler',StandardScaler()),('model',SVR(kernel='rbf',C=10.0,epsilon=0.05))]),
  'RandomForest':RandomForestRegressor(n_estimators=300,max_depth=None,min_samples_leaf=2,random_state=SEED,n_jobs=1),
  'GradientBoosting':GradientBoostingRegressor(n_estimators=200,learning_rate=0.05,max_depth=2,random_state=SEED),
  'XGBoost':XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=2,subsample=0.9,colsample_bytree=0.9,objective='reg:squarederror',random_state=SEED,n_jobs=1),
 }

def spear(y,p):
 if len(np.unique(y))<=1 or len(np.unique(p))<=1: return 0.0
 return float(spearmanr(y,p).correlation)
SCORING={
 'r2':'r2',
 'mae':'neg_mean_absolute_error',
 'rmse':'neg_root_mean_squared_error',
 'spearman':make_scorer(spear, greater_is_better=True),
}

def fsets(version):
 e=OLD_ENG if version=='old' else CLEAN_ENG
 a=OLD_ANGLE if version=='old' else CLEAN_ANGLE
 return {
  'core':CORE,
  'engineered':CORE+e,
  'rheology_enhanced':CORE+e+RHEO,
  'angle_engineered':CORE+e+a,
  'angle_rheology_enhanced':CORE+e+a+RHEO,
 }

# deterministic-feature audit
qm=pd.read_csv(DATA/'modeling_dataset_Qm.csv')
af=pd.read_csv(DATA/'modeling_dataset_AF.csv')
audit={
 'TP_minus_10x_TP_pct_max_abs':float(np.max(np.abs(qm.TP-10*qm.TP_pct_wv))),
 'HF_plus_Alg_fraction_minus_1_max_abs':float(np.max(np.abs(qm.HF+qm.Alg_fraction_feature-1))),
 'fraction_product_residual_max_abs':float(np.max(np.abs(qm.Alg_fraction_x_HA_fraction-qm.HF*(1-qm.HF)))),
 'Angle_rad_minus_deg2rad_max_abs':float(np.max(np.abs(af.Angle_rad-np.deg2rad(af.Angle_deg)))),
 'Angle_deg_cos_correlation':float(af[['Angle_deg','cos_angle']].corr().iloc[0,1]),
 'sampled_angles_deg':sorted(map(float,af.Angle_deg.unique())),
 'removed_features':REMOVED,
 'clean_general_features':CLEAN_ENG,
 'clean_angle_features':CLEAN_ANGLE,
}
(OUT/'feature_cleanup_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')

rows=[]
# Run clean grid only. Original grid is read from archived repository outputs below.
fs=fsets('clean')
for target,(fn,ycol,allowed) in TARGETS.items():
 df=pd.read_csv(DATA/fn).dropna(subset=[ycol]).reset_index(drop=True)
 y=df[ycol]
 groups=df['Formulation_label'].astype(str)
 for fsname in allowed:
  feats=[c for c in fs[fsname] if c in df.columns]
  X=df[feats]
  if X.isna().any().any(): continue
  for mname,m in models().items():
   print('RUN',target,fsname,mname,flush=True)
   rkf=RepeatedKFold(n_splits=5,n_repeats=20,random_state=SEED)
   cv=cross_validate(m,X,y,cv=rkf,scoring=SCORING,n_jobs=-1,return_train_score=False)
   logo=LeaveOneGroupOut()
   pred=cross_val_predict(m,X,y,cv=logo.split(X,y,groups),n_jobs=-1)
   rows.append({
    'Version':'clean','Target':target,'Feature_set':fsname,'Model':mname,'N':len(df),'N_features':len(feats),'Feature_list':', '.join(feats),
    'Repeated_R2_mean':np.mean(cv['test_r2']),'Repeated_R2_std':np.std(cv['test_r2'],ddof=1),
    'Repeated_MAE_mean':-np.mean(cv['test_mae']),'Repeated_RMSE_mean':-np.mean(cv['test_rmse']),'Repeated_Spearman_mean':np.mean(cv['test_spearman']),
    'LOFO_R2':r2_score(y,pred),'LOFO_MAE':mean_absolute_error(y,pred),'LOFO_RMSE':mean_squared_error(y,pred)**0.5,'LOFO_Spearman':spear(y,pred)
   })
clean=pd.DataFrame(rows)
clean.to_csv(OUT/'clean_feature_model_results.csv',index=False)
clean.to_excel(OUT/'clean_feature_model_results.xlsx',index=False)

# Load original archived results for matched comparison.
oldrep=pd.read_csv(ROOT/'supplementary/tables/037_tables_ml_model_performance_repeated_kfold.csv')
oldlofo=pd.read_csv(ROOT/'supplementary/tables/035_tables_ml_model_performance_leave_one_formulation_out.csv')

best=[]
for target in TARGETS:
 c=clean[clean.Target==target]
 cr=c.sort_values(['Repeated_RMSE_mean','Repeated_MAE_mean']).iloc[0]
 cl=c.sort_values(['LOFO_RMSE','LOFO_MAE']).iloc[0]
 orep=oldrep[oldrep.Target==target].sort_values(['RMSE_mean','MAE_mean']).iloc[0]
 olo=oldlofo[oldlofo.Target==target].sort_values(['LOFO_RMSE','LOFO_MAE']).iloc[0]
 best.append({
  'Target':target,
  'Old_repeated_feature_set':orep.Feature_set,'Old_repeated_model':orep.Model,'Old_repeated_N_features':int(orep.N_features),'Old_repeated_R2':orep.R2_mean,'Old_repeated_MAE':orep.MAE_mean,'Old_repeated_RMSE':orep.RMSE_mean,'Old_repeated_Spearman':orep.Spearman_mean,
  'Clean_repeated_feature_set':cr.Feature_set,'Clean_repeated_model':cr.Model,'Clean_repeated_N_features':int(cr.N_features),'Clean_repeated_R2':cr.Repeated_R2_mean,'Clean_repeated_MAE':cr.Repeated_MAE_mean,'Clean_repeated_RMSE':cr.Repeated_RMSE_mean,'Clean_repeated_Spearman':cr.Repeated_Spearman_mean,
  'Delta_repeated_R2':cr.Repeated_R2_mean-orep.R2_mean,
  'Old_LOFO_feature_set':olo.Feature_set,'Old_LOFO_model':olo.Model,'Old_LOFO_N_features':int(olo.N_features),'Old_LOFO_R2':olo.LOFO_R2,'Old_LOFO_MAE':olo.LOFO_MAE,'Old_LOFO_RMSE':olo.LOFO_RMSE,'Old_LOFO_Spearman':olo.LOFO_Spearman,
  'Clean_LOFO_feature_set':cl.Feature_set,'Clean_LOFO_model':cl.Model,'Clean_LOFO_N_features':int(cl.N_features),'Clean_LOFO_R2':cl.LOFO_R2,'Clean_LOFO_MAE':cl.LOFO_MAE,'Clean_LOFO_RMSE':cl.LOFO_RMSE,'Clean_LOFO_Spearman':cl.LOFO_Spearman,
  'Delta_LOFO_R2':cl.LOFO_R2-olo.LOFO_R2,
 })
best=pd.DataFrame(best)
best.to_csv(OUT/'old_vs_clean_best_model_comparison.csv',index=False)
best.to_excel(OUT/'old_vs_clean_best_model_comparison.xlsx',index=False)
print('\nAUDIT\n',json.dumps(audit,indent=2),flush=True)
print('\nBEST COMPARISON\n',best.to_string(index=False),flush=True)
