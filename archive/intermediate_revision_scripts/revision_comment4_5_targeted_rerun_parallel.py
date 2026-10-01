from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
from xgboost import XGBRegressor
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'processed'; OUT=ROOT/'results'/'revision_comments_4_5'; OUT.mkdir(parents=True,exist_ok=True)
SEED=42
CORE=['Alg_mg_ml','HA_mg_ml','Pressure_kPa']; RHEO=['Eta0_mean_Pa_s','Cross_m_mean','Cross_k_mean_s','G_prime_mean_Pa','G_double_prime_mean_Pa','Yield_stress_mean_Pa']
CLEAN_ENG=['TP','HF','Alg_HA_ratio_safe','P_over_TP','P_over_Alg','P_over_HA','Alg_x_HA','Alg_x_P','HA_x_P','TP_x_P','HF_x_P']
CLEAN_ANGLE=['Angle_deg','P_x_Angle','Alg_x_Angle','HA_x_Angle','TP_x_Angle']
def spear(y,p):
    if len(np.unique(y))<=1 or len(np.unique(p))<=1: return 0.0
    return float(spearmanr(y,p).correlation)
score={'r2':'r2','mae':'neg_mean_absolute_error','rmse':'neg_root_mean_squared_error','spearman':make_scorer(spear)}
models={
 'LinearRegression':Pipeline([('scaler',StandardScaler()),('model',LinearRegression())]),
 'SVR_RBF':Pipeline([('scaler',StandardScaler()),('model',SVR(kernel='rbf',C=10.0,epsilon=0.05))]),
 'RandomForest':RandomForestRegressor(n_estimators=300,max_depth=None,min_samples_leaf=2,random_state=SEED,n_jobs=1),
 'XGBoost':XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=2,subsample=0.9,colsample_bytree=0.9,objective='reg:squarederror',random_state=SEED,n_jobs=1),
}
jobs=[
 ('Pr','modeling_dataset_Pr.csv','Pr_mean','repeated','rheology_enhanced','RandomForest',CORE+CLEAN_ENG+RHEO),
 ('Pr','modeling_dataset_Pr.csv','Pr_mean','LOFO','core','LinearRegression',CORE),
 ('SR','modeling_dataset_SR.csv','SR_mean','repeated','engineered','SVR_RBF',CORE+CLEAN_ENG),
 ('SR','modeling_dataset_SR.csv','SR_mean','LOFO','rheology_enhanced','SVR_RBF',CORE+CLEAN_ENG+RHEO),
 ('Qm','modeling_dataset_Qm.csv','Qm_mean_mg_s','repeated','engineered','SVR_RBF',CORE+CLEAN_ENG),
 ('Qm','modeling_dataset_Qm.csv','Qm_mean_mg_s','LOFO','core','RandomForest',CORE),
 ('AF','modeling_dataset_AF.csv','AF_mean','repeated','angle_engineered','RandomForest',CORE+CLEAN_ENG+CLEAN_ANGLE),
 ('AF','modeling_dataset_AF.csv','AF_mean','LOFO','angle_engineered','XGBoost',CORE+CLEAN_ENG+CLEAN_ANGLE),
]
oldrep=pd.read_csv(ROOT/'supplementary/tables/037_tables_ml_model_performance_repeated_kfold.csv'); oldlo=pd.read_csv(ROOT/'supplementary/tables/035_tables_ml_model_performance_leave_one_formulation_out.csv')
rows=[]
for target,fn,ycol,val,fs,mn,features in jobs:
    print('RUN',target,val,mn,len(features),flush=True)
    df=pd.read_csv(DATA/fn).dropna(subset=[ycol]).reset_index(drop=True); X=df[features]; y=df[ycol]
    if val=='repeated':
        cv=RepeatedKFold(n_splits=5,n_repeats=20,random_state=SEED)
        rr=cross_validate(models[mn],X,y,cv=cv,scoring=score,n_jobs=-1,pre_dispatch='2*n_jobs')
        vals={'R2':rr['test_r2'].mean(),'MAE':-rr['test_mae'].mean(),'RMSE':-rr['test_rmse'].mean(),'Spearman':rr['test_spearman'].mean()}
        old=oldrep[(oldrep.Target==target)&(oldrep.Feature_set==fs)&(oldrep.Model==mn)].iloc[0]
        olds={'R2':old.R2_mean,'MAE':old.MAE_mean,'RMSE':old.RMSE_mean,'Spearman':old.Spearman_mean}; oldn=int(old.N_features)
    else:
        logo=LeaveOneGroupOut(); groups=df.Formulation_label.astype(str)
        pred=cross_val_predict(models[mn],X,y,cv=logo.split(X,y,groups),n_jobs=-1,pre_dispatch='2*n_jobs')
        vals={'R2':r2_score(y,pred),'MAE':mean_absolute_error(y,pred),'RMSE':mean_squared_error(y,pred)**0.5,'Spearman':spear(y,pred)}
        old=oldlo[(oldlo.Target==target)&(oldlo.Feature_set==fs)&(oldlo.Model==mn)].iloc[0]
        olds={'R2':old.LOFO_R2,'MAE':old.LOFO_MAE,'RMSE':old.LOFO_RMSE,'Spearman':old.LOFO_Spearman}; oldn=int(old.N_features)
    rows.append({'Target':target,'Validation':val,'Feature_set':fs,'Model':mn,'Old_N_features':oldn,'Clean_N_features':len(features),
                 'Old_R2':olds['R2'],'Clean_R2':vals['R2'],'Delta_R2':vals['R2']-olds['R2'],'Old_MAE':olds['MAE'],'Clean_MAE':vals['MAE'],'Old_RMSE':olds['RMSE'],'Clean_RMSE':vals['RMSE'],'Old_Spearman':olds['Spearman'],'Clean_Spearman':vals['Spearman'],'Clean_feature_list':', '.join(features)})
out=pd.DataFrame(rows); out.to_csv(OUT/'targeted_old_vs_clean_selected_models.csv',index=False); out.to_excel(OUT/'targeted_old_vs_clean_selected_models.xlsx',index=False)
print(out.to_string(index=False),flush=True)
