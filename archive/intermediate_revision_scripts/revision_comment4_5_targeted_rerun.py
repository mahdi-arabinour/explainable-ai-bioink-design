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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed'
OUT=ROOT/'results'/'revision_comments_4_5'; OUT.mkdir(parents=True,exist_ok=True)
SEED=42
CORE=['Alg_mg_ml','HA_mg_ml','Pressure_kPa']
RHEO=['Eta0_mean_Pa_s','Cross_m_mean','Cross_k_mean_s','G_prime_mean_Pa','G_double_prime_mean_Pa','Yield_stress_mean_Pa']
CLEAN_ENG=['TP','HF','Alg_HA_ratio_safe','P_over_TP','P_over_Alg','P_over_HA','Alg_x_HA','Alg_x_P','HA_x_P','TP_x_P','HF_x_P']
CLEAN_ANGLE=['Angle_deg','P_x_Angle','Alg_x_Angle','HA_x_Angle','TP_x_Angle']

def safe_spear(y,p):
    if len(np.unique(y))<=1 or len(np.unique(p))<=1: return np.nan
    return spearmanr(y,p).correlation

def metr(y,p):
    return {'R2':r2_score(y,p),'MAE':mean_absolute_error(y,p),'RMSE':mean_squared_error(y,p)**0.5,'Spearman':safe_spear(y,p)}

def repeated(df,features,target,model):
    X=df[features]; y=df[target]; cv=RepeatedKFold(n_splits=5,n_repeats=20,random_state=SEED)
    rows=[]
    for tr,te in cv.split(X):
        m=clone(model); m.fit(X.iloc[tr],y.iloc[tr]); rows.append(metr(y.iloc[te],m.predict(X.iloc[te])))
    d=pd.DataFrame(rows)
    return {k+'_mean':d[k].mean() for k in d}, {k+'_std':d[k].std() for k in d}

def lofo(df,features,target,model):
    X=df[features]; y=df[target]; g=df['Formulation_label'].astype(str)
    obs=[];pred=[]
    for grp in sorted(g.unique()):
        tr=g!=grp; te=g==grp
        m=clone(model); m.fit(X.loc[tr],y.loc[tr]); p=m.predict(X.loc[te]); obs += y.loc[te].tolist(); pred += p.tolist()
    return metr(np.array(obs),np.array(pred))

models={
 'LinearRegression': Pipeline([('scaler',StandardScaler()),('model',LinearRegression())]),
 'SVR_RBF': Pipeline([('scaler',StandardScaler()),('model',SVR(kernel='rbf',C=10.0,epsilon=0.05))]),
 'RandomForest': RandomForestRegressor(n_estimators=300,max_depth=None,min_samples_leaf=2,random_state=SEED,n_jobs=-1),
 'XGBoost': XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=2,subsample=0.9,colsample_bytree=0.9,objective='reg:squarederror',random_state=SEED,n_jobs=-1),
}
# exact same selected model/feature-set roles as current Table 2, but with duplicate-free engineered matrices.
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
old_rep=pd.read_csv(ROOT/'supplementary/tables/037_tables_ml_model_performance_repeated_kfold.csv')
old_lo=pd.read_csv(ROOT/'supplementary/tables/035_tables_ml_model_performance_leave_one_formulation_out.csv')
rows=[]
for target,fn,ycol,val,fs,mn,features in jobs:
    print('Running',target,val,mn,'with',len(features),'features',flush=True)
    df=pd.read_csv(DATA/fn).dropna(subset=[ycol]).reset_index(drop=True)
    if val=='repeated':
        mean,std=repeated(df,features,ycol,models[mn])
        old=old_rep[(old_rep.Target==target)&(old_rep.Feature_set==fs)&(old_rep.Model==mn)].iloc[0]
        rows.append({'Target':target,'Validation':'Repeated_KFold','Feature_set':fs,'Model':mn,'Old_N_features':int(old.N_features),'Clean_N_features':len(features),
                     'Old_R2':old.R2_mean,'Clean_R2':mean['R2_mean'],'Delta_R2':mean['R2_mean']-old.R2_mean,
                     'Old_MAE':old.MAE_mean,'Clean_MAE':mean['MAE_mean'],'Old_RMSE':old.RMSE_mean,'Clean_RMSE':mean['RMSE_mean'],
                     'Old_Spearman':old.Spearman_mean,'Clean_Spearman':mean['Spearman_mean'],'Clean_feature_list':', '.join(features)})
    else:
        res=lofo(df,features,ycol,models[mn])
        old=old_lo[(old_lo.Target==target)&(old_lo.Feature_set==fs)&(old_lo.Model==mn)].iloc[0]
        rows.append({'Target':target,'Validation':'LOFO','Feature_set':fs,'Model':mn,'Old_N_features':int(old.N_features),'Clean_N_features':len(features),
                     'Old_R2':old.LOFO_R2,'Clean_R2':res['R2'],'Delta_R2':res['R2']-old.LOFO_R2,
                     'Old_MAE':old.LOFO_MAE,'Clean_MAE':res['MAE'],'Old_RMSE':old.LOFO_RMSE,'Clean_RMSE':res['RMSE'],
                     'Old_Spearman':old.LOFO_Spearman,'Clean_Spearman':res['Spearman'],'Clean_feature_list':', '.join(features)})

out=pd.DataFrame(rows)
out.to_csv(OUT/'targeted_old_vs_clean_selected_models.csv',index=False)
out.to_excel(OUT/'targeted_old_vs_clean_selected_models.xlsx',index=False)
print('\n',out.to_string(index=False),flush=True)
