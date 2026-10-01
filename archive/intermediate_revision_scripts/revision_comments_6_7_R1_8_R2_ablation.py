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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed'
OUT=ROOT/'results'/'revision_comments_6_7_R1_8_R2'; OUT.mkdir(parents=True,exist_ok=True)
SEED=42
TARGETS={
 'Pr':('modeling_dataset_Pr.csv','Pr_mean',['Alg_mg_ml','HA_mg_ml','Pressure_kPa']),
 'SR':('modeling_dataset_SR.csv','SR_mean',['Alg_mg_ml','HA_mg_ml','Pressure_kPa']),
 'Qm':('modeling_dataset_Qm.csv','Qm_mean_mg_s',['Alg_mg_ml','HA_mg_ml','Pressure_kPa']),
 'AF':('modeling_dataset_AF.csv','AF_mean',['Alg_mg_ml','HA_mg_ml','Pressure_kPa','Angle_deg']),
}
ENG_GENERAL=['TP','HF','Alg_HA_ratio_safe','P_over_TP','P_over_Alg','P_over_HA','Alg_x_P','HA_x_P','TP_x_P','HF_x_P','Alg_x_HA']
ENG_ANGLE=['P_x_Angle','Alg_x_Angle','HA_x_Angle','TP_x_Angle']
# Fixed algorithm chosen a priori from the most frequently selected algorithm in the nested analysis,
# collapsing across feature-set labels. This keeps the algorithm identical between core and engineered
# variants so Table S19 isolates the feature-set effect rather than mixing model and feature selection.
FIXED={
 ('Pr','Repeated_KFold'):'RandomForest', ('Pr','LOFO'):'RandomForest',
 ('SR','Repeated_KFold'):'SVR_RBF', ('SR','LOFO'):'SVR_RBF',
 ('Qm','Repeated_KFold'):'SVR_RBF', ('Qm','LOFO'):'LinearRegression',
 ('AF','Repeated_KFold'):'RandomForest', ('AF','LOFO'):'RandomForest',
}

def get_model(name):
 if name=='LinearRegression': return Pipeline([('scaler',StandardScaler()),('model',LinearRegression())])
 if name=='SVR_RBF': return Pipeline([('scaler',StandardScaler()),('model',SVR(kernel='rbf',C=10.0,epsilon=0.05))])
 if name=='RandomForest': return RandomForestRegressor(n_estimators=300,random_state=SEED,min_samples_leaf=1,n_jobs=1)
 raise ValueError(name)

def sp(y,p):
 if len(np.unique(y))<2 or len(np.unique(p))<2: return 0.0
 v=spearmanr(y,p).correlation
 return 0.0 if pd.isna(v) else float(v)
SCORING={'r2':'r2','mae':'neg_mean_absolute_error','rmse':'neg_root_mean_squared_error','spearman':make_scorer(sp)}

def groups(df):
 return df['Formulation_label'].astype(str) if 'Formulation_label' in df else ((df.Alg_mg_ml/10).round().astype(int).astype(str)+'ALG'+(df.HA_mg_ml/10).round().astype(int).astype(str)+'HA')

def eval_repeated(model,X,y):
 cv=RepeatedKFold(n_splits=5,n_repeats=20,random_state=SEED)
 z=cross_validate(model,X,y,cv=cv,scoring=SCORING,n_jobs=-1,pre_dispatch='2*n_jobs')
 return {'R2':float(z['test_r2'].mean()),'MAE':float(-z['test_mae'].mean()),'RMSE':float(-z['test_rmse'].mean()),'Spearman':float(z['test_spearman'].mean())}

def eval_lofo(model,X,y,g):
 cv=LeaveOneGroupOut(); pred=cross_val_predict(model,X,y,cv=cv.split(X,y,g),n_jobs=-1,pre_dispatch='2*n_jobs')
 return {'R2':float(r2_score(y,pred)),'MAE':float(mean_absolute_error(y,pred)),'RMSE':float(mean_squared_error(y,pred)**0.5),'Spearman':sp(y,pred)}

rows=[]
for target,(fn,ycol,core) in TARGETS.items():
 df=pd.read_csv(DATA/fn).dropna(subset=[ycol]).reset_index(drop=True)
 eng=core+[c for c in ENG_GENERAL if c in df]
 if target=='AF': eng += [c for c in ENG_ANGLE if c in df]
 eng=list(dict.fromkeys(eng)); y=df[ycol]; g=groups(df)
 for val in ['Repeated_KFold','LOFO']:
  mname=FIXED[(target,val)]
  for setname,features in [('core',core),('engineered',eng)]:
   print('RUN',target,val,setname,mname,len(features),flush=True)
   model=get_model(mname); X=df[features]
   res=eval_repeated(model,X,y) if val=='Repeated_KFold' else eval_lofo(model,X,y,g)
   rows.append({'Target':target,'Validation':val,'Fixed_model':mname,'Feature_set':setname,'N':len(df),'N_features':len(features),**res,'Feature_list':', '.join(features)})

raw=pd.DataFrame(rows)
raw.to_csv(OUT/'Ablation_Fixed_Model_Raw_Comments_6-7-R1_8-R2.csv',index=False)
summary=[]
for (t,v,m),g in raw.groupby(['Target','Validation','Fixed_model']):
 c=g[g.Feature_set=='core'].iloc[0]; e=g[g.Feature_set=='engineered'].iloc[0]
 summary.append({
   'Target':t,'Validation':v,'Fixed_model':m,
   'Core_N_features':int(c.N_features),'Engineered_N_features':int(e.N_features),
   'Core_R2':c.R2,'Engineered_R2':e.R2,'Delta_R2_engineered_minus_core':e.R2-c.R2,
   'Core_MAE':c.MAE,'Engineered_MAE':e.MAE,'Delta_MAE_core_minus_engineered':c.MAE-e.MAE,
   'Core_RMSE':c.RMSE,'Engineered_RMSE':e.RMSE,'Delta_RMSE_core_minus_engineered':c.RMSE-e.RMSE,
   'Core_Spearman':c.Spearman,'Engineered_Spearman':e.Spearman,'Delta_Spearman_engineered_minus_core':e.Spearman-c.Spearman,
   'Core_features':c.Feature_list,'Engineered_features':e.Feature_list,
 })
summary=pd.DataFrame(summary).sort_values(['Target','Validation']).reset_index(drop=True)
summary.to_csv(OUT/'Table_S19_Ablation_Comments_6-7-R1_8-R2.csv',index=False)

# Read Table 2 reportable nested performance and make a reconciliation sheet.
t2=pd.read_csv(ROOT/'results'/'revision_comments_4_R1_16_R2'/'Reportable_Nested_Performance_Comments_4-R1_16-R2.csv')
comp=[]
for _,r in summary.iterrows():
 tr=t2[t2.Target==r.Target].iloc[0]
 if r.Validation=='Repeated_KFold':
  nr2=tr['Repeated_R2_mean']; nmae=tr['Repeated_MAE_mean']; nrmse=tr['Repeated_RMSE_mean']; ns=tr['Repeated_Spearman_mean']
 else:
  nr2=tr['LOFO_R2']; nmae=tr['LOFO_MAE']; nrmse=tr['LOFO_RMSE']; ns=tr['LOFO_Spearman']
 comp.append({
   'Target':r.Target,'Validation':r.Validation,
   'Table2_nested_R2':nr2,'Table2_nested_MAE':nmae,'Table2_nested_RMSE':nrmse,'Table2_nested_Spearman':ns,
   'TableS19_fixed_model':r.Fixed_model,'TableS19_core_R2':r.Core_R2,'TableS19_engineered_R2':r.Engineered_R2,
   'Explanation':'Table 2 reports selection-aware outer-test performance from nested algorithm/feature-set selection. Table S19 holds the algorithm fixed and changes only the feature set, so it is a controlled feature-ablation sensitivity analysis and is not expected to reproduce Table 2 values.'
 })
comp=pd.DataFrame(comp)
comp.to_csv(OUT/'Table2_vs_TableS19_Reconciliation_Comments_6-7-R1_8-R2.csv',index=False)

meta={
 'comments':['6-R1','7-R1','8-R2'],
 'seed':SEED,
 'repeated_validation':'Repeated 5-fold CV, 20 repeats; same algorithm used for core and engineered comparison.',
 'lofo_validation':'Leave-one-formulation-out; pooled predictions; same algorithm used for core and engineered comparison.',
 'fixed_model_rule':'For each target and validation strategy, the algorithm was chosen from the most frequently selected algorithm in the previously completed nested analysis after collapsing across feature-set labels. The algorithm was then held fixed for the core-versus-engineered ablation.',
 'delta_R2':'Engineered R2 - Core R2',
 'delta_MAE':'Core MAE - Engineered MAE; positive means lower error with engineered features',
 'delta_RMSE':'Core RMSE - Engineered RMSE; positive means lower error with engineered features',
 'delta_Spearman':'Engineered Spearman - Core Spearman',
 'table2_tableS19_distinction':'Table 2 is nested, selection-aware performance. Table S19 is controlled feature-ablation sensitivity analysis.'
}
(OUT/'Ablation_Metadata_Comments_6-7-R1_8-R2.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

xlsx=OUT/'Ablation_Analysis_Comments_6-7-R1_8-R2.xlsx'
with pd.ExcelWriter(xlsx,engine='openpyxl') as w:
 summary.to_excel(w,index=False,sheet_name='Table_S19')
 raw.to_excel(w,index=False,sheet_name='Raw_Ablation')
 comp.to_excel(w,index=False,sheet_name='Table2_vs_S19')
 pd.DataFrame([{'Item':k,'Value':json.dumps(v) if isinstance(v,list) else v} for k,v in meta.items()]).to_excel(w,index=False,sheet_name='Methodology')
print('\nTABLE S19 CONTROLLED ABLATION')
print(summary.to_string(index=False))
print('\nSaved',xlsx)
