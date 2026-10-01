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

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'processed'; OUT=ROOT/'results'/'revision_comments_6_7_R1_8_R2'; OUT.mkdir(parents=True,exist_ok=True)
SEED=42
FILES={'Pr':('modeling_dataset_Pr.csv','Pr_mean'),'SR':('modeling_dataset_SR.csv','SR_mean'),'Qm':('modeling_dataset_Qm.csv','Qm_mean_mg_s'),'AF':('modeling_dataset_AF.csv','AF_mean')}
CORE={'Pr':['Alg_mg_ml','HA_mg_ml','Pressure_kPa'],'SR':['Alg_mg_ml','HA_mg_ml','Pressure_kPa'],'Qm':['Alg_mg_ml','HA_mg_ml','Pressure_kPa'],'AF':['Alg_mg_ml','HA_mg_ml','Pressure_kPa','Angle_deg']}
GEN=['TP','HF','Alg_HA_ratio_safe','P_over_TP','P_over_Alg','P_over_HA','Alg_x_P','HA_x_P','TP_x_P','HF_x_P','Alg_x_HA']
ANG=['P_x_Angle','Alg_x_Angle','HA_x_Angle','TP_x_Angle']
# Preserve the model identities used by the submitted Table S19 comparison while re-running
# duplicate-reduced core and engineered representations. Because the submitted core and
# engineered algorithms differ in two target-validation rows, this is not uniformly a
# fixed-model feature ablation. Same-algorithm rows can be interpreted as representation
# sensitivity conditional on the algorithm; different-algorithm rows are continuity comparisons.
MODELS={
 ('AF','LOFO'):('RandomForest','RandomForest'),('AF','Repeated_KFold'):('RandomForest','RandomForest'),
 ('Pr','LOFO'):('LinearRegression','SVR_RBF'),('Pr','Repeated_KFold'):('RandomForest','RandomForest'),
 ('Qm','LOFO'):('SVR_RBF','SVR_RBF'),('Qm','Repeated_KFold'):('SVR_RBF','SVR_RBF'),
 ('SR','LOFO'):('SVR_RBF','SVR_RBF'),('SR','Repeated_KFold'):('LinearRegression','SVR_RBF'),
}
def model(name):
 if name=='LinearRegression': return Pipeline([('scaler',StandardScaler()),('model',LinearRegression())])
 if name=='SVR_RBF': return Pipeline([('scaler',StandardScaler()),('model',SVR(kernel='rbf',C=10.0,epsilon=0.05))])
 if name=='RandomForest': return RandomForestRegressor(n_estimators=300,random_state=SEED,min_samples_leaf=1,n_jobs=1)
 raise ValueError(name)
def sp(y,p):
 v=spearmanr(y,p).correlation
 return 0.0 if pd.isna(v) else float(v)
SCORING={'r2':'r2','mae':'neg_mean_absolute_error','rmse':'neg_root_mean_squared_error','spearman':make_scorer(sp)}
def eval_rep(m,X,y):
 z=cross_validate(m,X,y,cv=RepeatedKFold(n_splits=5,n_repeats=20,random_state=SEED),scoring=SCORING,n_jobs=-1,pre_dispatch='2*n_jobs')
 return {'R2':z['test_r2'].mean(),'MAE':-z['test_mae'].mean(),'RMSE':-z['test_rmse'].mean(),'Spearman':z['test_spearman'].mean()}
def eval_lofo(m,X,y,g):
 pred=cross_val_predict(m,X,y,cv=LeaveOneGroupOut().split(X,y,g),n_jobs=-1,pre_dispatch='2*n_jobs')
 return {'R2':r2_score(y,pred),'MAE':mean_absolute_error(y,pred),'RMSE':mean_squared_error(y,pred)**0.5,'Spearman':sp(y,pred)}
rows=[]
for target,(fn,ycol) in FILES.items():
 df=pd.read_csv(DATA/fn).dropna(subset=[ycol]).reset_index(drop=True); y=df[ycol]; g=df.Formulation_label.astype(str)
 eng=CORE[target]+GEN+(ANG if target=='AF' else []); eng=list(dict.fromkeys(eng))
 for val in ['LOFO','Repeated_KFold']:
  cm,em=MODELS[(target,val)]
  for setname,mname,features in [('core',cm,CORE[target]),('engineered',em,eng)]:
   print('RUN',target,val,setname,mname,len(features),flush=True)
   res=eval_lofo(model(mname),df[features],y,g) if val=='LOFO' else eval_rep(model(mname),df[features],y)
   rows.append({'Target':target,'Validation':val,'Feature_set':setname,'Model':mname,'N':len(df),'N_features':len(features),**res,'Feature_list':', '.join(features)})
raw=pd.DataFrame(rows); raw.to_csv(OUT/'Ablation_Selected_Models_Raw_Comments_6-7-R1_8-R2.csv',index=False)
summary=[]
for (t,v),g in raw.groupby(['Target','Validation']):
 c=g[g.Feature_set=='core'].iloc[0]; e=g[g.Feature_set=='engineered'].iloc[0]
 summary.append({'Target':t,'Validation':v,'Core_model':c.Model,'Engineered_model':e.Model,'Core_N_features':int(c.N_features),'Engineered_N_features':int(e.N_features),'Core_R2':c.R2,'Engineered_R2':e.R2,'Delta_R2_engineered_minus_core':e.R2-c.R2,'Core_MAE':c.MAE,'Engineered_MAE':e.MAE,'Delta_MAE_core_minus_engineered':c.MAE-e.MAE,'Core_RMSE':c.RMSE,'Engineered_RMSE':e.RMSE,'Delta_RMSE_core_minus_engineered':c.RMSE-e.RMSE,'Core_Spearman':c.Spearman,'Engineered_Spearman':e.Spearman,'Delta_Spearman_engineered_minus_core':e.Spearman-c.Spearman,'Core_features':c.Feature_list,'Engineered_features':e.Feature_list})
summary=pd.DataFrame(summary).sort_values(['Target','Validation']).reset_index(drop=True)
summary.to_csv(OUT/'Table_S19_Ablation_Comments_6-7-R1_8-R2.csv',index=False)
# Table 2 reconciliation
t2=pd.read_csv(ROOT/'results'/'revision_comments_4_R1_16_R2'/'Reportable_Nested_Performance_Comments_4-R1_16-R2.csv')
comp=[]
for _,r in summary.iterrows():
 tr=t2[t2.Target==r.Target].iloc[0]
 if r.Validation=='Repeated_KFold':
  nr2=tr['Nested repeated R2 mean']; nmae=tr['Nested repeated MAE mean']; nrmse=tr['Nested repeated RMSE mean']; nsp=tr['Nested repeated Spearman mean']; nmode=tr['Nested repeated selection mode']
 else:
  nr2=tr['Nested LOFO R2']; nmae=tr['Nested LOFO MAE']; nrmse=tr['Nested LOFO RMSE']; nsp=tr['Nested LOFO Spearman']; nmode=tr['Nested LOFO selection mode']
 comp.append({'Target':r.Target,'Validation':r.Validation,'Table2_nested_selection_mode':nmode,'Table2_nested_R2':nr2,'Table2_nested_MAE':nmae,'Table2_nested_RMSE':nrmse,'Table2_nested_Spearman':nsp,'TableS19_core_model':r.Core_model,'TableS19_engineered_model':r.Engineered_model,'TableS19_core_R2':r.Core_R2,'TableS19_engineered_R2':r.Engineered_R2,'Same_algorithm_core_vs_engineered': bool(r.Core_model == r.Engineered_model), 'Why_values_differ':'Table 2 reports selection-aware nested outer-test performance after inner algorithm/feature-set selection. Table S19 preserves the submitted core/engineered model identities after duplicate-feature cleanup and is not a second estimate of Table 2 performance. Same-algorithm rows support feature-representation sensitivity conditional on that algorithm; rows with different algorithms reflect combined model-plus-representation changes.'})
comp=pd.DataFrame(comp); comp.to_csv(OUT/'Table2_vs_TableS19_Reconciliation_Comments_6-7-R1_8-R2.csv',index=False)
meta={'comments':['6-R1','7-R1','8-R2'],'seed':SEED,'repeated_validation':'5-fold x 20 repeats; split-wise metrics averaged','lofo_validation':'LeaveOneGroupOut by formulation; pooled predictions','model_identity':'Model identities were preserved from the submitted Table S19 comparison and re-run after duplicate-feature cleanup. Six of eight target-validation comparisons use the same algorithm for core and engineered representations; Pr LOFO and SR repeated K-fold use different submitted algorithms and are not interpreted as pure feature-ablation effects.','delta_R2':'Engineered R2 - Core R2','delta_MAE':'Core MAE - Engineered MAE; positive means lower error for engineered','delta_RMSE':'Core RMSE - Engineered RMSE; positive means lower error for engineered','delta_Spearman':'Engineered Spearman - Core Spearman','table2_vs_s19':'Table 2 = nested selection-aware performance. Table S19 = submitted core-versus-engineered representation comparison after duplicate-feature cleanup; same-algorithm rows are representation sensitivity, while different-algorithm rows are combined model-plus-representation comparisons.'}
(OUT/'Ablation_Metadata_Comments_6-7-R1_8-R2.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
xlsx=OUT/'Ablation_Analysis_Comments_6-7-R1_8-R2.xlsx'
with pd.ExcelWriter(xlsx,engine='openpyxl') as w:
 summary.to_excel(w,index=False,sheet_name='Table_S19')
 raw.to_excel(w,index=False,sheet_name='Raw_Rerun')
 comp.to_excel(w,index=False,sheet_name='Table2_vs_S19')
 pd.DataFrame([{'Item':k,'Value':json.dumps(v) if isinstance(v,list) else v} for k,v in meta.items()]).to_excel(w,index=False,sheet_name='Methodology')
print(summary.to_string(index=False)); print('Saved',xlsx)
