from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
import shap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'results' / 'revision_comments_8_R1_17_18_R2'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
np.random.seed(SEED)
N_PERM = 10
N_XAI_REPEATS = 5

# Duplicate-reduced feature definitions retained after Comments 4-5-R2.
CORE = ['Alg_mg_ml','HA_mg_ml','Pressure_kPa']
ENG = ['TP','HF','Alg_HA_ratio_safe','P_over_TP','P_over_Alg','P_over_HA','Alg_x_HA','Alg_x_P','HA_x_P','TP_x_P','HF_x_P']
RHEO = ['Eta0_mean_Pa_s','Cross_m_mean','Cross_k_mean_s','G_prime_mean_Pa','G_double_prime_mean_Pa','Yield_stress_mean_Pa']
ANGLE = ['Angle_deg','P_x_Angle','Alg_x_Angle','HA_x_Angle','TP_x_Angle']
FEATURE_SETS = {
    'core': CORE,
    'engineered': CORE + ENG,
    'rheology_enhanced': CORE + ENG + RHEO,
    'angle_engineered': CORE + ENG + ANGLE,
    'angle_rheology_enhanced': CORE + ENG + ANGLE + RHEO,
}
TARGETS = {
    'Pr': ('modeling_dataset_Pr.csv','Pr_mean'),
    'SR': ('modeling_dataset_SR.csv','SR_mean'),
    'Qm': ('modeling_dataset_Qm.csv','Qm_mean_mg_s'),
    'AF': ('modeling_dataset_AF.csv','AF_mean'),
}
# Preserve the target-specific reference models used by the manuscript's downstream XAI/candidate workflow.
# Performance claims remain based on the separately reported nested-validation estimates.
REFERENCE = {
    'Pr': ('core','LinearRegression'),
    'SR': ('rheology_enhanced','SVR_RBF'),
    'Qm': ('engineered','SVR_RBF'),
    'AF': ('angle_engineered','XGBoost'),
}

def models():
    return {
        'LinearRegression': Pipeline([('scaler',StandardScaler()),('model',LinearRegression())]),
        'SVR_RBF': Pipeline([('scaler',StandardScaler()),('model',SVR(kernel='rbf',C=10.0,epsilon=0.05))]),
        'RandomForest': RandomForestRegressor(n_estimators=300,max_depth=None,min_samples_leaf=2,random_state=SEED,n_jobs=1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200,learning_rate=0.05,max_depth=2,random_state=SEED),
        'XGBoost': XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=2,subsample=0.9,colsample_bytree=0.9,objective='reg:squarederror',random_state=SEED,n_jobs=1),
    }

def rmse(y, pred):
    return float(mean_squared_error(y, pred) ** 0.5)

def fixed_reference_objects():
    out = {}
    for target,(fn,ycol) in TARGETS.items():
        df = pd.read_csv(DATA/fn).dropna(subset=[ycol]).reset_index(drop=True)
        fs,mn = REFERENCE[target]
        features = [f for f in FEATURE_SETS[fs] if f in df.columns]
        X = df[features].copy(); y=df[ycol].copy()
        if X.isna().any().any():
            raise ValueError(f'Missing X values for {target}')
        out[target] = dict(df=df,X=X,y=y,ycol=ycol,feature_set=fs,model_name=mn,features=features,model=models()[mn])
    return out

def oof_permutation_for_splits(target, obj, split_iter, validation_name):
    X=obj['X']; y=obj['y']; features=obj['features']; base_model=obj['model']
    detail=[]; fold_summary=[]
    for fold_id,(tr,te) in enumerate(split_iter, start=1):
        m=clone(base_model)
        Xtr=X.iloc[tr].copy(); Xte=X.iloc[te].copy(); ytr=y.iloc[tr].copy(); yte=y.iloc[te].copy()
        m.fit(Xtr,ytr)
        base_pred=m.predict(Xte)
        base_rmse=rmse(yte,base_pred)
        fold_label = None
        if validation_name == 'LOFO':
            labs = obj['df'].iloc[te]['Formulation_label'].astype(str).unique().tolist()
            fold_label = '|'.join(labs)
        # sklearn returns score decrease. With neg-RMSE scoring this equals increase in RMSE.
        pr = permutation_importance(
            m, Xte, yte, scoring='neg_root_mean_squared_error', n_repeats=N_PERM,
            random_state=SEED + fold_id, n_jobs=-1
        )
        for j,feature in enumerate(features):
            deltas=pr.importances[j,:].astype(float)
            for rep,delta in enumerate(deltas, start=1):
                detail.append({
                    'Target':target,'Validation':validation_name,'Fold':fold_id,'Heldout_group':fold_label,
                    'Feature':feature,'Permutation_repeat':rep,'Baseline_RMSE':base_rmse,
                    'Delta_RMSE':float(delta),'Model':obj['model_name'],'Feature_set':obj['feature_set'],'N_test':len(te)
                })
            fold_summary.append({
                'Target':target,'Validation':validation_name,'Fold':fold_id,'Heldout_group':fold_label,
                'Feature':feature,'Fold_mean_Delta_RMSE':float(np.mean(deltas)),
                'Fold_std_Delta_RMSE':float(np.std(deltas,ddof=1)) if len(deltas)>1 else np.nan,
                'Baseline_RMSE':base_rmse,'Model':obj['model_name'],'Feature_set':obj['feature_set'],'N_test':len(te)
            })
    det=pd.DataFrame(detail); fs=pd.DataFrame(fold_summary)
    agg=(fs.groupby(['Target','Validation','Model','Feature_set','Feature'],as_index=False)
         .agg(OOF_Permutation_Delta_RMSE_mean=('Fold_mean_Delta_RMSE','mean'),
              OOF_Permutation_Delta_RMSE_std=('Fold_mean_Delta_RMSE','std'),
              OOF_Permutation_Delta_RMSE_median=('Fold_mean_Delta_RMSE','median'),
              Fold_min=('Fold_mean_Delta_RMSE','min'),Fold_max=('Fold_mean_Delta_RMSE','max'),N_folds=('Fold','nunique')))
    agg['OOF_Permutation_rank']=agg.groupby('Target')['OOF_Permutation_Delta_RMSE_mean'].rank(ascending=False,method='min')
    agg=agg.sort_values(['Target','OOF_Permutation_rank','Feature']).reset_index(drop=True)
    return det,fs,agg

def shap_summary(target,obj):
    # Full-data reference fit is used only for descriptive SHAP attribution; it is not a performance estimate.
    model=clone(obj['model']); X=obj['X']; y=obj['y']; model.fit(X,y)
    masker=shap.maskers.Independent(X)
    explainer=shap.Explainer(model.predict,masker,algorithm='permutation',seed=SEED)
    max_evals=2*X.shape[1]+1
    sv=explainer(X,max_evals=max_evals,silent=True)
    arr=np.asarray(sv.values)
    mean_abs=np.abs(arr).mean(axis=0); mean_val=arr.mean(axis=0); std_abs=np.abs(arr).std(axis=0)
    sdf=pd.DataFrame({
        'Target':target,'Model':obj['model_name'],'Feature_set':obj['feature_set'],'Feature':obj['features'],
        'Mean_abs_SHAP':mean_abs,'Mean_SHAP':mean_val,'Std_abs_SHAP':std_abs,
    })
    sdf['SHAP_rank']=sdf['Mean_abs_SHAP'].rank(ascending=False,method='min')
    sdf=sdf.sort_values(['SHAP_rank','Feature']).reset_index(drop=True)
    return sdf,sv,model

objs=fixed_reference_objects()
rep_details=[]; rep_folds=[]; rep_aggs=[]; lo_details=[]; lo_folds=[]; lo_aggs=[]; shap_tables=[]; shap_values_store={}
for target,obj in objs.items():
    print('OOF repeated permutation:',target,flush=True)
    rkf=RepeatedKFold(n_splits=5,n_repeats=N_XAI_REPEATS,random_state=SEED)
    d,f,a=oof_permutation_for_splits(target,obj,rkf.split(obj['X']), 'Repeated5x5')
    rep_details.append(d); rep_folds.append(f); rep_aggs.append(a)
    print('OOF LOFO permutation:',target,flush=True)
    groups=obj['df']['Formulation_label'].astype(str)
    logo=LeaveOneGroupOut()
    d,f,a=oof_permutation_for_splits(target,obj,logo.split(obj['X'],obj['y'],groups), 'LOFO')
    lo_details.append(d); lo_folds.append(f); lo_aggs.append(a)
    print('SHAP:',target,flush=True)
    s,sv,mod=shap_summary(target,obj)
    shap_tables.append(s); shap_values_store[target]=sv

rep_detail=pd.concat(rep_details,ignore_index=True); rep_fold=pd.concat(rep_folds,ignore_index=True); rep_agg=pd.concat(rep_aggs,ignore_index=True)
lo_detail=pd.concat(lo_details,ignore_index=True); lo_fold=pd.concat(lo_folds,ignore_index=True); lo_agg=pd.concat(lo_aggs,ignore_index=True)
shap_df=pd.concat(shap_tables,ignore_index=True)

# Separate-method comparison: no combined or averaged score.
comp=rep_agg.merge(shap_df[['Target','Model','Feature_set','Feature','Mean_abs_SHAP','Mean_SHAP','Std_abs_SHAP','SHAP_rank']],
                   on=['Target','Model','Feature_set','Feature'],how='inner')
comp['Rank_difference']=np.abs(comp['OOF_Permutation_rank']-comp['SHAP_rank'])
comp=comp.sort_values(['Target','OOF_Permutation_rank','SHAP_rank']).reset_index(drop=True)

# Rank agreement is retained only for SR/Qm/AF, as requested by Reviewer 1; Pr is explicitly excluded.
rank_rows=[]
for target in ['SR','Qm','AF']:
    t=comp[comp.Target==target]
    rho=float(spearmanr(t['OOF_Permutation_rank'],t['SHAP_rank']).correlation) if len(t)>2 else np.nan
    rank_rows.append({'Target':target,'N_features':len(t),'Spearman_rank_agreement':rho,
                      'Interpretation':'Descriptive cross-method rank agreement; not a stability or performance estimate.'})
rank_df=pd.DataFrame(rank_rows)

# Primary hypothesis table: top three by held-out repeated-CV permutation importance; SHAP rank shown separately.
# Pr is marked descriptive only because nested repeated and LOFO R2 are negative.
primary=[]
for target in ['Pr','SR','Qm','AF']:
    t=comp[comp.Target==target].sort_values('OOF_Permutation_rank').head(3).copy()
    df=objs[target]['df']; ycol=objs[target]['ycol']
    for _,r in t.iterrows():
        feat=r['Feature']; tmp=df[[feat,ycol]].dropna().copy()
        sp=float(tmp[feat].corr(tmp[ycol],method='spearman')) if tmp[feat].nunique()>1 else np.nan
        try:
            tmp['bin']=pd.qcut(tmp[feat],3,labels=['Low','Mid','High'],duplicates='drop')
        except Exception:
            tmp['bin']=pd.cut(tmp[feat],3,labels=['Low','Mid','High'])
        means=tmp.groupby('bin',observed=True)[ycol].mean()
        low=float(means.iloc[0]) if len(means)>0 else np.nan
        high=float(means.iloc[-1]) if len(means)>0 else np.nan
        if pd.isna(sp): direction='Not determined'
        elif sp>0.20: direction='Higher feature values associated with higher target values'
        elif sp<-0.20: direction='Higher feature values associated with lower target values'
        else: direction='Weak or non-monotonic univariate association'
        primary.append({
            'Target':target,'Feature':feat,
            'OOF_Permutation_Delta_RMSE_mean':r['OOF_Permutation_Delta_RMSE_mean'],
            'OOF_Permutation_rank':r['OOF_Permutation_rank'],
            'Mean_abs_SHAP':r['Mean_abs_SHAP'],'SHAP_rank':r['SHAP_rank'],
            'Spearman_rho':sp,'Direction':direction,'Mean_outcome_low_bin':low,'Mean_outcome_high_bin':high,
            'High_minus_low':high-low,
            'Interpretation_status':('Descriptive model behavior only; Pr has negative nested R2.' if target=='Pr' else 'Hypothesis-generating model attribution; not causal.')
        })
primary_df=pd.DataFrame(primary)

# Save flat files.
rep_detail.to_csv(OUT/'OOF_Permutation_Repeated_Detail_Comments_18-R2.csv',index=False)
rep_fold.to_csv(OUT/'OOF_Permutation_Repeated_FoldMeans_Comments_18-R2.csv',index=False)
rep_agg.to_csv(OUT/'OOF_Permutation_Repeated_Summary_Comments_18-R2.csv',index=False)
lo_detail.to_csv(OUT/'OOF_Permutation_LOFO_Detail_Comments_18-R2.csv',index=False)
lo_fold.to_csv(OUT/'OOF_Permutation_LOFO_FoldMeans_Comments_18-R2.csv',index=False)
lo_agg.to_csv(OUT/'OOF_Permutation_LOFO_Summary_Comments_18-R2.csv',index=False)
shap_df.to_csv(OUT/'SHAP_Separate_Summary_Comments_17-R2_8-R1.csv',index=False)
comp.to_csv(OUT/'Separate_XAI_Ranks_Comments_17-R2_8-R1.csv',index=False)
rank_df.to_csv(OUT/'Rank_Agreement_Excluding_Pr_Comment_8-R1.csv',index=False)
primary_df.to_csv(OUT/'Table3_Separate_XAI_Comments_17-R2_8-R1.csv',index=False)

# Workbook
xlsx=OUT/'XAI_Analysis_Comments_8-R1_17-18-R2.xlsx'
with pd.ExcelWriter(xlsx,engine='openpyxl') as w:
    pd.DataFrame([{
        'Random_seed':SEED,'Permutation_repeats_per_fold':N_PERM,
        'Repeated_CV':f'5 folds x {N_XAI_REPEATS} repeats','LOFO':'Leave one formulation out',
        'Consensus_score':'Removed','Pr_rank_agreement':'Excluded from summary',
        'Permutation_method':'Feature shuffling performed only in held-out test fold; delta RMSE = permuted RMSE - baseline test RMSE',
        'SHAP_scope':'Descriptive SHAP from full-data refit of the same fixed reference model; not used as a performance estimate.',
        'SHAP_permutation_seed':SEED
    }]).to_excel(w,sheet_name='Metadata',index=False)
    pd.DataFrame([{'Target':t,'Feature_set':REFERENCE[t][0],'Model':REFERENCE[t][1],'N_features':len(objs[t]['features']),'Features':', '.join(objs[t]['features'])} for t in TARGETS]).to_excel(w,sheet_name='ReferenceModels',index=False)
    rep_agg.to_excel(w,sheet_name='OOF_PI_Repeated',index=False)
    rep_fold.to_excel(w,sheet_name='OOF_PI_Rep_Folds',index=False)
    lo_agg.to_excel(w,sheet_name='OOF_PI_LOFO',index=False)
    lo_fold.to_excel(w,sheet_name='OOF_PI_LOFO_Folds',index=False)
    shap_df.to_excel(w,sheet_name='SHAP',index=False)
    comp.to_excel(w,sheet_name='Separate_Ranks',index=False)
    rank_df.to_excel(w,sheet_name='Rank_Agreement_No_Pr',index=False)
    primary_df.to_excel(w,sheet_name='Table3',index=False)

# Figure 1 replacement: separate columns, no averaging.
fig,axes=plt.subplots(4,2,figsize=(13,16))
for i,target in enumerate(['Pr','SR','Qm','AF']):
    p=rep_agg[rep_agg.Target==target].sort_values('OOF_Permutation_rank').head(8).sort_values('OOF_Permutation_Delta_RMSE_mean')
    axes[i,0].barh(p['Feature'],p['OOF_Permutation_Delta_RMSE_mean'])
    axes[i,0].axvline(0,linewidth=0.8)
    axes[i,0].set_xlabel('Held-out permutation importance (delta RMSE)')
    axes[i,0].set_title(f'{target}: out-of-fold permutation')
    s=shap_df[shap_df.Target==target].sort_values('SHAP_rank').head(8).sort_values('Mean_abs_SHAP')
    axes[i,1].barh(s['Feature'],s['Mean_abs_SHAP'])
    axes[i,1].set_xlabel('Mean absolute SHAP value')
    axes[i,1].set_title(f'{target}: SHAP attribution')
fig.tight_layout()
fig.savefig(FIG/'Figure1_Separate_XAI_Comments_17-R2_8-R1.png',dpi=300,bbox_inches='tight')
plt.close(fig)

# Manuscript Figure 2: compare ranks directly, without combining the attribution scales.
# Show every feature appearing in the top six of either method; ties can yield >6 rows.
fig,axes=plt.subplots(2,2,figsize=(14,10.5)); axes=axes.ravel()
for ax,target in zip(axes,['Pr','SR','Qm','AF']):
    t=comp[comp.Target==target].copy()
    t['Best_rank']=t[['OOF_Permutation_rank','SHAP_rank']].min(axis=1)
    t=t[t['Best_rank']<=6].sort_values(['Best_rank','OOF_Permutation_rank','SHAP_rank','Feature']).copy()
    y=np.arange(len(t))
    for yi,(_,r) in enumerate(t.iterrows()):
        ax.plot([r['OOF_Permutation_rank'],r['SHAP_rank']],[yi,yi],linewidth=1)
    ax.scatter(t['OOF_Permutation_rank'],y,label='OOF permutation rank',zorder=3)
    ax.scatter(t['SHAP_rank'],y,marker='s',label='SHAP rank',zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(t['Feature']); ax.invert_yaxis()
    xmax=max(float(t['OOF_Permutation_rank'].max()),float(t['SHAP_rank'].max()))+0.5
    ax.set_xlim(xmax,0.5)
    ax.set_xlabel('Feature rank (1 = highest)')
    ax.set_title(f'{target} (descriptive only)' if target=='Pr' else target)
    ax.grid(axis='x',alpha=0.25)
fig.suptitle('Separate XAI rankings without a composite consensus score',fontsize=16,y=0.995)
handles,labels=axes[-1].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(0.5,-0.005))
fig.tight_layout(rect=[0,0.04,1,0.97])
fig.savefig(FIG/'Figure1_Separate_Ranks_Comments_17-R2_8-R1.png',dpi=300,bbox_inches='tight')
plt.close(fig)

# Replacement Figure S5: held-out repeated-CV permutation importance only.
fig,axes=plt.subplots(2,2,figsize=(12,10)); axes=axes.ravel()
for ax,target in zip(axes,['AF','Pr','Qm','SR']):
    p=rep_agg[rep_agg.Target==target].sort_values('OOF_Permutation_rank').head(12).sort_values('OOF_Permutation_Delta_RMSE_mean')
    ax.barh(p['Feature'],p['OOF_Permutation_Delta_RMSE_mean'])
    ax.axvline(0,linewidth=0.8)
    ax.set_xlabel('Held-out delta RMSE')
    ax.set_title(target)
fig.tight_layout(); fig.savefig(FIG/'FigureS5_OOF_Permutation_Comments_18-R2.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Replacement Figure S6: cleaned SHAP summaries, still kept separate from permutation.
fig,axes=plt.subplots(2,2,figsize=(12,10)); axes=axes.ravel()
for ax,target in zip(axes,['AF','Pr','Qm','SR']):
    s=shap_df[shap_df.Target==target].sort_values('SHAP_rank').head(12).sort_values('Mean_abs_SHAP')
    ax.barh(s['Feature'],s['Mean_abs_SHAP'])
    ax.set_xlabel('Mean absolute SHAP value')
    ax.set_title(target)
fig.tight_layout(); fig.savefig(FIG/'FigureS6_SHAP_Comments_17-R2_8-R1.png',dpi=300,bbox_inches='tight'); plt.close(fig)

metadata={
    'comments':['8-R1','17-R2','18-R2'], 'random_seed':SEED,'n_xai_repeats':N_XAI_REPEATS,'n_permutations_per_heldout_fold':N_PERM,
    'reference_models':{t:{'feature_set':REFERENCE[t][0],'model':REFERENCE[t][1],'features':objs[t]['features']} for t in TARGETS},
    'consensus_score_removed':True,'Pr_excluded_from_rank_agreement':True,'shap_permutation_seed':SEED,
    'notes':['OOF permutation is computed only on held-out test observations.','SHAP is reported separately and is not averaged with permutation importance.','Pr attribution is descriptive only because nested R2 is negative.']
}
(OUT/'XAI_Metadata_Comments_8-R1_17-18-R2.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

print('\nRepeated OOF permutation top features')
for t in ['Pr','SR','Qm','AF']:
    print('\n',t); print(rep_agg[rep_agg.Target==t][['Feature','OOF_Permutation_Delta_RMSE_mean','OOF_Permutation_rank']].head(8).to_string(index=False))
print('\nSHAP top features')
for t in ['Pr','SR','Qm','AF']:
    print('\n',t); print(shap_df[shap_df.Target==t][['Feature','Mean_abs_SHAP','SHAP_rank']].head(8).to_string(index=False))
print('\nRank agreement excluding Pr')
print(rank_df.to_string(index=False))
print('\nPrimary Table 3')
print(primary_df.to_string(index=False))
print('\nSaved',xlsx)
