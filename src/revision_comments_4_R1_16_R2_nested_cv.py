from pathlib import Path
import json, time, math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'results' / 'revision_comments_4_R1_16_R2'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_OUTER_REPEATS = 20
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
N_JOBS = 5

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
    'Pr': ('modeling_dataset_Pr.csv', 'Pr_mean', ['core','engineered','rheology_enhanced']),
    'SR': ('modeling_dataset_SR.csv', 'SR_mean', ['core','engineered','rheology_enhanced']),
    'Qm': ('modeling_dataset_Qm.csv', 'Qm_mean_mg_s', ['core','engineered','rheology_enhanced']),
    'AF': ('modeling_dataset_AF.csv', 'AF_mean', ['core','angle_engineered','angle_rheology_enhanced']),
}

def get_models():
    return {
        'LinearRegression': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
        'SVR_RBF': Pipeline([('scaler', StandardScaler()), ('model', SVR(kernel='rbf', C=10.0, epsilon=0.05))]),
        'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=SEED, n_jobs=1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=SEED),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.9, colsample_bytree=0.9, objective='reg:squarederror', random_state=SEED, n_jobs=1),
    }

def safe_spearman(y, p):
    y=np.asarray(y); p=np.asarray(p)
    if len(np.unique(y)) <= 1 or len(np.unique(p)) <= 1:
        return np.nan
    return float(spearmanr(y,p).correlation)

def metrics(y,p):
    return {
        'R2': float(r2_score(y,p)),
        'MAE': float(mean_absolute_error(y,p)),
        'RMSE': float(mean_squared_error(y,p)**0.5),
        'Spearman': safe_spearman(y,p),
    }

def candidate_defs(target, df):
    models=get_models()
    out=[]
    for fs in TARGETS[target][2]:
        feats=[c for c in FEATURE_SETS[fs] if c in df.columns]
        if df[feats].isna().sum().sum() > 0:
            continue
        for mn,m in models.items():
            out.append((fs,mn,feats,m))
    return out

def evaluate_candidate_kfold(candidate, Xdf, y, split_indices):
    fs,mn,feats,model = candidate
    pred=np.full(len(y), np.nan)
    for tr,va in split_indices:
        m=clone(model)
        m.fit(Xdf.iloc[tr][feats], y.iloc[tr])
        pred[va]=m.predict(Xdf.iloc[va][feats])
    met=metrics(y, pred)
    return fs,mn,met

def evaluate_candidate_logo(candidate, Xdf, y, groups):
    fs,mn,feats,model = candidate
    pred=np.full(len(y), np.nan)
    logo=LeaveOneGroupOut()
    for tr,va in logo.split(Xdf,y,groups):
        m=clone(model)
        m.fit(Xdf.iloc[tr][feats], y.iloc[tr])
        pred[va]=m.predict(Xdf.iloc[va][feats])
    met=metrics(y,pred)
    return fs,mn,met

def choose_candidate_kfold(target, train_df, target_col, inner_seed):
    y=train_df[target_col].reset_index(drop=True)
    X=train_df.reset_index(drop=True)
    kf=KFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=inner_seed)
    splits=list(kf.split(X))
    cands=candidate_defs(target,X)
    res=Parallel(n_jobs=N_JOBS, prefer='processes')(
        delayed(evaluate_candidate_kfold)(c,X,y,splits) for c in cands
    )
    rows=[]
    for fs,mn,met in res:
        rows.append({'Feature_set':fs,'Model':mn,**met})
    d=pd.DataFrame(rows).sort_values(['RMSE','MAE','Feature_set','Model']).reset_index(drop=True)
    best=d.iloc[0]
    return str(best.Feature_set), str(best.Model), d

def choose_candidate_logo(target, train_df, target_col):
    y=train_df[target_col].reset_index(drop=True)
    X=train_df.reset_index(drop=True)
    groups=X['Formulation_label'].astype(str).reset_index(drop=True)
    cands=candidate_defs(target,X)
    res=Parallel(n_jobs=N_JOBS, prefer='processes')(
        delayed(evaluate_candidate_logo)(c,X,y,groups) for c in cands
    )
    rows=[]
    for fs,mn,met in res:
        rows.append({'Feature_set':fs,'Model':mn,**met})
    d=pd.DataFrame(rows).sort_values(['RMSE','MAE','Feature_set','Model']).reset_index(drop=True)
    best=d.iloc[0]
    return str(best.Feature_set), str(best.Model), d

def fit_predict(target, train_df, test_df, target_col, fs, mn):
    model=get_models()[mn]
    feats=[c for c in FEATURE_SETS[fs] if c in train_df.columns]
    m=clone(model)
    m.fit(train_df[feats],train_df[target_col])
    return m.predict(test_df[feats])

start=time.time()
rep_summ=[]; rep_fold=[]; rep_pred=[]; rep_sel=[]; rep_inner=[]
lofo_summ=[]; lofo_group=[]; lofo_pred=[]; lofo_sel=[]; lofo_inner=[]

for target,(fn,target_col,_) in TARGETS.items():
    df=pd.read_csv(DATA/fn).dropna(subset=[target_col]).reset_index(drop=True)
    print(f'=== TARGET {target}: N={len(df)} ===', flush=True)
    # Nested repeated 5-fold: inner CV selects model x feature set; outer folds estimate performance.
    for rep in range(N_OUTER_REPEATS):
        outer=KFold(n_splits=N_OUTER_SPLITS,shuffle=True,random_state=SEED+rep)
        repeat_pred=np.full(len(df),np.nan)
        for fold,(tr,te) in enumerate(outer.split(df),start=1):
            train=df.iloc[tr].reset_index(drop=True); test=df.iloc[te].reset_index(drop=True)
            fs,mn,inner_table=choose_candidate_kfold(target,train,target_col,inner_seed=SEED+1000*rep+fold)
            p=fit_predict(target,train,test,target_col,fs,mn)
            repeat_pred[te]=p
            fm=metrics(test[target_col].values,p)
            rep_fold.append({'Target':target,'Repeat':rep+1,'Outer_fold':fold,'N_train':len(train),'N_test':len(test),'Selected_feature_set':fs,'Selected_model':mn,**fm})
            rep_sel.append({'Target':target,'Repeat':rep+1,'Outer_fold':fold,'Selected_feature_set':fs,'Selected_model':mn})
            inner_table=inner_table.copy(); inner_table['Target']=target; inner_table['Repeat']=rep+1; inner_table['Outer_fold']=fold
            rep_inner.append(inner_table)
            for local_i,orig_i in enumerate(te):
                rep_pred.append({'Target':target,'Repeat':rep+1,'Outer_fold':fold,'Row_index':int(orig_i),'Observed':float(df.loc[orig_i,target_col]),'Predicted':float(p[local_i]),'Selected_feature_set':fs,'Selected_model':mn})
        sm=metrics(df[target_col].values, repeat_pred)
        rep_summ.append({'Target':target,'Repeat':rep+1,**sm})
        print(f'{target} repeated nested {rep+1}/{N_OUTER_REPEATS}: R2={sm["R2"]:.4f}, RMSE={sm["RMSE"]:.4f}',flush=True)

    # Nested LOFO: outer held-out formulation; inner LOFO on remaining formulations selects model x feature set.
    groups=df['Formulation_label'].astype(str)
    for gi,held in enumerate(sorted(groups.unique()),start=1):
        te=(groups==held).to_numpy(); tr=~te
        train=df.loc[tr].reset_index(drop=True); test=df.loc[te].reset_index(drop=True)
        fs,mn,inner_table=choose_candidate_logo(target,train,target_col)
        p=fit_predict(target,train,test,target_col,fs,mn)
        gm=metrics(test[target_col].values,p)
        lofo_group.append({'Target':target,'Held_out_formulation':held,'N_train':len(train),'N_test':len(test),'Selected_feature_set':fs,'Selected_model':mn,**gm})
        lofo_sel.append({'Target':target,'Held_out_formulation':held,'Selected_feature_set':fs,'Selected_model':mn})
        inner_table=inner_table.copy(); inner_table['Target']=target; inner_table['Held_out_formulation']=held
        lofo_inner.append(inner_table)
        for j in range(len(test)):
            lofo_pred.append({'Target':target,'Held_out_formulation':held,'Observed':float(test.iloc[j][target_col]),'Predicted':float(p[j]),'Selected_feature_set':fs,'Selected_model':mn})
        print(f'{target} nested LOFO {gi}/{groups.nunique()} held={held}: model={mn}, fs={fs}',flush=True)

# frames and summaries
rep_summ_df=pd.DataFrame(rep_summ)
rep_fold_df=pd.DataFrame(rep_fold)
rep_pred_df=pd.DataFrame(rep_pred)
rep_sel_df=pd.DataFrame(rep_sel)
rep_inner_df=pd.concat(rep_inner,ignore_index=True)
lofo_group_df=pd.DataFrame(lofo_group)
lofo_pred_df=pd.DataFrame(lofo_pred)
lofo_sel_df=pd.DataFrame(lofo_sel)
lofo_inner_df=pd.concat(lofo_inner,ignore_index=True)

nested_rep_summary=[]
for target,g in rep_summ_df.groupby('Target'):
    row={'Target':target,'N_outer_repeats':len(g),'Outer_splits':N_OUTER_SPLITS,'Inner_splits':N_INNER_SPLITS}
    for met in ['R2','MAE','RMSE','Spearman']:
        row[f'{met}_mean']=g[met].mean(); row[f'{met}_std']=g[met].std(ddof=1); row[f'{met}_median']=g[met].median(); row[f'{met}_min']=g[met].min(); row[f'{met}_max']=g[met].max()
    nested_rep_summary.append(row)
nested_rep_summary_df=pd.DataFrame(nested_rep_summary)

nested_lofo_summary=[]
for target,g in lofo_pred_df.groupby('Target'):
    m=metrics(g['Observed'],g['Predicted'])
    gg=lofo_group_df[lofo_group_df.Target==target]
    nested_lofo_summary.append({'Target':target,'N_outer_formulations':len(gg),**{f'LOFO_{k}':v for k,v in m.items()},'Group_MAE_mean':gg.MAE.mean(),'Group_MAE_std':gg.MAE.std(ddof=1),'Group_RMSE_mean':gg.RMSE.mean(),'Group_RMSE_std':gg.RMSE.std(ddof=1)})
nested_lofo_summary_df=pd.DataFrame(nested_lofo_summary)

# selection frequencies
rep_freq=(rep_sel_df.groupby(['Target','Selected_feature_set','Selected_model']).size().reset_index(name='Count'))
rep_freq['Frequency']=rep_freq['Count']/rep_freq.groupby('Target')['Count'].transform('sum')
rep_freq=rep_freq.sort_values(['Target','Count'],ascending=[True,False])
lofo_freq=(lofo_sel_df.groupby(['Target','Selected_feature_set','Selected_model']).size().reset_index(name='Count'))
lofo_freq['Frequency']=lofo_freq['Count']/lofo_freq.groupby('Target')['Count'].transform('sum')
lofo_freq=lofo_freq.sort_values(['Target','Count'],ascending=[True,False])

# Build the reportable Table-2/S6-S8 summary entirely from outputs of this run.
# Repeated metrics here use the manuscript convention: arithmetic mean across the 100
# untouched outer-fold metrics. Repeat-pooled OOF R2 is generated separately in P05.
def _mode_row(freq_df, target):
    g=freq_df[freq_df['Target']==target].sort_values(
        ['Count','Selected_feature_set','Selected_model'],
        ascending=[False,True,True],
    )
    return g.iloc[0]

report_rows=[]
for target in TARGETS:
    fg=rep_fold_df[rep_fold_df['Target']==target]
    lr=nested_lofo_summary_df[nested_lofo_summary_df['Target']==target].iloc[0]
    rm=_mode_row(rep_freq,target); lm=_mode_row(lofo_freq,target)
    report_rows.append({
        'Target':target,
        'Nested repeated selection mode':f"{rm['Selected_model']} / {rm['Selected_feature_set']}",
        'Repeated selection frequency':float(rm['Frequency']),
        'Nested repeated R2 mean':float(fg['R2'].mean()),
        'Nested repeated R2 SD':float(fg['R2'].std(ddof=1)),
        'Nested repeated MAE mean':float(fg['MAE'].mean()),
        'Nested repeated RMSE mean':float(fg['RMSE'].mean()),
        'Nested repeated Spearman mean':float(fg['Spearman'].mean()),
        'Nested LOFO selection mode':f"{lm['Selected_model']} / {lm['Selected_feature_set']}",
        'LOFO selection frequency':float(lm['Frequency']),
        'Nested LOFO R2':float(lr['LOFO_R2']),
        'Nested LOFO MAE':float(lr['LOFO_MAE']),
        'Nested LOFO RMSE':float(lr['LOFO_RMSE']),
        'Nested LOFO Spearman':float(lr['LOFO_Spearman']),
    })
reportable_df=pd.DataFrame(report_rows)
reportable_df.to_csv(OUT/'Reportable_Nested_Performance_Comments_4-R1_16-R2.csv',index=False)

# Historical non-nested performance tables are not inputs to the canonical nested analysis.
# Any retrospective comparison with legacy analyses is kept outside the clean-room core.

# save CSV and a consolidated workbook
frames={
 'Nested_Repeated_Summary':nested_rep_summary_df,
 'Nested_LOFO_Summary':nested_lofo_summary_df,
 'Repeated_Selection_Freq':rep_freq,
 'LOFO_Selection_Freq':lofo_freq,
 'Repeated_Repeat_Metrics':rep_summ_df,
 'Repeated_Outer_Fold_Metrics':rep_fold_df,
 'Nested_LOFO_Group_Metrics':lofo_group_df,
 'Repeated_Predictions':rep_pred_df,
 'Nested_LOFO_Predictions':lofo_pred_df,
 'Repeated_Inner_Candidates':rep_inner_df,
 'LOFO_Inner_Candidates':lofo_inner_df,
 'Reportable_Nested_Performance':reportable_df,
}
for name,dfx in frames.items():
    dfx.to_csv(OUT/f'{name}_Comments_4-R1_16-R2.csv',index=False)
with pd.ExcelWriter(OUT/'Nested_CV_Analysis_Comments_4-R1_16-R2.xlsx',engine='openpyxl') as xw:
    for name,dfx in frames.items():
        dfx.to_excel(xw,sheet_name=name[:31],index=False)

# Dedicated SI Table S21 workbook, generated from the same canonical nested run.
with pd.ExcelWriter(OUT/'Supplementary_Table_S21_Comments_4-R1_16-R2.xlsx',engine='openpyxl') as xw:
    reportable_df.to_excel(xw,sheet_name='Table_S21',index=False)
    rep_fold_df.to_excel(xw,sheet_name='Repeated_Outer_Folds',index=False)
    lofo_group_df.to_excel(xw,sheet_name='LOFO_Outer_Groups',index=False)
    rep_freq.to_excel(xw,sheet_name='Repeated_Selection',index=False)
    lofo_freq.to_excel(xw,sheet_name='LOFO_Selection',index=False)

meta={'random_state':SEED,'outer_repeated_cv':{'splits':N_OUTER_SPLITS,'repeats':N_OUTER_REPEATS},'inner_cv':{'splits':N_INNER_SPLITS,'selection_metric':'pooled inner RMSE, MAE tie-break'},'nested_lofo':{'outer':'leave one formulation out','inner':'leave one formulation out among remaining formulations'},'models':list(get_models().keys()),'feature_sets':FEATURE_SETS,'elapsed_seconds':time.time()-start}
(OUT/'Nested_CV_Metadata_Comments_4-R1_16-R2.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

print('\n=== NESTED REPEATED SUMMARY ===')
print(nested_rep_summary_df.to_string(index=False))
print('\n=== NESTED LOFO SUMMARY ===')
print(nested_lofo_summary_df.to_string(index=False))
print('\n=== REPEATED SELECTION FREQUENCIES ===')
print(rep_freq.to_string(index=False))
print('\n=== LOFO SELECTION FREQUENCIES ===')
print(lofo_freq.to_string(index=False))
print(f'Elapsed seconds: {time.time()-start:.1f}')
