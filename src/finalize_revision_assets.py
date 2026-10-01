from pathlib import Path
import shutil, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
import shap
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
OUT = ROOT / 'results' / 'final_revision_assets'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
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
    'angle_rheology_enhanced': CORE + ENG + ANGLE + RHEO,
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

def models():
    return {
        'LinearRegression': Pipeline([('scaler',StandardScaler()),('model',LinearRegression())]),
        'SVR_RBF': Pipeline([('scaler',StandardScaler()),('model',SVR(kernel='rbf',C=10.0,epsilon=0.05))]),
        'XGBoost': XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=2,subsample=0.9,colsample_bytree=0.9,objective='reg:squarederror',random_state=SEED,n_jobs=1),
    }

def obj(target):
    fn,ycol = TARGETS[target]
    df = pd.read_csv(DATA/fn).dropna(subset=[ycol]).reset_index(drop=True)
    fs,mn = REFERENCE[target]
    feats = [f for f in FEATURE_SETS[fs] if f in df.columns]
    return df, ycol, feats, models()[mn]

# Figure S3: duplicate-reduced feature-target correlations.
fig, axes = plt.subplots(2,2,figsize=(12,9)); axes=axes.ravel()
for ax,target in zip(axes,['AF','Pr','Qm','SR']):
    fn,ycol=TARGETS[target]; df=pd.read_csv(DATA/fn).dropna(subset=[ycol]).reset_index(drop=True)
    maxset = 'angle_rheology_enhanced' if target=='AF' else 'rheology_enhanced'
    feats=[f for f in FEATURE_SETS[maxset] if f in df.columns]
    vals=[]
    for f in feats:
        if df[f].nunique() > 1:
            rho=df[f].corr(df[ycol],method='spearman')
            if pd.notna(rho): vals.append((f,abs(float(rho))))
    s=pd.DataFrame(vals,columns=['Feature','AbsSpearman']).sort_values('AbsSpearman',ascending=False).head(12).sort_values('AbsSpearman')
    ax.barh(s.Feature,s.AbsSpearman)
    ax.set_xlabel('Absolute Spearman correlation'); ax.set_title(f'Top feature-target correlations for {target}')
fig.tight_layout(); fig.savefig(FIG/'FigureS3_Clean_Correlations.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Figure S4: full-data reference-model fit diagnostics.
fig,axes=plt.subplots(2,2,figsize=(10,9)); axes=axes.ravel()
for ax,target in zip(axes,['AF','Pr','Qm','SR']):
    df,ycol,feats,m=obj(target); X=df[feats]; y=df[ycol]; m.fit(X,y); p=m.predict(X)
    ax.scatter(y,p,s=20); lo=min(y.min(),p.min()); hi=max(y.max(),p.max()); ax.plot([lo,hi],[lo,hi])
    ax.set_xlabel(f'Observed {target}'); ax.set_ylabel(f'Predicted {target}'); ax.set_title(f'Full-data reference fit: {target}')
fig.tight_layout(); fig.savefig(FIG/'FigureS4_Reference_FullFit.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Figure S7: cleaned SHAP beeswarms for the same reference models.
bees=[]
for target in ['AF','Pr','Qm','SR']:
    df,ycol,feats,m=obj(target); X=df[feats]; y=df[ycol]; m.fit(X,y)
    masker=shap.maskers.Independent(X)
    explainer=shap.Explainer(m.predict,masker,algorithm='permutation',seed=SEED)
    sv=explainer(X,max_evals=2*X.shape[1]+1,silent=True)
    plt.figure(figsize=(7.0,5.2))
    shap.plots.beeswarm(sv,max_display=min(12,X.shape[1]),show=False)
    plt.title(f'SHAP beeswarm for {target}')
    path=FIG/f'_tmp_beeswarm_{target}.png'; plt.savefig(path,dpi=220,bbox_inches='tight'); plt.close()
    bees.append((target,path))
# combine 2x2 and label panels
thumbs=[]
for idx,(target,path) in enumerate(bees):
    im=Image.open(path).convert('RGB'); im.thumbnail((1000,760))
    canvas=Image.new('RGB',(1030,800),'white'); canvas.paste(im,((1030-im.width)//2,30))
    d=ImageDraw.Draw(canvas); d.text((12,8),chr(ord('A')+idx),fill='black')
    thumbs.append(canvas)
sheet=Image.new('RGB',(2060,1600),'white')
for i,im in enumerate(thumbs): sheet.paste(im,((i%2)*1030,(i//2)*800))
sheet.save(FIG/'FigureS7_Clean_SHAP_Beeswarm.png')
for _,p in bees: p.unlink(missing_ok=True)

# Figure S8: measured-domain unordered candidate set (copy the reviewer-responsive result).
src = ROOT/'results'/'revision_comments_2_R1_6_7_R2'/'Figure3_Unordered_Candidate_Set_Comments_2-R1_6-7-R2.png'
shutil.copy2(src, FIG/'FigureS8_Measured_Domain_Candidate_Set.png')

# Figure S9: final duplicate-reduced ablation summary.
ablation = pd.read_excel(ROOT/'results'/'revision_comments_6_7_R1_8_R2'/'Ablation_Analysis_Comments_6-7-R1_8-R2.xlsx', sheet_name='Table_S19')
ablation['Label']=ablation['Target']+' | '+ablation['Validation'].str.replace('_',' ',regex=False)
x=np.arange(len(ablation)); w=0.38
fig,ax=plt.subplots(figsize=(12,5.5))
ax.bar(x-w/2,ablation['Delta_RMSE_core_minus_engineered'],w,label='Delta RMSE')
ax.bar(x+w/2,ablation['Delta_Spearman_engineered_minus_core'],w,label='Delta Spearman')
ax.axhline(0,linewidth=0.8); ax.set_xticks(x); ax.set_xticklabels(ablation.Label,rotation=45,ha='right'); ax.set_ylabel('Positive = engineered-side improvement'); ax.set_title('Core-versus-engineered representation sensitivity'); ax.legend()
fig.tight_layout(); fig.savefig(FIG/'FigureS9_Final_Ablation.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Main Figure 1: updated workflow schematic, intentionally free of the retired causal/ranking claims.
fig,ax=plt.subplots(figsize=(10,9.88)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
cols=[(0.035,0.11,0.28,0.80),(0.36,0.11,0.28,0.80),(0.685,0.11,0.28,0.80)]
headers=['1. Source data','2. Validation +\ninterpretation','3. Bounded outputs']
for (x,y,wc,h),hdr in zip(cols,headers):
    ax.add_patch(FancyBboxPatch((x,y),wc,h,boxstyle='round,pad=0.012,rounding_size=0.018',fill=False,linewidth=1.8))
    ax.text(x+wc/2,y+h-0.045,hdr,ha='center',va='center',fontsize=11.2,fontweight='bold')
def box(ax,x,y,w,h,title,sub,ts=10.3,ss=7.3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.007',alpha=0.08))
    ax.text(x+0.018,y+h-0.026,title,fontsize=ts,fontweight='bold',va='center')
    ax.text(x+0.018,y+0.020,sub,fontsize=ss,va='bottom',wrap=True)
# col 1
x,y,wc,h=cols[0]
items=[('Acellular Alg-HA inks','Source-study nomenclature retained'),('Inputs','Alg, HA, pressure, rheology, angle'),('Outputs','Pr, SR, Qm, AF'),('Data structure','40 Pr | 43 SR | 40 Qm | 96 AF')]
y0=y+h-0.16
for title,sub in items:
    box(ax,x+0.025,y0-0.09,wc-0.05,0.09,title,sub,10.4,7.9); y0-=0.145
# col2
x,y,wc,h=cols[1]
items=[('Duplicate-reduced features','Exact aliases and deterministic\nduplicates removed'),('Nested repeated CV','Outer 5-fold x 20; inner 5-fold\nmodel/feature-set selection'),('Nested LOFO','Outer formulation holdout;\ninner LOFO selection'),('Baseline checks','Angle-only AF and P/TP-only Qm'),('XAI','Held-out permutation and SHAP\nreported separately')]
y0=y+h-0.145
for title,sub in items:
    box(ax,x+0.022,y0-0.082,wc-0.044,0.082,title,sub,9.9,7.25); y0-=0.12
# col3
x,y,wc,h=cols[2]
items=[('Qm','Strong within-domain prediction;\npoor formulation transfer'),('AF','Predictability largely explained by\nthe sampled angle levels'),('Pr','Negative nested R2; attribution\nreported descriptively'),('SR','Formulation-level transfer\nis heterogeneous'),('Candidate set','14 measured-domain conditions;\nno within-set ranking')]
y0=y+h-0.145
for title,sub in items:
    box(ax,x+0.022,y0-0.082,wc-0.044,0.082,title,sub,9.9,7.15); y0-=0.12
for a,b in [(cols[0],cols[1]),(cols[1],cols[2])]:
    ax.add_patch(FancyArrowPatch((a[0]+a[2]+0.006,0.51),(b[0]-0.006,0.51),arrowstyle='-|>',mutation_scale=18,linewidth=1.5))
ax.text(0.5,0.050,'Hypothesis-generating interpretation only; no causal mechanism or model-derived optimum is claimed.',ha='center',fontsize=9.2,fontstyle='italic')
fig.tight_layout(); fig.savefig(FIG/'Figure1_Updated_Workflow.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Non-generative, data-derived TOC graphic from the measured candidate set.
cand=pd.read_csv(ROOT/'results'/'revision_comments_2_R1_6_7_R2'/'All_Observed_Overlap_Comments_2-R1_6-7-R2.csv')
form_col='Formulation_label'; press_col='Pressure_kPa'; flag='Candidate_set'
forms=list(dict.fromkeys(cand[form_col].astype(str).tolist()))
ypos={f:i for i,f in enumerate(forms)}
fig,ax=plt.subplots(figsize=(8.25/2.54,4.45/2.54),dpi=300)
non=cand[~cand[flag].astype(bool)]; yes=cand[cand[flag].astype(bool)]
ax.scatter(non[press_col], [ypos[str(v)] for v in non[form_col]], s=18, marker='o', color='0.65', label='Jointly observed')
ax.scatter(yes[press_col], [ypos[str(v)] for v in yes[form_col]], s=24, marker='s', color='#f28e2b', label='Candidate set')
ax.set_yticks(list(ypos.values())); ax.set_yticklabels(forms,fontsize=5.5); ax.tick_params(axis='x',labelsize=6); ax.set_xlabel('Pressure (kPa)',fontsize=7); ax.set_title('Measured-domain candidate set',fontsize=8); ax.grid(axis='x',alpha=0.12); ax.legend(fontsize=5.2,loc='upper left',frameon=False)
fig.tight_layout(pad=0.35); fig.savefig(FIG/'TOC_Graphic_Data_Derived.png',dpi=600,bbox_inches='tight'); plt.close(fig)

print('Final revision assets written to',FIG)
