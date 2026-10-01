#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

import pandas as pd

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'results' / 'master_run'
MASTER_LOG = LOG_DIR / 'master_run.log'
STATUS_JSON = LOG_DIR / 'master_status.json'
VALIDATION_CSV = LOG_DIR / 'numerical_validation.csv'
ENV_JSON = LOG_DIR / 'environment.json'

SOURCE_INPUTS = [
    ROOT/'data'/'processed'/'rheology_table_S1.csv',
    ROOT/'data'/'processed'/'printability_table_S2.csv',
    ROOT/'data'/'processed'/'angle_fidelity_table_S3.csv',
]
HISTORICAL_BOOTSTRAP_INPUT = ROOT/'archive'/'legacy_pre_revision_candidate_ranking'/'tables'/'029_tables_final_top_strict_design_candidates.csv'

DERIVED_PROCESSED = [
    'rheology_standardized.csv','printability_standardized.csv','angle_fidelity_standardized.csv',
    'rheology_ml_features.csv','printability_ml_features.csv','angle_fidelity_ml_features.csv',
    'rheology_formulation_summary.csv','master_printability_dataset.csv','master_angle_fidelity_dataset.csv',
    'modeling_dataset_Pr.csv','modeling_dataset_SR.csv','modeling_dataset_Qm.csv','modeling_dataset_AF.csv',
    'modeling_metadata.json',
]

CORE_STEPS = [
    ('P01 prepare canonical inputs', 'src/prepare_canonical_inputs.py'),
    ('P01b descriptive figures and manuscript heatmaps', 'src/generate_descriptive_figures.py'),
    ('P02 simple baselines', 'src/revision_comment_3_R2_baselines.py'),
    ('P03 dimensionless stress ratio', 'src/revision_comment_10_R1_dimensionless_stress_ratio.py'),
    ('P04 nested validation', 'src/revision_comments_4_R1_16_R2_nested_cv.py'),
    ('P05 R2 aggregation', 'src/revision_comments_4_R1_16_R2_r2_aggregation.py'),
    ('P06 baseline vs nested', 'src/revision_comment_3_R2_baseline_contrast.py'),
    ('P07 LOFO uncertainty', 'src/revision_comments_12_R1_9_R2_lofo_uncertainty.py'),
    ('P08 XAI', 'src/revision_comments_8_R1_17_18_R2_xai.py'),
    ('P10 representation sensitivity', 'src/revision_comments_6_7_R1_8_R2_ablation_selected.py'),
    ('P11 Table2-S19 reconciliation', 'src/revision_comment_6_R1_reconciliation.py'),
    ('P12 reconciliation audit', 'src/revision_comment_8_R2_reconciliation_audit.py'),
    ('P13 candidate set', 'src/revision_comments_2_R1_6_7_R2_candidate_set.py'),
    ('P14 Pr threshold sensitivity', 'src/revision_comment_7_R2_pr_gate_sensitivity.py'),
    ('P15 retrospective ranking bootstrap', 'src/revision_comments_2_R1_6_R2_bootstrap_ranking.py'),
    ('P16 final scientific assets', 'src/finalize_revision_assets.py'),
]


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def log(msg: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp=time.strftime('%Y-%m-%d %H:%M:%S')
    line=f'[{stamp}] {msg}'
    print(line, flush=True)
    with MASTER_LOG.open('a', encoding='utf-8') as f:
        f.write(line+'\n')


def preflight() -> dict:
    issues=[]
    if Path.cwd().resolve()!=ROOT:
        issues.append(f'Run from repository root: expected {ROOT}, cwd={Path.cwd().resolve()}')
    for p in SOURCE_INPUTS+[HISTORICAL_BOOTSTRAP_INPUT]:
        if not p.exists(): issues.append(f'Missing required input: {p.relative_to(ROOT)}')
    for _,rel in CORE_STEPS:
        p=ROOT/rel
        if not p.exists(): issues.append(f'Missing active script: {rel}')
    for rel in ['requirements.txt','environment.yml']:
        if not (ROOT/rel).exists(): issues.append(f'Missing environment file: {rel}')
    # Portable-path check on active Python scripts and master notebook once present.
    offenders=[]
    for p in list((ROOT/'src').glob('*.py')):
        txt=p.read_text(encoding='utf-8',errors='ignore')
        for token in ['/content/','/Users/','C:\\Users\\']:
            if token in txt: offenders.append(f'{p.relative_to(ROOT)} contains {token}')
    if offenders: issues.extend(offenders)
    # Archive policy: only the strict-candidate historical artifact may be read by active source.
    archive_refs=[]
    for p in (ROOT/'src').glob('*.py'):
        txt=p.read_text(encoding='utf-8',errors='ignore')
        if "'archive'" in txt or '"archive"' in txt:
            archive_refs.append(str(p.relative_to(ROOT)))
    allowed={'src/revision_comments_2_R1_6_R2_bootstrap_ranking.py'}
    unexpected=sorted(set(archive_refs)-allowed)
    if unexpected: issues.append('Unexpected active archive references: '+', '.join(unexpected))
    return {
        'status':'PASS' if not issues else 'FAIL',
        'issues':issues,
        'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in SOURCE_INPUTS if p.exists()},
        'historical_bootstrap_hash':sha256(HISTORICAL_BOOTSTRAP_INPUT) if HISTORICAL_BOOTSTRAP_INPUT.exists() else None,
    }


def clean_generated():
    log('Cleaning generated outputs while preserving the three curated source tables and archive input.')
    for name in DERIVED_PROCESSED:
        (ROOT/'data'/'processed'/name).unlink(missing_ok=True)
    for p in [ROOT/'results', ROOT/'supplementary'/'final_revision']:
        if p.exists(): shutil.rmtree(p)
    for pattern in ['__pycache__','.ipynb_checkpoints']:
        for p in ROOT.rglob(pattern):
            if p.is_dir(): shutil.rmtree(p,ignore_errors=True)
    LOG_DIR.mkdir(parents=True,exist_ok=True)


def run_script(label: str, rel: str) -> dict:
    path=ROOT/rel
    t0=time.time()
    log(f'START {label}: {rel}')
    proc=subprocess.run([sys.executable,str(path)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    dt=time.time()-t0
    step_log=LOG_DIR/(rel.replace('/','__')+'.log')
    step_log.write_text(proc.stdout or '',encoding='utf-8')
    if proc.returncode!=0:
        tail='\n'.join((proc.stdout or '').splitlines()[-40:])
        log(f'FAIL {label} after {dt:.1f}s, returncode={proc.returncode}')
        raise RuntimeError(f'{label} failed. Tail:\n{tail}')
    log(f'PASS {label} in {dt:.1f}s')
    return {'label':label,'script':rel,'status':'PASS','seconds':dt,'log':str(step_log.relative_to(ROOT))}


def make_s6_s8(dest: Path):
    rep=pd.read_csv(ROOT/'results'/'revision_comments_4_R1_16_R2'/'Reportable_Nested_Performance_Comments_4-R1_16-R2.csv')
    s6=rep[['Target','Nested repeated selection mode','Repeated selection frequency','Nested repeated R2 mean','Nested repeated R2 SD','Nested repeated MAE mean','Nested repeated RMSE mean','Nested repeated Spearman mean']].copy()
    s6.columns=['Target','Modal selected model / feature set','Selection frequency','R2 mean','R2 SD','MAE mean','RMSE mean','Spearman mean']
    s7=rep[['Target','Nested LOFO selection mode','LOFO selection frequency','Nested LOFO R2','Nested LOFO MAE','Nested LOFO RMSE','Nested LOFO Spearman']].copy()
    s7.columns=['Target','Modal selected model / feature set','Selection frequency','Pooled R2','MAE','RMSE','Spearman']
    s8=rep[['Target','Nested repeated selection mode','Repeated selection frequency','Nested LOFO selection mode','LOFO selection frequency']].copy()
    s8.columns=['Target','Nested repeated modal selection','Repeated frequency','Nested LOFO modal selection','LOFO frequency']
    with pd.ExcelWriter(dest) as w:
        s6.to_excel(w,sheet_name='Table S6',index=False)
        s7.to_excel(w,sheet_name='Table S7',index=False)
        s8.to_excel(w,sheet_name='Table S8',index=False)


def make_s20(dest: Path):
    src=pd.read_csv(ROOT/'results'/'revision_comment_3_R2'/'baseline_summary_Comment_3-R2.csv')
    out=src[['Target','Baseline','Validation','R2','MAE','RMSE','Spearman']].copy()
    out.columns=['Target','Baseline model','Validation','R2','MAE','RMSE','Spearman rho']
    cap=pd.DataFrame({'Proposed supplementary caption':['Simple baseline-model performance: AF angle-only linear/categorical mean and Qm P/TP linear/power-law models under repeated 5-fold CV and pooled LOFO.']})
    with pd.ExcelWriter(dest) as w:
        out.to_excel(w,sheet_name='Table S20',index=False)
        cap.to_excel(w,sheet_name='Caption',index=False)


def consolidate_final_revision():
    tab=ROOT/'supplementary'/'final_revision'/'tables'
    fig=ROOT/'supplementary'/'final_revision'/'figures'
    tab.mkdir(parents=True,exist_ok=True); fig.mkdir(parents=True,exist_ok=True)
    make_s6_s8(tab/'Table_S6-S8_Nested_Performance.xlsx')
    make_s20(tab/'Table_S20_Baselines.xlsx')
    copies={
      ROOT/'results'/'revision_comments_6_7_R1_8_R2'/'Ablation_Analysis_Comments_6-7-R1_8-R2.xlsx': tab/'Table_S19_Ablation.xlsx',
      ROOT/'results'/'revision_comments_4_R1_16_R2'/'Supplementary_Table_S21_Comments_4-R1_16-R2.xlsx': tab/'Table_S21_Nested_CV.xlsx',
      ROOT/'results'/'revision_comments_8_R1_17_18_R2'/'XAI_Analysis_Comments_8-R1_17-18-R2.xlsx': tab/'Table_S22_XAI.xlsx',
      ROOT/'results'/'revision_comments_12_R1_9_R2'/'LOFO_Analysis_Comments_12-R1_9-R2.xlsx': tab/'Table_S23_LOFO_Diagnostics.xlsx',
      ROOT/'results'/'revision_comments_2_R1_6_R2_bootstrap_ranking'/'Bootstrap_Candidate_Ranking_Summary.csv': tab/'Table_S25_Bootstrap_Ranking_Stability.csv',
      ROOT/'results'/'revision_comments_4_R1_16_R2'/'R2_Aggregation_Summary_Comments_4-R1_16-R2.csv': tab/'Table_S26_R2_Aggregation_Summary_Comments_4-R1_16-R2.csv',
      ROOT/'results'/'revision_comment_6_R1_reconciliation'/'Table_S27_Table2_vs_TableS19_Reconciliation.csv': tab/'Table_S27_Table2_vs_TableS19_Reconciliation.csv',
      ROOT/'results'/'revision_comment_10_R1_dimensionless_stress_ratio'/'Table_S28_Single_Descriptor_Summary.csv': tab/'Table_S28_Dimensionless_Stress_Ratio_Sensitivity.csv',
      ROOT/'results'/'revision_comments_12_R1_9_R2'/'All_Targets_LOFO_Formulation_Block_Bootstrap_Comments_12-R1_9-R2.csv': tab/'Table_S29_LOFO_Formulation_Block_Uncertainty.csv',
      ROOT/'results'/'revision_comment_3_R2_baseline_contrast'/'Baseline_vs_Nested_Apples_to_Apples_Comment_3-R2.csv': tab/'Table_S30_Baseline_vs_Nested_Apples_to_Apples_Comment_3-R2.csv',
      ROOT/'results'/'revision_comment_7_R2_pr_gate_sensitivity'/'Table_S31_Relaxed_Pr_Only_Conditions_Comment_7_R2.csv': tab/'Table_S31_Relaxed_Pr_Only_Conditions_Comment_7_R2.csv',
    }
    # Table S24 only if optional provenance audit was run in this execution.
    prov=ROOT/'results'/'revision_comment_2_R2'/'Source_vs_Repository_Audit_Summary_Comment_2-R2.csv'
    if prov.exists(): copies[prov]=tab/'Table_S24_Provenance.csv'
    # Figures generated in this run.
    asset=ROOT/'results'/'final_revision_assets'/'figures'
    xfig=ROOT/'results'/'revision_comments_8_R1_17_18_R2'/'figures'
    cfig=ROOT/'results'/'revision_comments_2_R1_6_7_R2'
    desc=ROOT/'results'/'descriptive_figures'
    figcopies={
      asset/'Figure1_Updated_Workflow.png':fig/'Figure_1_Updated_Workflow.png',
      desc/'FigureS1_Target_Output_Distributions.png':fig/'Figure_S1_Target_Output_Distributions.png',
      desc/'FigureS2_Exploratory_Feature_Output_Scatter.png':fig/'Figure_S2_Exploratory_Feature_Output_Scatter.png',
      asset/'FigureS3_Clean_Correlations.png':fig/'Figure_S3_Clean_Correlations.png',
      asset/'FigureS4_Reference_FullFit.png':fig/'Figure_S4_Reference_FullFit.png',
      xfig/'FigureS5_OOF_Permutation_Comments_18-R2.png':fig/'Figure_S5_OOF_Permutation.png',
      xfig/'FigureS6_SHAP_Comments_17-R2_8-R1.png':fig/'Figure_S6_SHAP.png',
      asset/'FigureS7_Clean_SHAP_Beeswarm.png':fig/'Figure_S7_Clean_SHAP_Beeswarm.png',
      cfig/'Figure3_Unordered_Candidate_Set_Comments_2-R1_6-7-R2.png':fig/'Figure_S8_Measured_Domain_Candidate_Set.png',
      asset/'FigureS9_Final_Ablation.png':fig/'Figure_S9_Final_Ablation.png',
    }
    for src,dst in {**copies,**figcopies}.items():
        if not src.exists(): raise FileNotFoundError(f'Consolidation source missing: {src.relative_to(ROOT)}')
        shutil.copy2(src,dst)
    log(f'PASS P18 SI consolidation: {len(copies)} tables + {len(figcopies)} figures')


def _value(df, target, col):
    return float(df.loc[df['Target'].eq(target),col].iloc[0])


def numerical_validation() -> list[dict]:
    checks=[]
    def add(name,actual,expected,tol=5e-4):
        ok=abs(float(actual)-float(expected))<=tol
        checks.append({'check':name,'actual':float(actual),'expected':float(expected),'tolerance':tol,'status':'PASS' if ok else 'FAIL'})
    rep=pd.read_csv(ROOT/'results'/'revision_comments_4_R1_16_R2'/'Reportable_Nested_Performance_Comments_4-R1_16-R2.csv')
    agg=pd.read_csv(ROOT/'results'/'revision_comments_4_R1_16_R2'/'R2_Aggregation_Summary_Comments_4-R1_16-R2.csv')
    for t,e in {'Pr':-0.083,'SR':-0.074,'Qm':0.762,'AF':0.712}.items(): add(f'Nested repeated R2 {t}',_value(rep,t,'Nested repeated R2 mean'),e)
    for t,e in {'Pr':0.321,'SR':0.033,'Qm':0.846,'AF':0.742}.items(): add(f'Repeat-pooled R2 {t}',_value(agg,t,'Repeat_pooled_OOF_R2_mean'),e)
    for t,e in {'Pr':-0.348,'SR':0.393,'Qm':-1.458,'AF':0.762}.items(): add(f'Nested LOFO R2 {t}',_value(rep,t,'Nested LOFO R2'),e)
    sr=pd.read_csv(ROOT/'results'/'revision_comments_12_R1_9_R2'/'SR_LOFO_Formulation_Block_Bootstrap_Comments_12-R1_9-R2.csv')
    add('SR bootstrap lower',sr['Bootstrap_R2_2p5'].iloc[0],0.179); add('SR bootstrap upper',sr['Bootstrap_R2_97p5'].iloc[0],0.537)
    audit=pd.read_csv(ROOT/'results'/'revision_comments_2_R1_6_7_R2'/'Candidate_Screening_Audit_Comments_2-R1_6-7-R2.csv').set_index('Item')['Value']
    def add_exact(name,actual,expected): checks.append({'check':name,'actual':actual,'expected':expected,'tolerance':0,'status':'PASS' if str(actual)==str(expected) else 'FAIL'})
    add_exact('Jointly observed',int(audit['Jointly observed formulation-pressure conditions']),34)
    add_exact('Strict candidates',int(audit['Candidate-set conditions (measured Pr 0.95-1.05 AND measured AF 1.00-1.25)']),14)
    add_exact('Relaxed candidates',int(audit['Relaxed sensitivity-set conditions (measured Pr 0.90-1.10 AND measured AF 1.00-1.25)']),21)
    add_exact('Relaxed additional',21-14,7)
    bs=pd.read_csv(ROOT/'results'/'revision_comments_2_R1_6_R2_bootstrap_ranking'/'Bootstrap_Candidate_Ranking_Summary.csv')
    leader=bs.sort_values('Bootstrap_P_top',ascending=False).iloc[0]
    add('Bootstrap leader rank1',leader['Bootstrap_P_top'],0.491,5e-4); add('Bootstrap leader top3',leader['Bootstrap_P_top3'],0.633,5e-4)
    pair=pd.read_csv(ROOT/'results'/'revision_comments_2_R1_6_R2_bootstrap_ranking'/'Bootstrap_Pairwise_Differences_vs_Original_Top.csv')
    separated=int(pair['Statistically_separated_at_95pct'].astype(bool).sum())
    add_exact('Pairwise intervals separated',separated,0); add_exact('Pairwise comparisons',len(pair),16)
    bc=pd.read_csv(ROOT/'results'/'revision_comment_3_R2_baseline_contrast'/'Baseline_vs_Nested_Apples_to_Apples_Comment_3-R2.csv')
    def bval(target,baseline,col): return float(bc[(bc.Target==target)&(bc.Baseline==baseline)&bc.Validation.str.contains('repeat-pooled',case=False,na=False)][col].iloc[0])
    add('AF categorical baseline repeat-pooled R2',bval('AF','AF: angle-only categorical mean','Baseline_R2_mean'),0.793)
    add('Nested AF repeat-pooled R2',bval('AF','AF: angle-only categorical mean','Nested_multivariable_R2_mean'),0.742)
    add('Qm power-law baseline repeat-pooled R2',bval('Qm','Qm: P/TP power law','Baseline_R2_mean'),-0.027)
    add('Nested Qm repeat-pooled R2',bval('Qm','Qm: P/TP power law','Nested_multivariable_R2_mean'),0.846)
    add('AF categorical nested-lower-RMSE fraction',bval('AF','AF: angle-only categorical mean','Fraction_repeats_nested_better_RMSE'),0.0,1e-12)
    add('Qm power-law nested-lower-RMSE fraction',bval('Qm','Qm: P/TP power law','Fraction_repeats_nested_better_RMSE'),1.0,1e-12)
    bpred=pd.read_csv(ROOT/'results'/'revision_comment_3_R2_baseline_contrast'/'Baseline_Repeated_Predictions_Comment_3-R2.csv')
    npred=pd.read_csv(ROOT/'results'/'revision_comments_4_R1_16_R2'/'Repeated_Predictions_Comments_4-R1_16-R2.csv')
    for t in ['AF','Qm']:
        bkeys=set(map(tuple,bpred[bpred.Target==t][['Repeat','Outer_fold','Row_index']].astype(int).to_numpy()))
        nkeys=set(map(tuple,npred[npred.Target==t][['Repeat','Outer_fold','Row_index']].astype(int).to_numpy()))
        add_exact(f'S30 exact repeated partitions {t}',bkeys==nkeys,True)
    rk=pd.read_csv(ROOT/'results'/'revision_comments_8_R1_17_18_R2'/'Rank_Agreement_Excluding_Pr_Comment_8-R1.csv')
    for t,e in {'SR':0.693,'Qm':0.895,'AF':0.381}.items(): add(f'XAI rank agreement {t}',_value(rk,t,'Spearman_rank_agreement'),e)
    add_exact('Pr excluded from XAI rank agreement', 'Pr' not in set(rk.Target), True)
    # Independent reconciliation guard: Table S27 must use the primary Table 2 fold-wise mean R2,
    # not the complementary repeat-pooled OOF aggregation from Table S26.
    s27=pd.read_csv(ROOT/'results'/'revision_comment_6_R1_reconciliation'/'Table_S27_Table2_vs_TableS19_Reconciliation.csv')
    for t in ['Pr','SR','Qm','AF']:
        rrep=s27[(s27.Target==t)&(s27.Validation=='Repeated_KFold')]['Table2_nested_R2'].iloc[0]
        rlofo=s27[(s27.Target==t)&(s27.Validation=='LOFO')]['Table2_nested_R2'].iloc[0]
        add(f'Table S27 repeated convention {t}',rrep,_value(rep,t,'Nested repeated R2 mean'),1e-12)
        add(f'Table S27 LOFO convention {t}',rlofo,_value(rep,t,'Nested LOFO R2'),1e-12)
    s19rec=pd.read_csv(ROOT/'results'/'revision_comments_6_7_R1_8_R2'/'Table2_vs_TableS19_Reconciliation_Comments_6-7-R1_8-R2.csv')
    for t in ['Pr','SR','Qm','AF']:
        rr=s19rec[(s19rec.Target==t)&(s19rec.Validation=='Repeated_KFold')]['Table2_nested_R2'].iloc[0]
        rl=s19rec[(s19rec.Target==t)&(s19rec.Validation=='LOFO')]['Table2_nested_R2'].iloc[0]
        add(f'S19 reconciliation repeated convention {t}',rr,_value(rep,t,'Nested repeated R2 mean'),1e-12)
        add(f'S19 reconciliation LOFO convention {t}',rl,_value(rep,t,'Nested LOFO R2'),1e-12)
    pd.DataFrame(checks).to_csv(VALIDATION_CSV,index=False)
    fails=[c for c in checks if c['status']!='PASS']
    if fails: raise RuntimeError('Numerical validation failed: '+json.dumps(fails,indent=2))
    log(f'PASS P19 numerical validation: {len(checks)} checks')
    return checks


def environment_info():
    info={'python':sys.version,'platform':platform.platform(),'executable':sys.executable,'cwd':str(Path.cwd()),'root':str(ROOT)}
    try:
        import numpy, pandas, sklearn, scipy, xgboost, shap
        info['packages']={'numpy':numpy.__version__,'pandas':pandas.__version__,'scikit-learn':sklearn.__version__,'scipy':scipy.__version__,'xgboost':xgboost.__version__,'shap':shap.__version__}
    except Exception as e: info['package_probe_error']=repr(e)
    ENV_JSON.write_text(json.dumps(info,indent=2),encoding='utf-8')
    return info


def main():
    ap=argparse.ArgumentParser(description='Canonical clean-room master runner')
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--dry-run',action='store_true',help='Show the canonical order without executing scientific steps.')
    ap.add_argument('--no-clean',action='store_true',help='Do not delete generated outputs before execution. Not permitted for final clean-room certification.')
    ap.add_argument('--provenance-pdf',type=Path,default=None,help='Optional local Perin Supporting Information PDF for Table S24 audit.')
    args=ap.parse_args()
    if Path.cwd().resolve()!=ROOT:
        print(f'ERROR: run from repository root: {ROOT}',file=sys.stderr); return 2
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    if MASTER_LOG.exists(): MASTER_LOG.unlink()
    status={'start_time':time.time(),'steps':[],'overall':'RUNNING'}
    try:
        pf=preflight(); status['preflight']=pf
        if pf['status']!='PASS': raise RuntimeError('Preflight failed: '+json.dumps(pf['issues'],indent=2))
        log('PASS P00 preflight')
        if args.preflight_only:
            status['overall']='PREFLIGHT_PASS'; STATUS_JSON.write_text(json.dumps(status,indent=2),encoding='utf-8'); return 0
        if args.dry_run:
            for label,rel in CORE_STEPS: log(f'DRY RUN {label}: {rel}')
            log('DRY RUN P18 SI consolidation')
            log('DRY RUN P09 XAI audits after consolidation')
            log('DRY RUN P19 numerical validation')
            status['overall']='DRY_RUN_PASS'; STATUS_JSON.write_text(json.dumps(status,indent=2),encoding='utf-8'); return 0
        if not args.no_clean: clean_generated()
        else: log('WARNING --no-clean used; this run cannot certify a final clean-room state.')
        environment_info()
        for label,rel in CORE_STEPS:
            status['steps'].append(run_script(label,rel))
        if args.provenance_pdf is not None:
            p=args.provenance_pdf.resolve()
            if not p.exists(): raise FileNotFoundError(p)
            t0=time.time(); proc=subprocess.run([sys.executable,str(ROOT/'src'/'audit_comment_2_R2.py'),str(p)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (LOG_DIR/'optional_provenance.log').write_text(proc.stdout or '',encoding='utf-8')
            if proc.returncode!=0: raise RuntimeError('Optional provenance audit was requested but failed.')
            status['steps'].append({'label':'P17 optional provenance','status':'PASS','seconds':time.time()-t0})
        else:
            status['steps'].append({'label':'P17 optional provenance','status':'SKIPPED-OPTIONAL'})
        consolidate_final_revision()
        # Audits that explicitly inspect consolidated final-review artifacts must run after P18.
        status['steps'].append(run_script('P09a XAI final-artifact audit','src/revision_comment_17_R2_xai_audit.py'))
        status['steps'].append(run_script('P09b OOF permutation final-artifact audit','src/revision_comment_18_R2_oof_permutation_audit.py'))
        checks=numerical_validation(); status['numerical_validation']=checks
        status['environment']=environment_info()
        status['overall']='PASS'; status['end_time']=time.time(); status['runtime_seconds']=status['end_time']-status['start_time']
        STATUS_JSON.write_text(json.dumps(status,indent=2),encoding='utf-8')
        log(f'FULL RUN PASS in {status["runtime_seconds"]:.1f}s')
        return 0
    except Exception as e:
        status['overall']='FAIL'; status['error']=repr(e); status['traceback']=traceback.format_exc(); status['end_time']=time.time(); status['runtime_seconds']=status['end_time']-status['start_time']
        STATUS_JSON.write_text(json.dumps(status,indent=2),encoding='utf-8')
        log('MASTER FAIL: '+repr(e))
        return 1

if __name__=='__main__':
    raise SystemExit(main())
