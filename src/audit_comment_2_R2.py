#!/usr/bin/env python3
"""Reviewer 2 Comment 2 source-table provenance audit.

Usage:
    python src/audit_comment_2_R2.py /path/to/Perin_supporting_information.pdf

The script parses Tables S1-S3 from a legally obtained local copy of the source
Supporting Information using the extraction logic preserved in the archived
workflow, then compares the parsed values with the curated repository CSVs.
The source PDF is never copied into the repository.
"""
from pathlib import Path
import argparse, re, json
import numpy as np
import pandas as pd
import pdfplumber

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'revision_comment_2_R2'
OUT.mkdir(parents=True,exist_ok=True)

ap=argparse.ArgumentParser()
ap.add_argument('source_pdf', type=Path, help='Legally obtained Perin et al. Supporting Information PDF')
args=ap.parse_args()
pdf_path=args.source_pdf
if not pdf_path.exists():
    raise FileNotFoundError(pdf_path)

pages=[]
with pdfplumber.open(pdf_path) as pdf:
    for i,p in enumerate(pdf.pages,1):
        pages.append({'page':i,'text':p.extract_text() or ''})
full_text='\n'.join(x['text'] for x in pages)
idx_t1=full_text.find('Table. S1 Data set used to fit the empirical model of the rheological properties')
idx_t2=full_text.find('Table. S2 Data set used to fit the empirical model of the 2D printing yields')
idx_t3=full_text.find('Table. S3 Data set used to fit the model of the angular 2D printing yields')
if min(idx_t1,idx_t2,idx_t3)<0:
    raise ValueError(f'Could not locate all source table captions: {idx_t1}, {idx_t2}, {idx_t3}')

def numbers(line):
    return re.findall(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', line)

# S1 appears before its caption; S2 and S3 likewise follow the preceding caption.
rrows=[]
for line in full_text[:idx_t1].splitlines():
    line=line.strip()
    if not re.match(r'^\d+\s+',line): continue
    n=numbers(line)
    if len(n)==9 and 1<=int(n[0])<=48: rrows.append(n)
rcols=['Run','Alg_mg_ml','HA_mg_ml','Eta0_Pa_s','Cross_m','Cross_k_s','G_prime_LVE_Pa','G_double_prime_LVE_Pa','Yield_stress_Pa']
s1=pd.DataFrame(rrows,columns=rcols).apply(pd.to_numeric)

prows=[]
for line in full_text[idx_t1:idx_t2].splitlines():
    line=line.strip()
    if not re.match(r'^\d+\s+',line): continue
    n=numbers(line)
    if len(n)==10 and 1<=int(n[0])<=43:
        prows.append(n)
    elif len(n)==6 and 1<=int(n[0])<=43:
        prows.append([n[0],n[1],n[2],n[3],np.nan,np.nan,np.nan,np.nan,n[4],n[5]])
pcols=['Run','Alg_mg_ml','HA_mg_ml','Pressure_kPa','Pr_mean','Pr_std','Qm_mean_mg_s','Qm_std_mg_s','SR_mean','SR_std']
s2=pd.DataFrame(prows,columns=pcols).apply(pd.to_numeric)

arows=[]
for line in full_text[idx_t2:idx_t3].splitlines():
    line=line.strip()
    if not re.match(r'^\d+\s+',line): continue
    n=numbers(line)
    if len(n)==7 and 1<=int(n[0])<=96: arows.append(n)
acols=['Run','Alg_mg_ml','HA_mg_ml','Pressure_kPa','Angle_deg','AF_mean','AF_std']
s3=pd.DataFrame(arows,columns=acols).apply(pd.to_numeric)

source={'S1':s1,'S2':s2,'S3':s3}
repo_paths={
    'S1':ROOT/'data/processed/rheology_table_S1.csv',
    'S2':ROOT/'data/processed/printability_table_S2.csv',
    'S3':ROOT/'data/processed/angle_fidelity_table_S3.csv',
}
summary=[]; mismatches=[]
for key,sdf in source.items():
    cdf=pd.read_csv(repo_paths[key])
    cols=list(sdf.columns)
    arr1=sdf[cols].astype(float).to_numpy(); arr2=cdf[cols].astype(float).to_numpy()
    same_nan=np.isnan(arr1)&np.isnan(arr2)
    finite=np.isfinite(arr1)&np.isfinite(arr2)
    diff=np.abs(arr1-arr2); diff[same_nan]=0
    mism=(~same_nan)&(~finite | (diff>1e-12))
    for i,j in zip(*np.where(mism)):
        mismatches.append({'Table':key,'Row_index':i,'Run':sdf.iloc[i]['Run'],'Column':cols[j],'Source':arr1[i,j],'Repository':arr2[i,j],'Abs_diff':diff[i,j] if np.isfinite(diff[i,j]) else None})
    summary.append({
        'Table':key,'Source_rows':len(sdf),'Repository_rows':len(cdf),
        'Nonmissing_numeric_entries':int(sdf.notna().sum().sum()),
        'Numeric_mismatch_count':int(mism.sum()),
        'Max_abs_difference':float(np.nanmax(np.where(finite,diff,np.nan))) if finite.any() else 0.0,
    })

pd.DataFrame(summary).to_csv(OUT/'Source_vs_Repository_Audit_Summary_Comment_2-R2.csv',index=False)
pd.DataFrame(mismatches).to_csv(OUT/'Source_vs_Repository_Audit_Mismatches_Comment_2-R2.csv',index=False)
meta={
    'comment':'2-R2','source_pdf_basename':pdf_path.name,
    'figure_digitization_used':False,
    'extraction_method':'pdfplumber text extraction and numeric parsing of Supporting Information Tables S1-S3',
    'audit_summary':summary,
    'total_nonmissing_numeric_entries':int(sum(x['Nonmissing_numeric_entries'] for x in summary)),
}
(OUT/'Data_Provenance_Audit_Metadata_Comment_2-R2.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(pd.DataFrame(summary).to_string(index=False))
print('Total mismatches:',len(mismatches))
