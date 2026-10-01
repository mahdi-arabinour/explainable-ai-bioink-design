from pathlib import Path
import json, re

ROOT=Path(__file__).resolve().parents[1]
PYFILE=ROOT/'src'/'run_analysis_from_curated_data.py'
NBFILE=ROOT/'notebooks'/'01_analysis_from_curated_data.ipynb'

remove_lines_patterns=[
    r'^\s*df\["TP_pct_wv"\]\s*=.*\n',
    r'^\s*df\["AF_alg_fraction"\]\s*=.*\n',
    r'^\s*df\["Alg_fraction_feature"\]\s*=.*\n',
    r'^\s*df\["Alg_fraction_x_HA_fraction"\]\s*=.*\n',
    r'^\s*df\["Angle_rad"\]\s*=.*\n',
    r'^\s*df\["sin_angle"\]\s*=.*\n',
    r'^\s*df\["cos_angle"\]\s*=.*\n',
]

old_eng='''engineered_features = [\n    "TP",\n    "TP_pct_wv",\n    "HF",\n    "Alg_fraction_feature",\n    "Alg_HA_ratio_safe",\n    "P_over_TP",\n    "P_over_Alg",\n    "P_over_HA",\n    "Alg_x_HA",\n    "Alg_x_P",\n    "HA_x_P",\n    "TP_x_P",\n    "HF_x_P",\n    "Alg_fraction_x_HA_fraction"\n]'''
new_eng='''engineered_features = [\n    # Reviewer-responsive nonredundant engineered representation.\n    # Exact scale aliases and deterministic fraction duplicates are excluded.\n    "TP",\n    "HF",\n    "Alg_HA_ratio_safe",\n    "P_over_TP",\n    "P_over_Alg",\n    "P_over_HA",\n    "Alg_x_HA",\n    "Alg_x_P",\n    "HA_x_P",\n    "TP_x_P",\n    "HF_x_P"\n]'''
old_ang='''angle_features = [\n    "Angle_deg",\n    "Angle_rad",\n    "sin_angle",\n    "cos_angle",\n    "P_x_Angle",\n    "Alg_x_Angle",\n    "HA_x_Angle",\n    "TP_x_Angle"\n]'''
new_ang='''angle_features = [\n    # Use the experimentally sampled angle directly; redundant reparameterizations\n    # (radian, sine, cosine) are excluded to avoid fragmented attribution.\n    "Angle_deg",\n    "P_x_Angle",\n    "Alg_x_Angle",\n    "HA_x_Angle",\n    "TP_x_Angle"\n]'''
old_gen='''engineered_general = [\n    "TP",\n    "TP_pct_wv",\n    "HF",\n    "Alg_fraction_feature",\n    "Alg_HA_ratio_safe",\n    "P_over_TP",\n    "P_over_Alg",\n    "P_over_HA",\n    "Alg_x_P",\n    "HA_x_P",\n    "TP_x_P",\n    "HF_x_P",\n    "Alg_x_HA",\n    "Alg_fraction_x_HA_fraction"\n]'''
new_gen='''engineered_general = [\n    "TP",\n    "HF",\n    "Alg_HA_ratio_safe",\n    "P_over_TP",\n    "P_over_Alg",\n    "P_over_HA",\n    "Alg_x_P",\n    "HA_x_P",\n    "TP_x_P",\n    "HF_x_P",\n    "Alg_x_HA"\n]'''
old_eang='''engineered_angle = [\n    "Angle_rad",\n    "sin_angle",\n    "cos_angle",\n    "P_x_Angle",\n    "Alg_x_Angle",\n    "HA_x_Angle",\n    "TP_x_Angle"\n]'''
new_eang='''engineered_angle = [\n    "P_x_Angle",\n    "Alg_x_Angle",\n    "HA_x_Angle",\n    "TP_x_Angle"\n]'''

# dictionary blocks to drop entirely
feature_names_to_drop=['TP_pct_wv','Alg_fraction_feature','Alg_fraction_x_HA_fraction','sin_angle','cos_angle']
def drop_feature_dict_blocks(text):
    for name in feature_names_to_drop:
        pat=re.compile(r'\n\s*\{\n\s*"Feature_name":\s*"'+re.escape(name)+r'",.*?\n\s*\},',re.S)
        text,n=pat.subn('',text,count=1)
        if n!=1:
            print('Warning: dictionary block not found',name)
    return text

def patch_text(text):
    for pat in remove_lines_patterns:
        text=re.sub(pat,'',text,flags=re.M)
    # clean_feature_names becomes an intentional no-op retained for notebook compatibility
    text=text.replace('''    rename_map = {\n        "AF_alg_fraction": "Alg_fraction_feature"\n    }\n\n    df = df.rename(columns=rename_map)\n    return df''','''    # No renaming is required after redundant fraction aliases were removed.\n    return df''')
    text=text.replace(old_eng,new_eng)
    text=text.replace(old_ang,new_ang)
    text=text.replace(old_gen,new_gen)
    text=text.replace(old_eang,new_eang)
    # diagnostic lists
    text=text.replace('''new_cols_printability = [\n    "TP", "TP_pct_wv", "HF", "AF_alg_fraction", "Alg_HA_ratio_safe",\n    "P_over_TP", "P_over_Alg", "P_over_HA",\n    "Alg_x_HA", "Alg_x_P", "HA_x_P", "TP_x_P", "HF_x_P",\n    "Alg_fraction_x_HA_fraction"\n]''','''new_cols_printability = [\n    "TP", "HF", "Alg_HA_ratio_safe",\n    "P_over_TP", "P_over_Alg", "P_over_HA",\n    "Alg_x_HA", "Alg_x_P", "HA_x_P", "TP_x_P", "HF_x_P"\n]''')
    text=text.replace('''new_cols_angle = new_cols_printability + [\n    "Angle_deg", "Angle_rad", "sin_angle", "cos_angle",\n    "P_x_Angle", "Alg_x_Angle", "HA_x_Angle", "TP_x_Angle"\n]''','''new_cols_angle = new_cols_printability + [\n    "Angle_deg", "P_x_Angle", "Alg_x_Angle", "HA_x_Angle", "TP_x_Angle"\n]''')
    text=drop_feature_dict_blocks(text)
    return text

# patch Python export
text=PYFILE.read_text(encoding='utf-8')
patched=patch_text(text)
PYFILE.write_text(patched,encoding='utf-8')

# patch notebook code cells
nb=json.loads(NBFILE.read_text(encoding='utf-8'))
for cell in nb['cells']:
    if cell.get('cell_type')!='code':
        continue
    s=''.join(cell.get('source',[]))
    ns=patch_text(s)
    if ns!=s:
        cell['source']=ns.splitlines(keepends=True)
NBFILE.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print('Patched:',PYFILE)
print('Patched:',NBFILE)
