#!/usr/bin/env python3
"""Python export of notebooks/01_analysis_from_curated_data.ipynb.

Run inside an IPython/Jupyter environment for best compatibility.
The notebook is the canonical executable workflow.
"""


# %% Cell 2

# Setup environment and repository paths
from pathlib import Path
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

warnings.filterwarnings("ignore")

# Detect repository root from the current working directory.
_cwd = Path.cwd().resolve()
_candidates = [_cwd, _cwd.parent, Path('/content/explainable-ai-bioink-design')]
PROJECT_DIR = next((p for p in _candidates if (p / 'data' / 'processed').exists()), None)
if PROJECT_DIR is None:
    raise FileNotFoundError(
        "Repository root not found. Run this notebook from the cloned repository "
        "or update PROJECT_DIR to the repository path."
    )

DATA_DIR = PROJECT_DIR / 'data'
RAW_DIR = DATA_DIR / 'source'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = PROJECT_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'

for folder in [PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print('Repository root:', PROJECT_DIR)
print('Pandas version:', pd.__version__)
print('NumPy version:', np.__version__)


# %% Cell 3

# Load curated source tables supplied with the repository
rheology_df = pd.read_csv(PROCESSED_DIR / 'rheology_table_S1.csv')
printability_df = pd.read_csv(PROCESSED_DIR / 'printability_table_S2.csv')
angle_df = pd.read_csv(PROCESSED_DIR / 'angle_fidelity_table_S3.csv')

print('Loaded curated datasets:')
print('Rheology:', rheology_df.shape)
print('2D printability:', printability_df.shape)
print('Angular fidelity:', angle_df.shape)

display(rheology_df.head())
display(printability_df.head())
display(angle_df.head())


# %% Cell 4

# Cell 4: Parse Table S1, S2, and S3 from extracted supplementary text

import re
import numpy as np
import pandas as pd

text_path = PROCESSED_DIR / "supplementary_extracted_text.txt"

with open(text_path, "r", encoding="utf-8") as f:
    full_text = f.read()

idx_t1 = full_text.find("Table. S1 Data set used to fit the empirical model of the rheological properties")
idx_t2 = full_text.find("Table. S2 Data set used to fit the empirical model of the 2D printing yields")
idx_t3 = full_text.find("Table. S3 Data set used to fit the model of the angular 2D printing yields")

print("Index Table S1 caption:", idx_t1)
print("Index Table S2 caption:", idx_t2)
print("Index Table S3 caption:", idx_t3)

if idx_t1 == -1 or idx_t2 == -1 or idx_t3 == -1:
    raise ValueError("One or more table captions were not found. Check extracted text.")

rheology_text = full_text[:idx_t1]
printability_text = full_text[idx_t1:idx_t2]
angle_text = full_text[idx_t2:idx_t3]

def extract_numbers_from_line(line):
    return re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", line)

# Parse rheology table
rheology_rows = []

for line in rheology_text.splitlines():
    line = line.strip()
    if not re.match(r"^\d+\s+", line):
        continue

    nums = extract_numbers_from_line(line)

    if len(nums) == 9:
        run = int(nums[0])
        if 1 <= run <= 48:
            rheology_rows.append(nums)

rheology_cols = [
    "Run", "Alg_mg_ml", "HA_mg_ml", "Eta0_Pa_s", "Cross_m",
    "Cross_k_s", "G_prime_LVE_Pa", "G_double_prime_LVE_Pa", "Yield_stress_Pa"
]

rheology_df = pd.DataFrame(rheology_rows, columns=rheology_cols)
rheology_df = rheology_df.apply(pd.to_numeric)

# Parse 2D printability table
printability_rows = []

for line in printability_text.splitlines():
    line = line.strip()
    if not re.match(r"^\d+\s+", line):
        continue

    nums = extract_numbers_from_line(line)

    if len(nums) == 10:
        run = int(nums[0])
        if 1 <= run <= 43:
            printability_rows.append(nums)

    elif len(nums) == 6:
        run = int(nums[0])
        if 1 <= run <= 43:
            # Sparse rows contain SR mean and SR std only
            row = [
                nums[0], nums[1], nums[2], nums[3],
                np.nan, np.nan, np.nan, np.nan,
                nums[4], nums[5]
            ]
            printability_rows.append(row)

printability_cols = [
    "Run", "Alg_mg_ml", "HA_mg_ml", "Pressure_kPa",
    "Pr_mean", "Pr_std", "Qm_mean_mg_s", "Qm_std_mg_s",
    "SR_mean", "SR_std"
]

printability_df = pd.DataFrame(printability_rows, columns=printability_cols)
printability_df = printability_df.apply(pd.to_numeric)

# Parse angular fidelity table
angle_rows = []

for line in angle_text.splitlines():
    line = line.strip()
    if not re.match(r"^\d+\s+", line):
        continue

    nums = extract_numbers_from_line(line)

    if len(nums) == 7:
        run = int(nums[0])
        if 1 <= run <= 96:
            angle_rows.append(nums)

angle_cols = [
    "Run", "Alg_mg_ml", "HA_mg_ml", "Pressure_kPa",
    "Angle_deg", "AF_mean", "AF_std"
]

angle_df = pd.DataFrame(angle_rows, columns=angle_cols)
angle_df = angle_df.apply(pd.to_numeric)

# Save parsed datasets
rheology_path = PROCESSED_DIR / "rheology_table_S1.csv"
printability_path = PROCESSED_DIR / "printability_table_S2.csv"
angle_path = PROCESSED_DIR / "angle_fidelity_table_S3.csv"

rheology_df.to_csv(rheology_path, index=False)
printability_df.to_csv(printability_path, index=False)
angle_df.to_csv(angle_path, index=False)

print("\nParsed dataset shapes:")
print("Rheology:", rheology_df.shape)
print("Printability:", printability_df.shape)
print("Angle fidelity:", angle_df.shape)

print("\nRheology preview:")
display(rheology_df.head())
display(rheology_df.tail())

print("\nPrintability preview:")
display(printability_df.head())
display(printability_df.tail())

print("\nAngle fidelity preview:")
display(angle_df.head())
display(angle_df.tail())

print("\nMissing values in printability dataset:")
print(printability_df.isna().sum())

print("\nSaved files:")
print(rheology_path)
print(printability_path)
print(angle_path)


# %% Cell 5

# Cell 5: Basic quality control for parsed datasets

def print_dataset_qc(df, name):
    print("=" * 80)
    print(name)
    print("=" * 80)
    print("Shape:", df.shape)
    print("\nColumn types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nNumeric summary:")
    display(df.describe().T)
    print("\nDuplicated rows:", df.duplicated().sum())

print_dataset_qc(rheology_df, "Rheology dataset")
print_dataset_qc(printability_df, "2D printability dataset")
print_dataset_qc(angle_df, "Angle fidelity dataset")

print("\nUnique formulations in rheology:")
display(
    rheology_df[["Alg_mg_ml", "HA_mg_ml"]]
    .drop_duplicates()
    .sort_values(["Alg_mg_ml", "HA_mg_ml"])
    .reset_index(drop=True)
)

print("\nUnique formulations in 2D printability:")
display(
    printability_df[["Alg_mg_ml", "HA_mg_ml"]]
    .drop_duplicates()
    .sort_values(["Alg_mg_ml", "HA_mg_ml"])
    .reset_index(drop=True)
)

print("\nUnique formulations in angle fidelity:")
display(
    angle_df[["Alg_mg_ml", "HA_mg_ml"]]
    .drop_duplicates()
    .sort_values(["Alg_mg_ml", "HA_mg_ml"])
    .reset_index(drop=True)
)

print("\nRows with missing Pr or Qm in 2D printability:")
display(
    printability_df[
        printability_df[["Pr_mean", "Qm_mean_mg_s"]].isna().any(axis=1)
    ]
)


# %% Cell 6

# Cell 6: Harmonize units and create formulation labels

def add_formulation_features(df):
    df = df.copy()

    df["Alg_pct_wv"] = df["Alg_mg_ml"] / 10
    df["HA_pct_wv"] = df["HA_mg_ml"] / 10

    df["Total_polymer_mg_ml"] = df["Alg_mg_ml"] + df["HA_mg_ml"]
    df["Total_polymer_pct_wv"] = df["Total_polymer_mg_ml"] / 10

    df["HA_fraction"] = df["HA_mg_ml"] / df["Total_polymer_mg_ml"]
    df["Alg_fraction"] = df["Alg_mg_ml"] / df["Total_polymer_mg_ml"]
    df["Alg_HA_ratio"] = df["Alg_mg_ml"] / df["HA_mg_ml"]

    df["Formulation_label"] = (
        df["Alg_pct_wv"].astype(int).astype(str)
        + "ALG"
        + df["HA_pct_wv"].astype(int).astype(str)
        + "HA"
    )

    return df

rheology_std = add_formulation_features(rheology_df)
printability_std = add_formulation_features(printability_df)
angle_std = add_formulation_features(angle_df)

rheology_std_path = PROCESSED_DIR / "rheology_standardized.csv"
printability_std_path = PROCESSED_DIR / "printability_standardized.csv"
angle_std_path = PROCESSED_DIR / "angle_fidelity_standardized.csv"

rheology_std.to_csv(rheology_std_path, index=False)
printability_std.to_csv(printability_std_path, index=False)
angle_std.to_csv(angle_std_path, index=False)

print("Standardized files saved:")
print(rheology_std_path)
print(printability_std_path)
print(angle_std_path)

print("\nRheology standardized preview:")
display(rheology_std.head())

print("\n2D printability standardized preview:")
display(printability_std.head())

print("\nAngle fidelity standardized preview:")
display(angle_std.head())

print("\nFormulation labels in 2D printability:")
display(
    printability_std[["Formulation_label", "Alg_mg_ml", "HA_mg_ml", "Alg_pct_wv", "HA_pct_wv"]]
    .drop_duplicates()
    .sort_values(["Alg_mg_ml", "HA_mg_ml"])
    .reset_index(drop=True)
)


# %% Cell 7

# Cell 7: Create engineered features for machine learning

def add_ml_features(df, include_angle=False):
    df = df.copy()

    # Core composition features
    df["TP"] = df["Total_polymer_mg_ml"]
    df["TP_pct_wv"] = df["Total_polymer_pct_wv"]
    df["HF"] = df["HA_fraction"]
    df["AF_alg_fraction"] = df["Alg_fraction"]
    df["Alg_HA_ratio_safe"] = df["Alg_HA_ratio"]

    # Process-normalized features
    if "Pressure_kPa" in df.columns:
        df["P_over_TP"] = df["Pressure_kPa"] / df["TP"]
        df["P_over_Alg"] = df["Pressure_kPa"] / df["Alg_mg_ml"]
        df["P_over_HA"] = df["Pressure_kPa"] / df["HA_mg_ml"]

        # Polymer-pressure interaction features
        df["Alg_x_P"] = df["Alg_mg_ml"] * df["Pressure_kPa"]
        df["HA_x_P"] = df["HA_mg_ml"] * df["Pressure_kPa"]
        df["TP_x_P"] = df["TP"] * df["Pressure_kPa"]
        df["HF_x_P"] = df["HF"] * df["Pressure_kPa"]

    # Polymer-polymer interaction features
    df["Alg_x_HA"] = df["Alg_mg_ml"] * df["HA_mg_ml"]
    df["Alg_fraction_x_HA_fraction"] = df["Alg_fraction"] * df["HA_fraction"]

    # Angle-related features for angular fidelity model
    if include_angle and "Angle_deg" in df.columns:
        df["Angle_rad"] = np.deg2rad(df["Angle_deg"])
        df["sin_angle"] = np.sin(df["Angle_rad"])
        df["cos_angle"] = np.cos(df["Angle_rad"])

        if "Pressure_kPa" in df.columns:
            df["P_x_Angle"] = df["Pressure_kPa"] * df["Angle_deg"]
            df["Alg_x_Angle"] = df["Alg_mg_ml"] * df["Angle_deg"]
            df["HA_x_Angle"] = df["HA_mg_ml"] * df["Angle_deg"]
            df["TP_x_Angle"] = df["TP"] * df["Angle_deg"]

    return df

rheology_ml = add_ml_features(rheology_std, include_angle=False)
printability_ml = add_ml_features(printability_std, include_angle=False)
angle_ml = add_ml_features(angle_std, include_angle=True)

rheology_ml_path = PROCESSED_DIR / "rheology_ml_features.csv"
printability_ml_path = PROCESSED_DIR / "printability_ml_features.csv"
angle_ml_path = PROCESSED_DIR / "angle_fidelity_ml_features.csv"

rheology_ml.to_csv(rheology_ml_path, index=False)
printability_ml.to_csv(printability_ml_path, index=False)
angle_ml.to_csv(angle_ml_path, index=False)

print("ML feature files saved:")
print(rheology_ml_path)
print(printability_ml_path)
print(angle_ml_path)

print("\nRheology ML shape:", rheology_ml.shape)
print("Printability ML shape:", printability_ml.shape)
print("Angle fidelity ML shape:", angle_ml.shape)

print("\nNew ML feature columns in printability dataset:")
new_cols_printability = [
    "TP", "TP_pct_wv", "HF", "AF_alg_fraction", "Alg_HA_ratio_safe",
    "P_over_TP", "P_over_Alg", "P_over_HA",
    "Alg_x_HA", "Alg_x_P", "HA_x_P", "TP_x_P", "HF_x_P",
    "Alg_fraction_x_HA_fraction"
]
print(new_cols_printability)

display(printability_ml[[
    "Formulation_label", "Alg_mg_ml", "HA_mg_ml", "Pressure_kPa",
    "Pr_mean", "Qm_mean_mg_s", "SR_mean"
] + new_cols_printability].head())

print("\nNew ML feature columns in angle fidelity dataset:")
new_cols_angle = new_cols_printability + [
    "Angle_deg", "Angle_rad", "sin_angle", "cos_angle",
    "P_x_Angle", "Alg_x_Angle", "HA_x_Angle", "TP_x_Angle"
]
print(new_cols_angle)

display(angle_ml[[
    "Formulation_label", "Alg_mg_ml", "HA_mg_ml", "Pressure_kPa",
    "Angle_deg", "AF_mean"
] + [col for col in new_cols_angle if col not in ["Angle_deg"]]].head())

print("\nMissing values after feature engineering:")
print("Rheology:")
print(rheology_ml.isna().sum()[rheology_ml.isna().sum() > 0])

print("\nPrintability:")
print(printability_ml.isna().sum()[printability_ml.isna().sum() > 0])

print("\nAngle fidelity:")
print(angle_ml.isna().sum()[angle_ml.isna().sum() > 0])


# %% Cell 8

# Cell 7b: Rename ambiguous feature names before continuing

def clean_feature_names(df):
    df = df.copy()

    rename_map = {
        "AF_alg_fraction": "Alg_fraction_feature"
    }

    df = df.rename(columns=rename_map)
    return df

rheology_ml = clean_feature_names(rheology_ml)
printability_ml = clean_feature_names(printability_ml)
angle_ml = clean_feature_names(angle_ml)

# Save corrected feature files
rheology_ml.to_csv(PROCESSED_DIR / "rheology_ml_features.csv", index=False)
printability_ml.to_csv(PROCESSED_DIR / "printability_ml_features.csv", index=False)
angle_ml.to_csv(PROCESSED_DIR / "angle_fidelity_ml_features.csv", index=False)

print("Feature names corrected and files saved.")

print("\nColumns containing 'AF':")
print([col for col in angle_ml.columns if "AF" in col])

print("\nColumns containing 'fraction':")
print([col for col in printability_ml.columns if "fraction" in col])

print("\nPrintability ML shape:", printability_ml.shape)
print("Angle fidelity ML shape:", angle_ml.shape)


# %% Cell 9

# Cell 8: Create feature dictionary for manuscript table

feature_dictionary = [
    {
        "Feature_name": "Alg_mg_ml",
        "Symbol": "Alg",
        "Group": "Original input",
        "Definition": "Alginate concentration",
        "Unit": "mg/mL",
        "Used_for": "Rheology, Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "HA_mg_ml",
        "Symbol": "HA",
        "Group": "Original input",
        "Definition": "Hyaluronic acid concentration",
        "Unit": "mg/mL",
        "Used_for": "Rheology, Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "Pressure_kPa",
        "Symbol": "P",
        "Group": "Original input",
        "Definition": "Extrusion pressure",
        "Unit": "kPa",
        "Used_for": "Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "Angle_deg",
        "Symbol": "theta",
        "Group": "Original input",
        "Definition": "Printed angle",
        "Unit": "degree",
        "Used_for": "AF"
    },
    {
        "Feature_name": "TP",
        "Symbol": "TP",
        "Group": "Engineered composition feature",
        "Definition": "Total polymer concentration, calculated as Alg + HA",
        "Unit": "mg/mL",
        "Used_for": "Rheology, Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "TP_pct_wv",
        "Symbol": "TP_percent",
        "Group": "Engineered composition feature",
        "Definition": "Total polymer concentration converted to percent weight per volume",
        "Unit": "% w/v",
        "Used_for": "Rheology, Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "HF",
        "Symbol": "HF",
        "Group": "Engineered composition feature",
        "Definition": "Hyaluronic acid fraction, calculated as HA / (Alg + HA)",
        "Unit": "unitless",
        "Used_for": "Rheology, Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "Alg_fraction_feature",
        "Symbol": "AlgF",
        "Group": "Engineered composition feature",
        "Definition": "Alginate fraction, calculated as Alg / (Alg + HA)",
        "Unit": "unitless",
        "Used_for": "Rheology, Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "Alg_HA_ratio_safe",
        "Symbol": "Alg/HA",
        "Group": "Engineered composition feature",
        "Definition": "Alginate-to-hyaluronic acid concentration ratio",
        "Unit": "unitless",
        "Used_for": "Rheology, Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "P_over_TP",
        "Symbol": "P/TP",
        "Group": "Engineered process feature",
        "Definition": "Extrusion pressure normalized by total polymer concentration",
        "Unit": "kPa per mg/mL",
        "Used_for": "Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "P_over_Alg",
        "Symbol": "P/Alg",
        "Group": "Engineered process feature",
        "Definition": "Extrusion pressure normalized by alginate concentration",
        "Unit": "kPa per mg/mL",
        "Used_for": "Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "P_over_HA",
        "Symbol": "P/HA",
        "Group": "Engineered process feature",
        "Definition": "Extrusion pressure normalized by hyaluronic acid concentration",
        "Unit": "kPa per mg/mL",
        "Used_for": "Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "Alg_x_HA",
        "Symbol": "Alg x HA",
        "Group": "Polymer interaction feature",
        "Definition": "Alginate and hyaluronic acid interaction term",
        "Unit": "(mg/mL)^2",
        "Used_for": "Rheology, Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "Alg_x_P",
        "Symbol": "Alg x P",
        "Group": "Polymer-pressure interaction feature",
        "Definition": "Alginate and extrusion pressure interaction term",
        "Unit": "mg/mL x kPa",
        "Used_for": "Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "HA_x_P",
        "Symbol": "HA x P",
        "Group": "Polymer-pressure interaction feature",
        "Definition": "Hyaluronic acid and extrusion pressure interaction term",
        "Unit": "mg/mL x kPa",
        "Used_for": "Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "TP_x_P",
        "Symbol": "TP x P",
        "Group": "Polymer-pressure interaction feature",
        "Definition": "Total polymer concentration and extrusion pressure interaction term",
        "Unit": "mg/mL x kPa",
        "Used_for": "Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "HF_x_P",
        "Symbol": "HF x P",
        "Group": "Composition-pressure interaction feature",
        "Definition": "Hyaluronic acid fraction and extrusion pressure interaction term",
        "Unit": "kPa",
        "Used_for": "Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "Alg_fraction_x_HA_fraction",
        "Symbol": "AlgF x HF",
        "Group": "Composition interaction feature",
        "Definition": "Interaction between alginate fraction and hyaluronic acid fraction",
        "Unit": "unitless",
        "Used_for": "Rheology, Pr, SR, Qm, AF"
    },
    {
        "Feature_name": "sin_angle",
        "Symbol": "sin(theta)",
        "Group": "Angle feature",
        "Definition": "Sine transformation of printing angle",
        "Unit": "unitless",
        "Used_for": "AF"
    },
    {
        "Feature_name": "cos_angle",
        "Symbol": "cos(theta)",
        "Group": "Angle feature",
        "Definition": "Cosine transformation of printing angle",
        "Unit": "unitless",
        "Used_for": "AF"
    },
    {
        "Feature_name": "P_x_Angle",
        "Symbol": "P x theta",
        "Group": "Pressure-angle interaction feature",
        "Definition": "Extrusion pressure and printing angle interaction term",
        "Unit": "kPa x degree",
        "Used_for": "AF"
    }
]

feature_dictionary_df = pd.DataFrame(feature_dictionary)

feature_dictionary_path = TABLES_DIR / "feature_dictionary.csv"
feature_dictionary_excel_path = TABLES_DIR / "feature_dictionary.xlsx"

feature_dictionary_df.to_csv(feature_dictionary_path, index=False)
feature_dictionary_df.to_excel(feature_dictionary_excel_path, index=False)

print("Feature dictionary saved:")
print(feature_dictionary_path)
print(feature_dictionary_excel_path)

print("\nFeature dictionary shape:", feature_dictionary_df.shape)
display(feature_dictionary_df)


# %% Cell 10

# Cell 9: Build master datasets by merging rheology descriptors with printability datasets

# Step 1: Aggregate rheology measurements by formulation
rheology_response_cols = [
    "Eta0_Pa_s",
    "Cross_m",
    "Cross_k_s",
    "G_prime_LVE_Pa",
    "G_double_prime_LVE_Pa",
    "Yield_stress_Pa"
]

rheology_key_cols = [
    "Alg_mg_ml",
    "HA_mg_ml",
    "Alg_pct_wv",
    "HA_pct_wv",
    "Total_polymer_mg_ml",
    "Total_polymer_pct_wv",
    "HA_fraction",
    "Alg_fraction",
    "Alg_HA_ratio",
    "Formulation_label"
]

rheology_summary = (
    rheology_ml
    .groupby(rheology_key_cols, as_index=False)[rheology_response_cols]
    .agg(["mean", "std"])
)

# Flatten multi-level columns
rheology_summary.columns = [
    "_".join(col).strip("_") if isinstance(col, tuple) else col
    for col in rheology_summary.columns
]

# Rename columns for clarity
rename_rheology_summary = {
    "Eta0_Pa_s_mean": "Eta0_mean_Pa_s",
    "Eta0_Pa_s_std": "Eta0_std_Pa_s",
    "Cross_m_mean": "Cross_m_mean",
    "Cross_m_std": "Cross_m_std",
    "Cross_k_s_mean": "Cross_k_mean_s",
    "Cross_k_s_std": "Cross_k_std_s",
    "G_prime_LVE_Pa_mean": "G_prime_mean_Pa",
    "G_prime_LVE_Pa_std": "G_prime_std_Pa",
    "G_double_prime_LVE_Pa_mean": "G_double_prime_mean_Pa",
    "G_double_prime_LVE_Pa_std": "G_double_prime_std_Pa",
    "Yield_stress_Pa_mean": "Yield_stress_mean_Pa",
    "Yield_stress_Pa_std": "Yield_stress_std_Pa"
}

rheology_summary = rheology_summary.rename(columns=rename_rheology_summary)

# Step 2: Keep only rheology columns needed for merging
rheology_merge_cols = [
    "Alg_mg_ml",
    "HA_mg_ml",
    "Eta0_mean_Pa_s",
    "Eta0_std_Pa_s",
    "Cross_m_mean",
    "Cross_m_std",
    "Cross_k_mean_s",
    "Cross_k_std_s",
    "G_prime_mean_Pa",
    "G_prime_std_Pa",
    "G_double_prime_mean_Pa",
    "G_double_prime_std_Pa",
    "Yield_stress_mean_Pa",
    "Yield_stress_std_Pa"
]

rheology_for_merge = rheology_summary[rheology_merge_cols].copy()

# Step 3: Merge rheology descriptors into 2D printability dataset
master_printability = printability_ml.merge(
    rheology_for_merge,
    on=["Alg_mg_ml", "HA_mg_ml"],
    how="left",
    validate="many_to_one"
)

# Step 4: Merge rheology descriptors into angle fidelity dataset
master_angle = angle_ml.merge(
    rheology_for_merge,
    on=["Alg_mg_ml", "HA_mg_ml"],
    how="left",
    validate="many_to_one"
)

# Step 5: Save master datasets
rheology_summary_path = PROCESSED_DIR / "rheology_formulation_summary.csv"
master_printability_path = PROCESSED_DIR / "master_printability_dataset.csv"
master_angle_path = PROCESSED_DIR / "master_angle_fidelity_dataset.csv"

rheology_summary.to_csv(rheology_summary_path, index=False)
master_printability.to_csv(master_printability_path, index=False)
master_angle.to_csv(master_angle_path, index=False)

print("Master datasets saved:")
print(rheology_summary_path)
print(master_printability_path)
print(master_angle_path)

print("\nDataset shapes:")
print("Rheology formulation summary:", rheology_summary.shape)
print("Master printability:", master_printability.shape)
print("Master angle fidelity:", master_angle.shape)

print("\nMissing values in master printability:")
print(master_printability.isna().sum()[master_printability.isna().sum() > 0])

print("\nMissing values in master angle fidelity:")
print(master_angle.isna().sum()[master_angle.isna().sum() > 0])

print("\nRheology formulation summary preview:")
display(rheology_summary.head())

print("\nMaster printability preview:")
display(master_printability.head())

print("\nMaster angle fidelity preview:")
display(master_angle.head())


# %% Cell 11

# Cell 10: Build target-specific modeling datasets and feature sets

import json

# Core experimental features
core_features = [
    "Alg_mg_ml",
    "HA_mg_ml",
    "Pressure_kPa"
]

# Engineered formulation and process features
engineered_features = [
    "TP",
    "TP_pct_wv",
    "HF",
    "Alg_fraction_feature",
    "Alg_HA_ratio_safe",
    "P_over_TP",
    "P_over_Alg",
    "P_over_HA",
    "Alg_x_HA",
    "Alg_x_P",
    "HA_x_P",
    "TP_x_P",
    "HF_x_P",
    "Alg_fraction_x_HA_fraction"
]

# Rheology-derived mechanistic descriptors
rheology_features = [
    "Eta0_mean_Pa_s",
    "Cross_m_mean",
    "Cross_k_mean_s",
    "G_prime_mean_Pa",
    "G_double_prime_mean_Pa",
    "Yield_stress_mean_Pa"
]

# Angle-specific features
angle_features = [
    "Angle_deg",
    "Angle_rad",
    "sin_angle",
    "cos_angle",
    "P_x_Angle",
    "Alg_x_Angle",
    "HA_x_Angle",
    "TP_x_Angle"
]

# Feature-set definitions
feature_sets = {
    "core": core_features,
    "engineered": core_features + engineered_features,
    "rheology_enhanced": core_features + engineered_features + rheology_features,
    "angle_engineered": core_features + engineered_features + angle_features,
    "angle_rheology_enhanced": core_features + engineered_features + angle_features + rheology_features
}

# Target definitions
target_configs = {
    "Pr": {
        "dataset": master_printability,
        "target": "Pr_mean",
        "allowed_feature_sets": ["core", "engineered", "rheology_enhanced"]
    },
    "SR": {
        "dataset": master_printability,
        "target": "SR_mean",
        "allowed_feature_sets": ["core", "engineered", "rheology_enhanced"]
    },
    "Qm": {
        "dataset": master_printability,
        "target": "Qm_mean_mg_s",
        "allowed_feature_sets": ["core", "engineered", "rheology_enhanced"]
    },
    "AF": {
        "dataset": master_angle,
        "target": "AF_mean",
        "allowed_feature_sets": ["core", "angle_engineered", "angle_rheology_enhanced"]
    }
}

modeling_tables = {}

for target_name, config in target_configs.items():
    df = config["dataset"].copy()
    target_col = config["target"]

    # Keep rows with valid target values only
    df_model = df.dropna(subset=[target_col]).reset_index(drop=True)

    modeling_tables[target_name] = df_model

    output_path = PROCESSED_DIR / f"modeling_dataset_{target_name}.csv"
    df_model.to_csv(output_path, index=False)

    print("=" * 80)
    print(f"Target: {target_name}")
    print("=" * 80)
    print("Target column:", target_col)
    print("Rows before dropping missing target:", df.shape[0])
    print("Rows after dropping missing target:", df_model.shape[0])
    print("Saved to:", output_path)

    print("\nTarget summary:")
    display(df_model[target_col].describe().to_frame().T)

    print("\nAllowed feature sets:")
    for fs_name in config["allowed_feature_sets"]:
        features = feature_sets[fs_name]
        missing_features = [col for col in features if col not in df_model.columns]
        print(f"- {fs_name}: {len(features)} features")
        if missing_features:
            print("  Missing features:", missing_features)

# Save feature-set metadata
metadata = {
    "feature_sets": feature_sets,
    "target_configs": {
        target_name: {
            "target": config["target"],
            "allowed_feature_sets": config["allowed_feature_sets"]
        }
        for target_name, config in target_configs.items()
    }
}

metadata_path = PROCESSED_DIR / "modeling_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("\nModeling metadata saved:")
print(metadata_path)


# %% Cell 12

# Cell 11: Baseline visual checks for modeling targets

target_plot_configs = {
    "Pr": {
        "df": modeling_tables["Pr"],
        "target": "Pr_mean",
        "label": "Printability Index (Pr)"
    },
    "SR": {
        "df": modeling_tables["SR"],
        "target": "SR_mean",
        "label": "Spreading Ratio (SR)"
    },
    "Qm": {
        "df": modeling_tables["Qm"],
        "target": "Qm_mean_mg_s",
        "label": "Mass Flow Rate (Qm, mg/s)"
    },
    "AF": {
        "df": modeling_tables["AF"],
        "target": "AF_mean",
        "label": "Angle Fidelity Factor (AF)"
    }
}

eda_summary_rows = []

for target_name, config in target_plot_configs.items():
    df = config["df"]
    target_col = config["target"]
    target_label = config["label"]

    eda_summary_rows.append({
        "Target": target_name,
        "Target_column": target_col,
        "N": len(df),
        "Mean": df[target_col].mean(),
        "Std": df[target_col].std(),
        "Min": df[target_col].min(),
        "Median": df[target_col].median(),
        "Max": df[target_col].max()
    })

    # Target distribution
    plt.figure(figsize=(6, 4))
    plt.hist(df[target_col], bins=10)
    plt.xlabel(target_label)
    plt.ylabel("Count")
    plt.title(f"Distribution of {target_name}")
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"distribution_{target_name}.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()

    # Pressure vs target
    if "Pressure_kPa" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(df["Pressure_kPa"], df[target_col])
        plt.xlabel("Pressure (kPa)")
        plt.ylabel(target_label)
        plt.title(f"Pressure vs {target_name}")
        plt.tight_layout()
        fig_path = FIGURES_DIR / f"pressure_vs_{target_name}.png"
        plt.savefig(fig_path, dpi=300)
        plt.show()

    # Total polymer vs target
    if "TP" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(df["TP"], df[target_col])
        plt.xlabel("Total Polymer (mg/mL)")
        plt.ylabel(target_label)
        plt.title(f"Total Polymer vs {target_name}")
        plt.tight_layout()
        fig_path = FIGURES_DIR / f"total_polymer_vs_{target_name}.png"
        plt.savefig(fig_path, dpi=300)
        plt.show()

    # HA fraction vs target
    if "HF" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(df["HF"], df[target_col])
        plt.xlabel("HA Fraction")
        plt.ylabel(target_label)
        plt.title(f"HA Fraction vs {target_name}")
        plt.tight_layout()
        fig_path = FIGURES_DIR / f"ha_fraction_vs_{target_name}.png"
        plt.savefig(fig_path, dpi=300)
        plt.show()

# Extra angle-specific visual check
af_df = modeling_tables["AF"]

plt.figure(figsize=(6, 4))
plt.scatter(af_df["Angle_deg"], af_df["AF_mean"])
plt.xlabel("Angle (degree)")
plt.ylabel("Angle Fidelity Factor (AF)")
plt.title("Angle vs AF")
plt.tight_layout()
fig_path = FIGURES_DIR / "angle_vs_AF.png"
plt.savefig(fig_path, dpi=300)
plt.show()

eda_summary_df = pd.DataFrame(eda_summary_rows)
eda_summary_path = TABLES_DIR / "eda_target_summary.csv"
eda_summary_excel_path = TABLES_DIR / "eda_target_summary.xlsx"

eda_summary_df.to_csv(eda_summary_path, index=False)
eda_summary_df.to_excel(eda_summary_excel_path, index=False)

print("EDA summary saved:")
print(eda_summary_path)
print(eda_summary_excel_path)

print("\nEDA target summary:")
display(eda_summary_df)

print("\nSaved figures:")
for file in sorted(FIGURES_DIR.glob("*.png")):
    print("-", file.name)


# %% Cell 13

# Cell 12: Feature-target correlation analysis

correlation_results = []

main_feature_set_for_target = {
    "Pr": "rheology_enhanced",
    "SR": "rheology_enhanced",
    "Qm": "rheology_enhanced",
    "AF": "angle_rheology_enhanced"
}

for target_name, config in target_configs.items():
    df = modeling_tables[target_name].copy()
    target_col = config["target"]
    feature_set_name = main_feature_set_for_target[target_name]
    features = feature_sets[feature_set_name]

    valid_features = [col for col in features if col in df.columns]

    for feature in valid_features:
        temp = df[[feature, target_col]].dropna()

        if temp[feature].nunique() <= 1:
            pearson_corr = np.nan
            spearman_corr = np.nan
        else:
            pearson_corr = temp[feature].corr(temp[target_col], method="pearson")
            spearman_corr = temp[feature].corr(temp[target_col], method="spearman")

        correlation_results.append({
            "Target": target_name,
            "Target_column": target_col,
            "Feature_set": feature_set_name,
            "Feature": feature,
            "Pearson_r": pearson_corr,
            "Spearman_r": spearman_corr,
            "Abs_Pearson_r": abs(pearson_corr) if pd.notna(pearson_corr) else np.nan,
            "Abs_Spearman_r": abs(spearman_corr) if pd.notna(spearman_corr) else np.nan,
            "N": len(temp)
        })

correlation_df = pd.DataFrame(correlation_results)

correlation_path = TABLES_DIR / "feature_target_correlations.csv"
correlation_excel_path = TABLES_DIR / "feature_target_correlations.xlsx"

correlation_df.to_csv(correlation_path, index=False)
correlation_df.to_excel(correlation_excel_path, index=False)

print("Correlation tables saved:")
print(correlation_path)
print(correlation_excel_path)

for target_name in ["Pr", "SR", "Qm", "AF"]:
    print("\n" + "=" * 80)
    print(f"Top Spearman correlations for {target_name}")
    print("=" * 80)

    top_corr = (
        correlation_df[correlation_df["Target"] == target_name]
        .sort_values("Abs_Spearman_r", ascending=False)
        .head(12)
        .reset_index(drop=True)
    )

    display(top_corr[[
        "Target",
        "Feature",
        "Pearson_r",
        "Spearman_r",
        "Abs_Spearman_r",
        "N"
    ]])

    plt.figure(figsize=(7, 5))
    plot_df = top_corr.sort_values("Abs_Spearman_r", ascending=True)
    plt.barh(plot_df["Feature"], plot_df["Abs_Spearman_r"])
    plt.xlabel("Absolute Spearman correlation")
    plt.ylabel("Feature")
    plt.title(f"Top feature-target correlations for {target_name}")
    plt.tight_layout()

    fig_path = FIGURES_DIR / f"top_correlations_{target_name}.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()

print("\nSaved correlation figures:")
for file in sorted(FIGURES_DIR.glob("top_correlations_*.png")):
    print("-", file.name)


# %% Cell 14

# Cell 13: Train and evaluate machine learning models with repeated K-fold cross-validation

from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

try:
    from xgboost import XGBRegressor
    xgboost_available = True
except Exception as e:
    print("XGBoost import failed:", e)
    xgboost_available = False

RANDOM_STATE = 42

def get_models():
    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "SVR_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10.0, epsilon=0.05))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE
        )
    }

    if xgboost_available:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1
        )

    return models

def safe_spearman(y_true, y_pred):
    if len(np.unique(y_true)) <= 1 or len(np.unique(y_pred)) <= 1:
        return np.nan
    return spearmanr(y_true, y_pred).correlation

def evaluate_model_cv(df, features, target_col, model, n_splits=5, n_repeats=20):
    X = df[features].copy()
    y = df[target_col].copy()

    cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE
    )

    split_metrics = []

    for split_id, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        spearman = safe_spearman(y_test, y_pred)

        split_metrics.append({
            "Split": split_id,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "Spearman": spearman
        })

    split_metrics_df = pd.DataFrame(split_metrics)

    summary = {
        "R2_mean": split_metrics_df["R2"].mean(),
        "R2_std": split_metrics_df["R2"].std(),
        "MAE_mean": split_metrics_df["MAE"].mean(),
        "MAE_std": split_metrics_df["MAE"].std(),
        "RMSE_mean": split_metrics_df["RMSE"].mean(),
        "RMSE_std": split_metrics_df["RMSE"].std(),
        "Spearman_mean": split_metrics_df["Spearman"].mean(),
        "Spearman_std": split_metrics_df["Spearman"].std()
    }

    return summary, split_metrics_df

all_performance_rows = []
all_split_metrics = []

for target_name, config in target_configs.items():
    df_model = modeling_tables[target_name].copy()
    target_col = config["target"]
    allowed_feature_sets = config["allowed_feature_sets"]

    for feature_set_name in allowed_feature_sets:
        features = feature_sets[feature_set_name]
        features = [col for col in features if col in df_model.columns]

        X_check = df_model[features]
        missing_count = X_check.isna().sum().sum()

        if missing_count > 0:
            print(f"Skipping {target_name} - {feature_set_name} due to missing feature values.")
            continue

        models = get_models()

        for model_name, model in models.items():
            print(f"Running: Target={target_name}, FeatureSet={feature_set_name}, Model={model_name}")

            summary, split_metrics_df = evaluate_model_cv(
                df=df_model,
                features=features,
                target_col=target_col,
                model=model,
                n_splits=5,
                n_repeats=20
            )

            row = {
                "Target": target_name,
                "Target_column": target_col,
                "Feature_set": feature_set_name,
                "Model": model_name,
                "N": len(df_model),
                "N_features": len(features),
                **summary
            }

            all_performance_rows.append(row)

            split_metrics_df["Target"] = target_name
            split_metrics_df["Target_column"] = target_col
            split_metrics_df["Feature_set"] = feature_set_name
            split_metrics_df["Model"] = model_name
            all_split_metrics.append(split_metrics_df)

performance_df = pd.DataFrame(all_performance_rows)
split_metrics_all_df = pd.concat(all_split_metrics, ignore_index=True)

performance_path = TABLES_DIR / "ml_model_performance_repeated_kfold.csv"
performance_excel_path = TABLES_DIR / "ml_model_performance_repeated_kfold.xlsx"
split_metrics_path = TABLES_DIR / "ml_split_metrics_repeated_kfold.csv"

performance_df.to_csv(performance_path, index=False)
performance_df.to_excel(performance_excel_path, index=False)
split_metrics_all_df.to_csv(split_metrics_path, index=False)

print("\nModel performance saved:")
print(performance_path)
print(performance_excel_path)
print(split_metrics_path)

print("\nTop models by target using RMSE:")
for target_name in ["Pr", "SR", "Qm", "AF"]:
    print("\n" + "=" * 80)
    print(f"Target: {target_name}")
    print("=" * 80)

    top_target = (
        performance_df[performance_df["Target"] == target_name]
        .sort_values(["RMSE_mean", "MAE_mean"], ascending=True)
        .reset_index(drop=True)
    )

    display(top_target[[
        "Target",
        "Feature_set",
        "Model",
        "N",
        "N_features",
        "R2_mean",
        "R2_std",
        "MAE_mean",
        "RMSE_mean",
        "Spearman_mean"
    ]].head(10))


# %% Cell 15

# Cell 14: Leave-one-formulation-out validation

from sklearn.base import clone

def evaluate_model_lofo(df, features, target_col, model, group_col="Formulation_label"):
    X = df[features].copy()
    y = df[target_col].copy()
    groups = df[group_col].copy()

    all_predictions = []
    group_metrics = []

    unique_groups = sorted(groups.unique())

    for group in unique_groups:
        train_mask = groups != group
        test_mask = groups == group

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]

        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)
        y_pred = fitted_model.predict(X_test)

        for idx, obs, pred in zip(y_test.index, y_test.values, y_pred):
            all_predictions.append({
                "Index": idx,
                "Held_out_formulation": group,
                "Observed": obs,
                "Predicted": pred,
                "Error": pred - obs,
                "Abs_error": abs(pred - obs)
            })

        if len(y_test) > 1:
            group_r2 = r2_score(y_test, y_pred)
            group_spearman = safe_spearman(y_test, y_pred)
        else:
            group_r2 = np.nan
            group_spearman = np.nan

        group_metrics.append({
            "Held_out_formulation": group,
            "N_test": len(y_test),
            "R2": group_r2,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
            "Spearman": group_spearman
        })

    predictions_df = pd.DataFrame(all_predictions)
    group_metrics_df = pd.DataFrame(group_metrics)

    pooled_summary = {
        "LOFO_R2": r2_score(predictions_df["Observed"], predictions_df["Predicted"]),
        "LOFO_MAE": mean_absolute_error(predictions_df["Observed"], predictions_df["Predicted"]),
        "LOFO_RMSE": mean_squared_error(predictions_df["Observed"], predictions_df["Predicted"]) ** 0.5,
        "LOFO_Spearman": safe_spearman(predictions_df["Observed"], predictions_df["Predicted"]),
        "LOFO_group_MAE_mean": group_metrics_df["MAE"].mean(),
        "LOFO_group_MAE_std": group_metrics_df["MAE"].std(),
        "LOFO_group_RMSE_mean": group_metrics_df["RMSE"].mean(),
        "LOFO_group_RMSE_std": group_metrics_df["RMSE"].std()
    }

    return pooled_summary, predictions_df, group_metrics_df

lofo_summary_rows = []
lofo_predictions_all = []
lofo_group_metrics_all = []

for target_name, config in target_configs.items():
    df_model = modeling_tables[target_name].copy()
    target_col = config["target"]
    allowed_feature_sets = config["allowed_feature_sets"]

    for feature_set_name in allowed_feature_sets:
        features = feature_sets[feature_set_name]
        features = [col for col in features if col in df_model.columns]

        if df_model[features].isna().sum().sum() > 0:
            print(f"Skipping {target_name} - {feature_set_name} due to missing feature values.")
            continue

        models = get_models()

        for model_name, model in models.items():
            print(f"LOFO running: Target={target_name}, FeatureSet={feature_set_name}, Model={model_name}")

            pooled_summary, predictions_df, group_metrics_df = evaluate_model_lofo(
                df=df_model,
                features=features,
                target_col=target_col,
                model=model,
                group_col="Formulation_label"
            )

            summary_row = {
                "Target": target_name,
                "Target_column": target_col,
                "Feature_set": feature_set_name,
                "Model": model_name,
                "N": len(df_model),
                "N_features": len(features),
                "N_formulations": df_model["Formulation_label"].nunique(),
                **pooled_summary
            }

            lofo_summary_rows.append(summary_row)

            predictions_df["Target"] = target_name
            predictions_df["Target_column"] = target_col
            predictions_df["Feature_set"] = feature_set_name
            predictions_df["Model"] = model_name
            lofo_predictions_all.append(predictions_df)

            group_metrics_df["Target"] = target_name
            group_metrics_df["Target_column"] = target_col
            group_metrics_df["Feature_set"] = feature_set_name
            group_metrics_df["Model"] = model_name
            lofo_group_metrics_all.append(group_metrics_df)

lofo_summary_df = pd.DataFrame(lofo_summary_rows)
lofo_predictions_df = pd.concat(lofo_predictions_all, ignore_index=True)
lofo_group_metrics_df = pd.concat(lofo_group_metrics_all, ignore_index=True)

lofo_summary_path = TABLES_DIR / "ml_model_performance_leave_one_formulation_out.csv"
lofo_summary_excel_path = TABLES_DIR / "ml_model_performance_leave_one_formulation_out.xlsx"
lofo_predictions_path = TABLES_DIR / "lofo_predictions_all.csv"
lofo_group_metrics_path = TABLES_DIR / "lofo_group_metrics_all.csv"

lofo_summary_df.to_csv(lofo_summary_path, index=False)
lofo_summary_df.to_excel(lofo_summary_excel_path, index=False)
lofo_predictions_df.to_csv(lofo_predictions_path, index=False)
lofo_group_metrics_df.to_csv(lofo_group_metrics_path, index=False)

print("\nLOFO validation saved:")
print(lofo_summary_path)
print(lofo_summary_excel_path)
print(lofo_predictions_path)
print(lofo_group_metrics_path)

print("\nTop LOFO models by target using LOFO_RMSE:")
for target_name in ["Pr", "SR", "Qm", "AF"]:
    print("\n" + "=" * 80)
    print(f"Target: {target_name}")
    print("=" * 80)

    top_lofo = (
        lofo_summary_df[lofo_summary_df["Target"] == target_name]
        .sort_values(["LOFO_RMSE", "LOFO_MAE"], ascending=True)
        .reset_index(drop=True)
    )

    display(top_lofo[[
        "Target",
        "Feature_set",
        "Model",
        "N",
        "N_features",
        "N_formulations",
        "LOFO_R2",
        "LOFO_MAE",
        "LOFO_RMSE",
        "LOFO_Spearman"
    ]].head(10))


# %% Cell 16

# Cell 15: Compare repeated K-fold and LOFO results and select final models for XAI

# Select best repeated K-fold model for each target based on RMSE
best_repeated_rows = []

for target_name in ["Pr", "SR", "Qm", "AF"]:
    best_row = (
        performance_df[performance_df["Target"] == target_name]
        .sort_values(["RMSE_mean", "MAE_mean"], ascending=True)
        .iloc[0]
        .to_dict()
    )
    best_repeated_rows.append(best_row)

best_repeated_df = pd.DataFrame(best_repeated_rows)

# Select best LOFO model for each target based on LOFO_RMSE
best_lofo_rows = []

for target_name in ["Pr", "SR", "Qm", "AF"]:
    best_row = (
        lofo_summary_df[lofo_summary_df["Target"] == target_name]
        .sort_values(["LOFO_RMSE", "LOFO_MAE"], ascending=True)
        .iloc[0]
        .to_dict()
    )
    best_lofo_rows.append(best_row)

best_lofo_df = pd.DataFrame(best_lofo_rows)

# Merge best repeated K-fold and best LOFO results
model_selection_df = best_repeated_df[[
    "Target",
    "Feature_set",
    "Model",
    "N",
    "N_features",
    "R2_mean",
    "MAE_mean",
    "RMSE_mean",
    "Spearman_mean"
]].rename(columns={
    "Feature_set": "Best_repeated_feature_set",
    "Model": "Best_repeated_model",
    "N_features": "Best_repeated_N_features",
    "R2_mean": "Repeated_R2_mean",
    "MAE_mean": "Repeated_MAE_mean",
    "RMSE_mean": "Repeated_RMSE_mean",
    "Spearman_mean": "Repeated_Spearman_mean"
}).merge(
    best_lofo_df[[
        "Target",
        "Feature_set",
        "Model",
        "N_features",
        "N_formulations",
        "LOFO_R2",
        "LOFO_MAE",
        "LOFO_RMSE",
        "LOFO_Spearman"
    ]].rename(columns={
        "Feature_set": "Best_LOFO_feature_set",
        "Model": "Best_LOFO_model",
        "N_features": "Best_LOFO_N_features"
    }),
    on="Target",
    how="left"
)

# Rule-based final model selection for downstream XAI
# Preference:
# 1. If LOFO is strong and stable, use LOFO-best model.
# 2. If LOFO collapses but repeated K-fold is strong, keep repeated-best as in-distribution model and flag limited extrapolation.
# 3. Prefer interpretable and SHAP-compatible models when performance is similar.

final_model_rows = []

for _, row in model_selection_df.iterrows():
    target_name = row["Target"]

    if target_name == "Pr":
        final_feature_set = "core"
        final_model = "LinearRegression"
        rationale = "LOFO-best model; simple composition-process model generalized better than higher-dimensional models."
        generalization_note = "Moderate LOFO performance; nonlinear models showed limited formulation-level generalization."

    elif target_name == "SR":
        final_feature_set = "rheology_enhanced"
        final_model = "SVR_RBF"
        rationale = "LOFO-best model; rheology-enhanced SVR gave the lowest formulation-level prediction error."
        generalization_note = "Moderate generalization; rheology descriptors improved LOFO performance."

    elif target_name == "Qm":
        final_feature_set = "engineered"
        final_model = "SVR_RBF"
        rationale = "Repeated K-fold-best model; LOFO performance was weak, so Qm is treated as reliable mainly for interpolation within the sampled formulation space."
        generalization_note = "Limited LOFO generalization; interpret extrapolation to unseen formulations cautiously."

    elif target_name == "AF":
        final_feature_set = "angle_engineered"
        final_model = "XGBoost"
        rationale = "LOFO-best model; angle-engineered XGBoost provided strong formulation-level generalization and is suitable for SHAP analysis."
        generalization_note = "Strong LOFO generalization; angle terms dominate AF prediction."

    final_model_rows.append({
        "Target": target_name,
        "Final_feature_set": final_feature_set,
        "Final_model": final_model,
        "Selection_rationale": rationale,
        "Generalization_note": generalization_note
    })

final_model_selection_df = pd.DataFrame(final_model_rows)

model_selection_final_df = model_selection_df.merge(
    final_model_selection_df,
    on="Target",
    how="left"
)

# Save model selection tables
model_selection_path = TABLES_DIR / "model_selection_summary.csv"
model_selection_excel_path = TABLES_DIR / "model_selection_summary.xlsx"

model_selection_final_df.to_csv(model_selection_path, index=False)
model_selection_final_df.to_excel(model_selection_excel_path, index=False)

print("Model selection summary saved:")
print(model_selection_path)
print(model_selection_excel_path)

print("\nModel selection summary:")
display(model_selection_final_df)

print("\nFinal models selected for XAI:")
display(final_model_selection_df)


# %% Cell 17

# Cell 16: Train final selected models on the full target-specific datasets

import joblib
from sklearn.base import clone

MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def get_final_model_instance(model_name):
    models = get_models()

    if model_name not in models:
        raise ValueError(f"Model not found: {model_name}")

    return clone(models[model_name])

final_trained_models = {}
final_training_rows = []
final_prediction_rows = []

for _, row in final_model_selection_df.iterrows():
    target_name = row["Target"]
    feature_set_name = row["Final_feature_set"]
    model_name = row["Final_model"]

    df_model = modeling_tables[target_name].copy()
    target_col = target_configs[target_name]["target"]
    features = feature_sets[feature_set_name]
    features = [col for col in features if col in df_model.columns]

    X = df_model[features].copy()
    y = df_model[target_col].copy()

    if X.isna().sum().sum() > 0:
        raise ValueError(f"Missing feature values found for {target_name}")

    model = get_final_model_instance(model_name)
    model.fit(X, y)

    y_pred = model.predict(X)

    train_r2 = r2_score(y, y_pred)
    train_mae = mean_absolute_error(y, y_pred)
    train_rmse = mean_squared_error(y, y_pred) ** 0.5
    train_spearman = safe_spearman(y, y_pred)

    model_file = MODELS_DIR / f"final_model_{target_name}_{model_name}_{feature_set_name}.joblib"
    joblib.dump(model, model_file)

    final_trained_models[target_name] = {
        "model": model,
        "model_name": model_name,
        "feature_set": feature_set_name,
        "features": features,
        "target_col": target_col,
        "data": df_model,
        "X": X,
        "y": y,
        "y_pred": y_pred,
        "model_file": str(model_file)
    }

    final_training_rows.append({
        "Target": target_name,
        "Target_column": target_col,
        "Final_model": model_name,
        "Final_feature_set": feature_set_name,
        "N": len(df_model),
        "N_features": len(features),
        "Full_fit_R2": train_r2,
        "Full_fit_MAE": train_mae,
        "Full_fit_RMSE": train_rmse,
        "Full_fit_Spearman": train_spearman,
        "Model_file": str(model_file)
    })

    temp_pred_df = pd.DataFrame({
        "Target": target_name,
        "Observed": y.values,
        "Predicted": y_pred,
        "Residual": y_pred - y.values,
        "Abs_error": np.abs(y_pred - y.values)
    })

    if "Formulation_label" in df_model.columns:
        temp_pred_df["Formulation_label"] = df_model["Formulation_label"].values

    if "Pressure_kPa" in df_model.columns:
        temp_pred_df["Pressure_kPa"] = df_model["Pressure_kPa"].values

    if "Angle_deg" in df_model.columns:
        temp_pred_df["Angle_deg"] = df_model["Angle_deg"].values

    final_prediction_rows.append(temp_pred_df)

    plt.figure(figsize=(5, 5))
    plt.scatter(y, y_pred)

    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel(f"Observed {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"Final full-data fit: {target_name}")
    plt.tight_layout()

    fig_path = FIGURES_DIR / f"final_observed_vs_predicted_{target_name}.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print("=" * 80)
    print(f"Final model trained for {target_name}")
    print("=" * 80)
    print("Model:", model_name)
    print("Feature set:", feature_set_name)
    print("Number of rows:", len(df_model))
    print("Number of features:", len(features))
    print("Full-fit R2:", train_r2)
    print("Full-fit MAE:", train_mae)
    print("Full-fit RMSE:", train_rmse)
    print("Full-fit Spearman:", train_spearman)
    print("Saved model:", model_file)

final_training_summary_df = pd.DataFrame(final_training_rows)
final_predictions_df = pd.concat(final_prediction_rows, ignore_index=True)

final_training_summary_path = TABLES_DIR / "final_model_full_fit_summary.csv"
final_training_summary_excel_path = TABLES_DIR / "final_model_full_fit_summary.xlsx"
final_predictions_path = TABLES_DIR / "final_model_full_fit_predictions.csv"

final_training_summary_df.to_csv(final_training_summary_path, index=False)
final_training_summary_df.to_excel(final_training_summary_excel_path, index=False)
final_predictions_df.to_csv(final_predictions_path, index=False)

final_model_metadata = {
    target: {
        "model_name": info["model_name"],
        "feature_set": info["feature_set"],
        "features": info["features"],
        "target_col": info["target_col"],
        "model_file": info["model_file"]
    }
    for target, info in final_trained_models.items()
}

final_model_metadata_path = MODELS_DIR / "final_model_metadata.json"

with open(final_model_metadata_path, "w", encoding="utf-8") as f:
    json.dump(final_model_metadata, f, indent=4)

print("\nFinal model outputs saved:")
print(final_training_summary_path)
print(final_training_summary_excel_path)
print(final_predictions_path)
print(final_model_metadata_path)

print("\nFinal full-fit summary:")
display(final_training_summary_df)

print("\nSaved final model figures:")
for file in sorted(FIGURES_DIR.glob("final_observed_vs_predicted_*.png")):
    print("-", file.name)


# %% Cell 18

# Cell 17: XAI preparation and permutation importance for final models

from sklearn.inspection import permutation_importance

XAI_DIR = RESULTS_DIR / "xai"
XAI_DIR.mkdir(parents=True, exist_ok=True)

permutation_rows = []
xai_prepared_objects = {}

for target_name, info in final_trained_models.items():
    model = info["model"]
    model_name = info["model_name"]
    feature_set_name = info["feature_set"]
    features = info["features"]
    target_col = info["target_col"]
    X = info["X"].copy()
    y = info["y"].copy()

    print("=" * 80)
    print(f"XAI preparation for {target_name}")
    print("=" * 80)
    print("Model:", model_name)
    print("Feature set:", feature_set_name)
    print("Number of rows:", X.shape[0])
    print("Number of features:", X.shape[1])

    # Store clean XAI objects
    xai_prepared_objects[target_name] = {
        "model": model,
        "model_name": model_name,
        "feature_set": feature_set_name,
        "features": features,
        "target_col": target_col,
        "X": X,
        "y": y
    }

    # Permutation importance based on RMSE increase
    perm_result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring="neg_root_mean_squared_error",
        n_repeats=50,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    target_perm_df = pd.DataFrame({
        "Target": target_name,
        "Model": model_name,
        "Feature_set": feature_set_name,
        "Feature": features,
        "Importance_mean_RMSE_increase": perm_result.importances_mean,
        "Importance_std": perm_result.importances_std
    })

    target_perm_df["Abs_importance_mean"] = target_perm_df["Importance_mean_RMSE_increase"].abs()
    target_perm_df = target_perm_df.sort_values(
        "Importance_mean_RMSE_increase",
        ascending=False
    ).reset_index(drop=True)

    permutation_rows.append(target_perm_df)

    print("\nTop permutation importance features:")
    display(target_perm_df.head(12))

    # Plot top features
    plot_df = target_perm_df.head(12).sort_values(
        "Importance_mean_RMSE_increase",
        ascending=True
    )

    plt.figure(figsize=(7, 5))
    plt.barh(plot_df["Feature"], plot_df["Importance_mean_RMSE_increase"])
    plt.xlabel("Permutation importance: RMSE increase")
    plt.ylabel("Feature")
    plt.title(f"Permutation importance for {target_name}")
    plt.tight_layout()

    fig_path = XAI_DIR / f"permutation_importance_{target_name}.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()

permutation_importance_df = pd.concat(permutation_rows, ignore_index=True)

permutation_importance_path = TABLES_DIR / "xai_permutation_importance.csv"
permutation_importance_excel_path = TABLES_DIR / "xai_permutation_importance.xlsx"

permutation_importance_df.to_csv(permutation_importance_path, index=False)
permutation_importance_df.to_excel(permutation_importance_excel_path, index=False)

print("\nPermutation importance saved:")
print(permutation_importance_path)
print(permutation_importance_excel_path)

print("\nSaved XAI figures:")
for file in sorted(XAI_DIR.glob("permutation_importance_*.png")):
    print("-", file.name)


# %% Cell 19

# Cell 18: SHAP analysis for final selected models

import shap

shap_summary_rows = []
shap_values_objects = {}

for target_name, info in final_trained_models.items():
    model = info["model"]
    model_name = info["model_name"]
    feature_set_name = info["feature_set"]
    features = info["features"]
    X = info["X"].copy()
    y = info["y"].copy()

    print("=" * 80)
    print(f"SHAP analysis for {target_name}")
    print("=" * 80)
    print("Model:", model_name)
    print("Feature set:", feature_set_name)
    print("Rows:", X.shape[0])
    print("Features:", X.shape[1])

    # Use a model-agnostic permutation SHAP explainer for consistent behavior across models
    masker = shap.maskers.Independent(X)
    explainer = shap.Explainer(
        model.predict,
        masker,
        algorithm="permutation"
    )

    max_evals = 2 * X.shape[1] + 1

    shap_values = explainer(
        X,
        max_evals=max_evals
    )

    shap_values_objects[target_name] = shap_values

    shap_array = shap_values.values

    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    mean_shap = shap_array.mean(axis=0)
    std_abs_shap = np.abs(shap_array).std(axis=0)

    target_shap_df = pd.DataFrame({
        "Target": target_name,
        "Model": model_name,
        "Feature_set": feature_set_name,
        "Feature": features,
        "Mean_abs_SHAP": mean_abs_shap,
        "Mean_SHAP": mean_shap,
        "Std_abs_SHAP": std_abs_shap
    })

    target_shap_df = target_shap_df.sort_values(
        "Mean_abs_SHAP",
        ascending=False
    ).reset_index(drop=True)

    shap_summary_rows.append(target_shap_df)

    target_shap_path = TABLES_DIR / f"shap_summary_{target_name}.csv"
    target_shap_excel_path = TABLES_DIR / f"shap_summary_{target_name}.xlsx"

    target_shap_df.to_csv(target_shap_path, index=False)
    target_shap_df.to_excel(target_shap_excel_path, index=False)

    print("\nTop SHAP features:")
    display(target_shap_df.head(12))

    # Custom SHAP bar plot
    plot_df = target_shap_df.head(12).sort_values("Mean_abs_SHAP", ascending=True)

    plt.figure(figsize=(7, 5))
    plt.barh(plot_df["Feature"], plot_df["Mean_abs_SHAP"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title(f"SHAP feature importance for {target_name}")
    plt.tight_layout()

    fig_path = XAI_DIR / f"shap_bar_{target_name}.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()

    # SHAP beeswarm plot
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title(f"SHAP beeswarm for {target_name}")
    plt.tight_layout()

    beeswarm_path = XAI_DIR / f"shap_beeswarm_{target_name}.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.show()

shap_summary_df = pd.concat(shap_summary_rows, ignore_index=True)

shap_summary_path = TABLES_DIR / "xai_shap_summary_all_targets.csv"
shap_summary_excel_path = TABLES_DIR / "xai_shap_summary_all_targets.xlsx"

shap_summary_df.to_csv(shap_summary_path, index=False)
shap_summary_df.to_excel(shap_summary_excel_path, index=False)

print("\nSHAP summary saved:")
print(shap_summary_path)
print(shap_summary_excel_path)

print("\nSaved SHAP figures:")
for file in sorted(XAI_DIR.glob("shap_*.png")):
    print("-", file.name)


# %% Cell 20

# Cell 19: Compare permutation importance and SHAP feature importance

from scipy.stats import spearmanr

xai_comparison_rows = []
rank_agreement_rows = []

for target_name in ["Pr", "SR", "Qm", "AF"]:
    perm_df = permutation_importance_df[
        permutation_importance_df["Target"] == target_name
    ].copy()

    shap_df = shap_summary_df[
        shap_summary_df["Target"] == target_name
    ].copy()

    merged = perm_df.merge(
        shap_df[[
            "Target",
            "Model",
            "Feature_set",
            "Feature",
            "Mean_abs_SHAP",
            "Mean_SHAP",
            "Std_abs_SHAP"
        ]],
        on=["Target", "Model", "Feature_set", "Feature"],
        how="inner"
    )

    merged["Permutation_positive"] = merged["Importance_mean_RMSE_increase"].clip(lower=0)

    if merged["Permutation_positive"].max() > 0:
        merged["Permutation_norm"] = merged["Permutation_positive"] / merged["Permutation_positive"].max()
    else:
        merged["Permutation_norm"] = 0

    if merged["Mean_abs_SHAP"].max() > 0:
        merged["SHAP_norm"] = merged["Mean_abs_SHAP"] / merged["Mean_abs_SHAP"].max()
    else:
        merged["SHAP_norm"] = 0

    merged["Permutation_rank"] = merged["Permutation_positive"].rank(
        ascending=False,
        method="min"
    )
    merged["SHAP_rank"] = merged["Mean_abs_SHAP"].rank(
        ascending=False,
        method="min"
    )

    merged["Rank_difference"] = (
        merged["Permutation_rank"] - merged["SHAP_rank"]
    ).abs()

    merged["Consensus_score"] = (
        merged["Permutation_norm"] + merged["SHAP_norm"]
    ) / 2

    merged = merged.sort_values(
        ["Consensus_score", "Permutation_norm", "SHAP_norm"],
        ascending=False
    ).reset_index(drop=True)

    xai_comparison_rows.append(merged)

    if merged.shape[0] > 2:
        rank_corr = spearmanr(
            merged["Permutation_rank"],
            merged["SHAP_rank"]
        ).correlation
    else:
        rank_corr = np.nan

    rank_agreement_rows.append({
        "Target": target_name,
        "N_features": merged.shape[0],
        "Rank_agreement_Spearman": rank_corr,
        "Top_consensus_feature": merged.loc[0, "Feature"],
        "Top_consensus_score": merged.loc[0, "Consensus_score"]
    })

    print("\n" + "=" * 80)
    print(f"XAI consensus for {target_name}")
    print("=" * 80)

    display(merged[[
        "Target",
        "Feature",
        "Importance_mean_RMSE_increase",
        "Mean_abs_SHAP",
        "Permutation_rank",
        "SHAP_rank",
        "Rank_difference",
        "Consensus_score"
    ]].head(12))

    plot_df = merged.head(12).sort_values("Consensus_score", ascending=True)

    plt.figure(figsize=(7, 5))
    plt.barh(plot_df["Feature"], plot_df["Consensus_score"])
    plt.xlabel("Consensus importance score")
    plt.ylabel("Feature")
    plt.title(f"Permutation-SHAP consensus for {target_name}")
    plt.tight_layout()

    fig_path = XAI_DIR / f"xai_consensus_{target_name}.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()

xai_comparison_df = pd.concat(xai_comparison_rows, ignore_index=True)
rank_agreement_df = pd.DataFrame(rank_agreement_rows)

xai_comparison_path = TABLES_DIR / "xai_permutation_shap_comparison.csv"
xai_comparison_excel_path = TABLES_DIR / "xai_permutation_shap_comparison.xlsx"
rank_agreement_path = TABLES_DIR / "xai_rank_agreement_summary.csv"
rank_agreement_excel_path = TABLES_DIR / "xai_rank_agreement_summary.xlsx"

xai_comparison_df.to_csv(xai_comparison_path, index=False)
xai_comparison_df.to_excel(xai_comparison_excel_path, index=False)

rank_agreement_df.to_csv(rank_agreement_path, index=False)
rank_agreement_df.to_excel(rank_agreement_excel_path, index=False)

print("\nXAI comparison saved:")
print(xai_comparison_path)
print(xai_comparison_excel_path)
print(rank_agreement_path)
print(rank_agreement_excel_path)

print("\nRank agreement summary:")
display(rank_agreement_df)

print("\nSaved XAI consensus figures:")
for file in sorted(XAI_DIR.glob("xai_consensus_*.png")):
    print("-", file.name)


# %% Cell 21

# Cell 20: Extract XAI-guided design rules from consensus features

design_rule_rows = []
binned_response_rows = []

# Use top consensus features for rule extraction
top_n_rules = 8

def classify_effect_direction(spearman_value, weak_threshold=0.20):
    if pd.isna(spearman_value):
        return "Not determined"
    elif spearman_value > weak_threshold:
        return "Higher feature values are associated with higher target values"
    elif spearman_value < -weak_threshold:
        return "Higher feature values are associated with lower target values"
    else:
        return "Weak or non-monotonic association"

def classify_rule_strength(consensus_score):
    if consensus_score >= 0.70:
        return "Strong"
    elif consensus_score >= 0.30:
        return "Moderate"
    else:
        return "Weak"

for target_name in ["Pr", "SR", "Qm", "AF"]:
    target_col = target_configs[target_name]["target"]
    df_model = modeling_tables[target_name].copy()

    target_xai = (
        xai_comparison_df[xai_comparison_df["Target"] == target_name]
        .sort_values("Consensus_score", ascending=False)
        .head(top_n_rules)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 80)
    print(f"Design-rule extraction for {target_name}")
    print("=" * 80)

    for _, xai_row in target_xai.iterrows():
        feature = xai_row["Feature"]

        if feature not in df_model.columns:
            continue

        temp = df_model[[feature, target_col]].dropna().copy()

        if temp[feature].nunique() <= 1:
            continue

        spearman_value = temp[feature].corr(temp[target_col], method="spearman")
        pearson_value = temp[feature].corr(temp[target_col], method="pearson")

        effect_direction = classify_effect_direction(spearman_value)
        rule_strength = classify_rule_strength(xai_row["Consensus_score"])

        # Create tertile bins for interpretable response summaries
        try:
            temp["Feature_bin"] = pd.qcut(
                temp[feature],
                q=3,
                labels=["Low", "Mid", "High"],
                duplicates="drop"
            )
        except Exception:
            temp["Feature_bin"] = pd.cut(
                temp[feature],
                bins=3,
                labels=["Low", "Mid", "High"]
            )

        bin_summary = (
            temp
            .groupby("Feature_bin", observed=True)
            .agg(
                Feature_min=(feature, "min"),
                Feature_max=(feature, "max"),
                Target_mean=(target_col, "mean"),
                Target_std=(target_col, "std"),
                N=(target_col, "count")
            )
            .reset_index()
        )

        for _, bin_row in bin_summary.iterrows():
            binned_response_rows.append({
                "Target": target_name,
                "Target_column": target_col,
                "Feature": feature,
                "Feature_bin": bin_row["Feature_bin"],
                "Feature_min": bin_row["Feature_min"],
                "Feature_max": bin_row["Feature_max"],
                "Target_mean": bin_row["Target_mean"],
                "Target_std": bin_row["Target_std"],
                "N": bin_row["N"]
            })

        low_mean = bin_summary.loc[
            bin_summary["Feature_bin"].astype(str) == "Low",
            "Target_mean"
        ]
        high_mean = bin_summary.loc[
            bin_summary["Feature_bin"].astype(str) == "High",
            "Target_mean"
        ]

        if len(low_mean) > 0 and len(high_mean) > 0:
            low_mean_value = float(low_mean.iloc[0])
            high_mean_value = float(high_mean.iloc[0])
            high_minus_low = high_mean_value - low_mean_value
        else:
            low_mean_value = np.nan
            high_mean_value = np.nan
            high_minus_low = np.nan

        rule_text = (
            f"For {target_name}, {feature} showed a {rule_strength.lower()} XAI consensus "
            f"score ({xai_row['Consensus_score']:.3f}). "
            f"{effect_direction}. "
            f"The mean target changed from {low_mean_value:.3f} in the low-feature range "
            f"to {high_mean_value:.3f} in the high-feature range."
        )

        design_rule_rows.append({
            "Target": target_name,
            "Target_column": target_col,
            "Feature": feature,
            "Consensus_score": xai_row["Consensus_score"],
            "Permutation_rank": xai_row["Permutation_rank"],
            "SHAP_rank": xai_row["SHAP_rank"],
            "Rank_difference": xai_row["Rank_difference"],
            "Pearson_r": pearson_value,
            "Spearman_r": spearman_value,
            "Effect_direction": effect_direction,
            "Rule_strength": rule_strength,
            "Low_bin_target_mean": low_mean_value,
            "High_bin_target_mean": high_mean_value,
            "High_minus_low_target_mean": high_minus_low,
            "Rule_text": rule_text
        })

    target_rules_preview = pd.DataFrame([
        row for row in design_rule_rows if row["Target"] == target_name
    ])

    display(target_rules_preview[[
        "Target",
        "Feature",
        "Consensus_score",
        "Spearman_r",
        "Effect_direction",
        "Rule_strength",
        "Low_bin_target_mean",
        "High_bin_target_mean",
        "High_minus_low_target_mean"
    ]].head(top_n_rules))

design_rules_df = pd.DataFrame(design_rule_rows)
binned_response_df = pd.DataFrame(binned_response_rows)

design_rules_path = TABLES_DIR / "xai_design_rules.csv"
design_rules_excel_path = TABLES_DIR / "xai_design_rules.xlsx"

binned_response_path = TABLES_DIR / "xai_feature_bin_response_summary.csv"
binned_response_excel_path = TABLES_DIR / "xai_feature_bin_response_summary.xlsx"

design_rules_df.to_csv(design_rules_path, index=False)
design_rules_df.to_excel(design_rules_excel_path, index=False)

binned_response_df.to_csv(binned_response_path, index=False)
binned_response_df.to_excel(binned_response_excel_path, index=False)

print("\nDesign-rule tables saved:")
print(design_rules_path)
print(design_rules_excel_path)
print(binned_response_path)
print(binned_response_excel_path)

print("\nTop design rules by target:")
for target_name in ["Pr", "SR", "Qm", "AF"]:
    print("\n" + "=" * 80)
    print(f"Target: {target_name}")
    print("=" * 80)

    display(
        design_rules_df[design_rules_df["Target"] == target_name]
        .sort_values("Consensus_score", ascending=False)
        [[
            "Target",
            "Feature",
            "Consensus_score",
            "Spearman_r",
            "Effect_direction",
            "Rule_strength",
            "Rule_text"
        ]]
        .head(5)
    )


# %% Cell 22

# Cell 21-fix: Redefine ML feature function with corrected feature name

def add_ml_features(df, include_angle=False):
    df = df.copy()

    # Core composition features
    df["TP"] = df["Total_polymer_mg_ml"]
    df["TP_pct_wv"] = df["Total_polymer_pct_wv"]
    df["HF"] = df["HA_fraction"]
    df["Alg_fraction_feature"] = df["Alg_fraction"]
    df["Alg_HA_ratio_safe"] = df["Alg_HA_ratio"]

    # Process-normalized features
    if "Pressure_kPa" in df.columns:
        df["P_over_TP"] = df["Pressure_kPa"] / df["TP"]
        df["P_over_Alg"] = df["Pressure_kPa"] / df["Alg_mg_ml"]
        df["P_over_HA"] = df["Pressure_kPa"] / df["HA_mg_ml"]

        # Polymer-pressure interaction features
        df["Alg_x_P"] = df["Alg_mg_ml"] * df["Pressure_kPa"]
        df["HA_x_P"] = df["HA_mg_ml"] * df["Pressure_kPa"]
        df["TP_x_P"] = df["TP"] * df["Pressure_kPa"]
        df["HF_x_P"] = df["HF"] * df["Pressure_kPa"]

    # Polymer-polymer interaction features
    df["Alg_x_HA"] = df["Alg_mg_ml"] * df["HA_mg_ml"]
    df["Alg_fraction_x_HA_fraction"] = df["Alg_fraction"] * df["HA_fraction"]

    # Angle-related features for angular fidelity model
    if include_angle and "Angle_deg" in df.columns:
        df["Angle_rad"] = np.deg2rad(df["Angle_deg"])
        df["sin_angle"] = np.sin(df["Angle_rad"])
        df["cos_angle"] = np.cos(df["Angle_rad"])

        if "Pressure_kPa" in df.columns:
            df["P_x_Angle"] = df["Pressure_kPa"] * df["Angle_deg"]
            df["Alg_x_Angle"] = df["Alg_mg_ml"] * df["Angle_deg"]
            df["HA_x_Angle"] = df["HA_mg_ml"] * df["Angle_deg"]
            df["TP_x_Angle"] = df["TP"] * df["Angle_deg"]

    return df

print("add_ml_features function has been corrected.")
print("Now rerun Cell 21 from the beginning.")


# %% Cell 23

# Cell 21: Build an XAI-guided candidate screening within the experimentally supported design space

CANDIDATE_SCREENING_DIR = RESULTS_DIR / "candidate_screening"
CANDIDATE_SCREENING_DIR.mkdir(parents=True, exist_ok=True)

# Use only experimentally observed formulation space
supported_formulations = (
    master_printability[["Alg_mg_ml", "HA_mg_ml", "Formulation_label"]]
    .drop_duplicates()
    .sort_values(["Alg_mg_ml", "HA_mg_ml"])
    .reset_index(drop=True)
)

# Use observed pressure values to avoid unsupported interpolation artifacts
supported_pressures = sorted(
    set(master_printability["Pressure_kPa"].dropna().unique()).union(
        set(master_angle["Pressure_kPa"].dropna().unique())
    )
)

# Pressure ranges observed for each formulation in printability data
printability_pressure_ranges = (
    master_printability
    .groupby(["Alg_mg_ml", "HA_mg_ml"], as_index=False)
    .agg(
        Pressure_min=("Pressure_kPa", "min"),
        Pressure_max=("Pressure_kPa", "max")
    )
)

# Pressure ranges observed for each formulation in angle-fidelity data
angle_pressure_ranges = (
    master_angle
    .groupby(["Alg_mg_ml", "HA_mg_ml"], as_index=False)
    .agg(
        Pressure_min=("Pressure_kPa", "min"),
        Pressure_max=("Pressure_kPa", "max")
    )
)

# -------------------------------------------------------------------------
# Part 1: Printability candidate screening for Pr, SR, and Qm
# -------------------------------------------------------------------------

printability_grid_rows = []

for _, form_row in supported_formulations.iterrows():
    alg = form_row["Alg_mg_ml"]
    ha = form_row["HA_mg_ml"]

    range_row = printability_pressure_ranges[
        (printability_pressure_ranges["Alg_mg_ml"] == alg) &
        (printability_pressure_ranges["HA_mg_ml"] == ha)
    ]

    if range_row.empty:
        continue

    p_min = range_row["Pressure_min"].iloc[0]
    p_max = range_row["Pressure_max"].iloc[0]

    for pressure in supported_pressures:
        if pressure < p_min or pressure > p_max:
            continue

        printability_grid_rows.append({
            "Alg_mg_ml": alg,
            "HA_mg_ml": ha,
            "Pressure_kPa": pressure
        })

printability_grid = pd.DataFrame(printability_grid_rows)

printability_grid = add_formulation_features(printability_grid)
printability_grid = add_ml_features(printability_grid, include_angle=False)

# Add rheology descriptors for SR model
printability_grid = printability_grid.merge(
    rheology_for_merge,
    on=["Alg_mg_ml", "HA_mg_ml"],
    how="left",
    validate="many_to_one"
)

candidate_screening_printability = printability_grid.copy()

for target_name in ["Pr", "SR", "Qm"]:
    info = final_trained_models[target_name]
    model = info["model"]
    features = info["features"]

    missing_features = [col for col in features if col not in candidate_screening_printability.columns]
    if missing_features:
        raise ValueError(f"Missing features for {target_name}: {missing_features}")

    candidate_screening_printability[f"Predicted_{target_name}"] = model.predict(
        candidate_screening_printability[features]
    )

# Add simple target-oriented descriptors
candidate_screening_printability["Pr_error_from_ideal_1"] = (
    candidate_screening_printability["Predicted_Pr"] - 1.0
).abs()

candidate_screening_printability["Pr_within_0p95_1p05"] = (
    (candidate_screening_printability["Predicted_Pr"] >= 0.95) &
    (candidate_screening_printability["Predicted_Pr"] <= 1.05)
)

candidate_screening_printability["Pr_within_0p90_1p10"] = (
    (candidate_screening_printability["Predicted_Pr"] >= 0.90) &
    (candidate_screening_printability["Predicted_Pr"] <= 1.10)
)

# -------------------------------------------------------------------------
# Part 2: Angle-fidelity candidate screening for AF
# -------------------------------------------------------------------------

supported_angles = sorted(master_angle["Angle_deg"].dropna().unique())

angle_grid_rows = []

for _, form_row in supported_formulations.iterrows():
    alg = form_row["Alg_mg_ml"]
    ha = form_row["HA_mg_ml"]

    range_row = angle_pressure_ranges[
        (angle_pressure_ranges["Alg_mg_ml"] == alg) &
        (angle_pressure_ranges["HA_mg_ml"] == ha)
    ]

    if range_row.empty:
        continue

    p_min = range_row["Pressure_min"].iloc[0]
    p_max = range_row["Pressure_max"].iloc[0]

    for pressure in supported_pressures:
        if pressure < p_min or pressure > p_max:
            continue

        for angle in supported_angles:
            angle_grid_rows.append({
                "Alg_mg_ml": alg,
                "HA_mg_ml": ha,
                "Pressure_kPa": pressure,
                "Angle_deg": angle
            })

angle_grid = pd.DataFrame(angle_grid_rows)

angle_grid = add_formulation_features(angle_grid)
angle_grid = add_ml_features(angle_grid, include_angle=True)

candidate_screening_angle = angle_grid.copy()

af_info = final_trained_models["AF"]
af_model = af_info["model"]
af_features = af_info["features"]

missing_features = [col for col in af_features if col not in candidate_screening_angle.columns]
if missing_features:
    raise ValueError(f"Missing features for AF: {missing_features}")

candidate_screening_angle["Predicted_AF"] = af_model.predict(
    candidate_screening_angle[af_features]
)

candidate_screening_angle["AF_error_from_ideal_1"] = (
    candidate_screening_angle["Predicted_AF"] - 1.0
).abs()

candidate_screening_angle["AF_within_1p00_1p25"] = (
    (candidate_screening_angle["Predicted_AF"] >= 1.00) &
    (candidate_screening_angle["Predicted_AF"] <= 1.25)
)

# Summarize AF across angles for each formulation-pressure pair
af_pressure_summary = (
    candidate_screening_angle
    .groupby(["Alg_mg_ml", "HA_mg_ml", "Formulation_label", "Pressure_kPa"], as_index=False)
    .agg(
        AF_mean_pred=("Predicted_AF", "mean"),
        AF_max_pred=("Predicted_AF", "max"),
        AF_min_pred=("Predicted_AF", "min"),
        AF_mean_error_from_1=("AF_error_from_ideal_1", "mean"),
        AF_max_error_from_1=("AF_error_from_ideal_1", "max"),
        AF_fraction_within_1p00_1p25=("AF_within_1p00_1p25", "mean")
    )
)

# -------------------------------------------------------------------------
# Part 3: Combined candidate screening
# -------------------------------------------------------------------------

combined_candidate_screening = candidate_screening_printability.merge(
    af_pressure_summary,
    on=["Alg_mg_ml", "HA_mg_ml", "Formulation_label", "Pressure_kPa"],
    how="left"
)

combined_candidate_screening["Combined_Pr_AF_error"] = (
    combined_candidate_screening["Pr_error_from_ideal_1"] +
    combined_candidate_screening["AF_mean_error_from_1"]
)

combined_candidate_screening["Conservative_candidate"] = (
    combined_candidate_screening["Pr_within_0p90_1p10"] &
    (combined_candidate_screening["AF_fraction_within_1p00_1p25"] >= 0.50)
)

# Rank candidates
combined_candidate_screening = combined_candidate_screening.sort_values(
    ["Combined_Pr_AF_error", "Pr_error_from_ideal_1", "AF_mean_error_from_1"],
    ascending=True
).reset_index(drop=True)

# -------------------------------------------------------------------------
# Save outputs
# -------------------------------------------------------------------------

printability_candidate_screening_path = CANDIDATE_SCREENING_DIR / "candidate_screening_printability_predictions.csv"
angle_candidate_screening_path = CANDIDATE_SCREENING_DIR / "candidate_screening_angle_fidelity_predictions.csv"
af_summary_path = CANDIDATE_SCREENING_DIR / "candidate_screening_af_pressure_summary.csv"
combined_candidate_screening_path = CANDIDATE_SCREENING_DIR / "candidate_screening_combined_candidates.csv"

printability_candidate_screening_excel_path = CANDIDATE_SCREENING_DIR / "candidate_screening_printability_predictions.xlsx"
angle_candidate_screening_excel_path = CANDIDATE_SCREENING_DIR / "candidate_screening_angle_fidelity_predictions.xlsx"
af_summary_excel_path = CANDIDATE_SCREENING_DIR / "candidate_screening_af_pressure_summary.xlsx"
combined_candidate_screening_excel_path = CANDIDATE_SCREENING_DIR / "candidate_screening_combined_candidates.xlsx"

candidate_screening_printability.to_csv(printability_candidate_screening_path, index=False)
candidate_screening_angle.to_csv(angle_candidate_screening_path, index=False)
af_pressure_summary.to_csv(af_summary_path, index=False)
combined_candidate_screening.to_csv(combined_candidate_screening_path, index=False)

candidate_screening_printability.to_excel(printability_candidate_screening_excel_path, index=False)
candidate_screening_angle.to_excel(angle_candidate_screening_excel_path, index=False)
af_pressure_summary.to_excel(af_summary_excel_path, index=False)
combined_candidate_screening.to_excel(combined_candidate_screening_excel_path, index=False)

print("Candidate screening files saved:")
print(printability_candidate_screening_path)
print(angle_candidate_screening_path)
print(af_summary_path)
print(combined_candidate_screening_path)

print("\nCandidate screening shapes:")
print("Printability candidate screening:", candidate_screening_printability.shape)
print("Angle fidelity candidate screening:", candidate_screening_angle.shape)
print("AF pressure summary:", af_pressure_summary.shape)
print("Combined candidate screening:", combined_candidate_screening.shape)

print("\nTop combined candidates:")
display(combined_candidate_screening[[
    "Formulation_label",
    "Alg_mg_ml",
    "HA_mg_ml",
    "Pressure_kPa",
    "Predicted_Pr",
    "Pr_error_from_ideal_1",
    "Predicted_SR",
    "Predicted_Qm",
    "AF_mean_pred",
    "AF_max_pred",
    "AF_mean_error_from_1",
    "AF_fraction_within_1p00_1p25",
    "Combined_Pr_AF_error",
    "Conservative_candidate"
]].head(20))

print("\nConservative candidates only:")
display(combined_candidate_screening[
    combined_candidate_screening["Conservative_candidate"] == True
][[
    "Formulation_label",
    "Alg_mg_ml",
    "HA_mg_ml",
    "Pressure_kPa",
    "Predicted_Pr",
    "Predicted_SR",
    "Predicted_Qm",
    "AF_mean_pred",
    "AF_fraction_within_1p00_1p25",
    "Combined_Pr_AF_error"
]].head(20))

# -------------------------------------------------------------------------
# Simple candidate screening heatmap-like tables
# -------------------------------------------------------------------------

print("\nPredicted Pr candidate screening:")
display(
    candidate_screening_printability
    .pivot_table(
        index="Formulation_label",
        columns="Pressure_kPa",
        values="Predicted_Pr",
        aggfunc="mean"
    )
)

print("\nPredicted SR candidate screening:")
display(
    candidate_screening_printability
    .pivot_table(
        index="Formulation_label",
        columns="Pressure_kPa",
        values="Predicted_SR",
        aggfunc="mean"
    )
)

print("\nPredicted Qm candidate screening:")
display(
    candidate_screening_printability
    .pivot_table(
        index="Formulation_label",
        columns="Pressure_kPa",
        values="Predicted_Qm",
        aggfunc="mean"
    )
)

print("\nMean predicted AF candidate screening across angles:")
display(
    af_pressure_summary
    .pivot_table(
        index="Formulation_label",
        columns="Pressure_kPa",
        values="AF_mean_pred",
        aggfunc="mean"
    )
)


# %% Cell 24

# Cell 21b: Angle-specific candidate ranking for the candidate screening

ANGLE_SPECIFIC_DIR = CANDIDATE_SCREENING_DIR / "angle_specific_candidates"
ANGLE_SPECIFIC_DIR.mkdir(parents=True, exist_ok=True)

# Find the best angle for each formulation-pressure pair based on AF error from ideal value of 1
best_angle_per_condition = (
    candidate_screening_angle
    .sort_values("AF_error_from_ideal_1", ascending=True)
    .groupby(
        ["Alg_mg_ml", "HA_mg_ml", "Formulation_label", "Pressure_kPa"],
        as_index=False
    )
    .first()
)

best_angle_per_condition = best_angle_per_condition[[
    "Alg_mg_ml",
    "HA_mg_ml",
    "Formulation_label",
    "Pressure_kPa",
    "Angle_deg",
    "Predicted_AF",
    "AF_error_from_ideal_1",
    "AF_within_1p00_1p25"
]].rename(columns={
    "Angle_deg": "Best_angle_deg",
    "Predicted_AF": "Best_angle_predicted_AF",
    "AF_error_from_ideal_1": "Best_angle_AF_error_from_1",
    "AF_within_1p00_1p25": "Best_angle_AF_within_1p00_1p25"
})

# Merge printability predictions with the best angle-specific AF prediction
angle_specific_candidates = candidate_screening_printability.merge(
    best_angle_per_condition,
    on=["Alg_mg_ml", "HA_mg_ml", "Formulation_label", "Pressure_kPa"],
    how="left"
)

# Define practical candidate rules
angle_specific_candidates["Pr_good_0p90_1p10"] = (
    (angle_specific_candidates["Predicted_Pr"] >= 0.90) &
    (angle_specific_candidates["Predicted_Pr"] <= 1.10)
)

angle_specific_candidates["Pr_strict_0p95_1p05"] = (
    (angle_specific_candidates["Predicted_Pr"] >= 0.95) &
    (angle_specific_candidates["Predicted_Pr"] <= 1.05)
)

angle_specific_candidates["AF_good_best_angle"] = (
    (angle_specific_candidates["Best_angle_predicted_AF"] >= 1.00) &
    (angle_specific_candidates["Best_angle_predicted_AF"] <= 1.25)
)

# Combined score prioritizes Pr close to 1 and best-angle AF close to 1
angle_specific_candidates["Angle_specific_combined_error"] = (
    angle_specific_candidates["Pr_error_from_ideal_1"] +
    angle_specific_candidates["Best_angle_AF_error_from_1"]
)

# Recommended candidate: acceptable Pr and at least one angle with acceptable AF
angle_specific_candidates["Recommended_candidate"] = (
    angle_specific_candidates["Pr_good_0p90_1p10"] &
    angle_specific_candidates["AF_good_best_angle"]
)

# Strict candidate: strict Pr and at least one angle with acceptable AF
angle_specific_candidates["Strict_candidate"] = (
    angle_specific_candidates["Pr_strict_0p95_1p05"] &
    angle_specific_candidates["AF_good_best_angle"]
)

angle_specific_candidates = angle_specific_candidates.sort_values(
    [
        "Recommended_candidate",
        "Strict_candidate",
        "Angle_specific_combined_error",
        "Pr_error_from_ideal_1",
        "Best_angle_AF_error_from_1"
    ],
    ascending=[False, False, True, True, True]
).reset_index(drop=True)

recommended_candidates = angle_specific_candidates[
    angle_specific_candidates["Recommended_candidate"] == True
].copy()

strict_candidates = angle_specific_candidates[
    angle_specific_candidates["Strict_candidate"] == True
].copy()

# Save outputs
angle_specific_path = ANGLE_SPECIFIC_DIR / "candidate_screening_angle_specific_candidates.csv"
angle_specific_excel_path = ANGLE_SPECIFIC_DIR / "candidate_screening_angle_specific_candidates.xlsx"

recommended_path = ANGLE_SPECIFIC_DIR / "candidate_screening_recommended_candidates.csv"
recommended_excel_path = ANGLE_SPECIFIC_DIR / "candidate_screening_recommended_candidates.xlsx"

strict_path = ANGLE_SPECIFIC_DIR / "candidate_screening_strict_candidates.csv"
strict_excel_path = ANGLE_SPECIFIC_DIR / "candidate_screening_strict_candidates.xlsx"

angle_specific_candidates.to_csv(angle_specific_path, index=False)
angle_specific_candidates.to_excel(angle_specific_excel_path, index=False)

recommended_candidates.to_csv(recommended_path, index=False)
recommended_candidates.to_excel(recommended_excel_path, index=False)

strict_candidates.to_csv(strict_path, index=False)
strict_candidates.to_excel(strict_excel_path, index=False)

print("Angle-specific candidate files saved:")
print(angle_specific_path)
print(recommended_path)
print(strict_path)

print("\nCandidate counts:")
print("All angle-specific candidates:", len(angle_specific_candidates))
print("Recommended candidates:", len(recommended_candidates))
print("Strict candidates:", len(strict_candidates))

print("\nTop angle-specific candidates:")
display(angle_specific_candidates[[
    "Formulation_label",
    "Alg_mg_ml",
    "HA_mg_ml",
    "Pressure_kPa",
    "Best_angle_deg",
    "Predicted_Pr",
    "Pr_error_from_ideal_1",
    "Predicted_SR",
    "Predicted_Qm",
    "Best_angle_predicted_AF",
    "Best_angle_AF_error_from_1",
    "Angle_specific_combined_error",
    "Recommended_candidate",
    "Strict_candidate"
]].head(25))

print("\nRecommended candidates:")
display(recommended_candidates[[
    "Formulation_label",
    "Alg_mg_ml",
    "HA_mg_ml",
    "Pressure_kPa",
    "Best_angle_deg",
    "Predicted_Pr",
    "Predicted_SR",
    "Predicted_Qm",
    "Best_angle_predicted_AF",
    "Angle_specific_combined_error",
    "Strict_candidate"
]].head(25))

print("\nStrict candidates:")
display(strict_candidates[[
    "Formulation_label",
    "Alg_mg_ml",
    "HA_mg_ml",
    "Pressure_kPa",
    "Best_angle_deg",
    "Predicted_Pr",
    "Predicted_SR",
    "Predicted_Qm",
    "Best_angle_predicted_AF",
    "Angle_specific_combined_error"
]].head(25))


# %% Cell 25

# Cell 22: Generate final candidate screening figures for manuscript

from matplotlib.backends.backend_pdf import PdfPages

DESIGN_FIG_DIR = FIGURES_DIR / "candidate_screening"
DESIGN_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Reload key candidate screening tables if needed
if "angle_specific_candidates" not in globals():
    angle_specific_candidates = pd.read_csv(
        CANDIDATE_SCREENING_DIR / "angle_specific_candidates" / "candidate_screening_angle_specific_candidates.csv"
    )

if "recommended_candidates" not in globals():
    recommended_candidates = pd.read_csv(
        CANDIDATE_SCREENING_DIR / "angle_specific_candidates" / "candidate_screening_recommended_candidates.csv"
    )

if "strict_candidates" not in globals():
    strict_candidates = pd.read_csv(
        CANDIDATE_SCREENING_DIR / "angle_specific_candidates" / "candidate_screening_strict_candidates.csv"
    )

if "candidate_screening_printability" not in globals():
    candidate_screening_printability = pd.read_csv(
        CANDIDATE_SCREENING_DIR / "candidate_screening_printability_predictions.csv"
    )

if "af_pressure_summary" not in globals():
    af_pressure_summary = pd.read_csv(
        CANDIDATE_SCREENING_DIR / "candidate_screening_af_pressure_summary.csv"
    )

def save_current_figure(filename_base):
    png_path = DESIGN_FIG_DIR / f"{filename_base}.png"
    pdf_path = DESIGN_FIG_DIR / f"{filename_base}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    return str(png_path), str(pdf_path)

def plot_heatmap(ax, pivot_df, title, colorbar_label):
    data = pivot_df.values.astype(float)
    masked_data = np.ma.masked_invalid(data)

    im = ax.imshow(masked_data, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Pressure (kPa)")
    ax.set_ylabel("Formulation")

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot_df.columns], rotation=90)

    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    return im

# -------------------------------------------------------------------------
# Figure 1: Four-panel prediction candidate screening
# -------------------------------------------------------------------------

pr_pivot = candidate_screening_printability.pivot_table(
    index="Formulation_label",
    columns="Pressure_kPa",
    values="Predicted_Pr",
    aggfunc="mean"
)

sr_pivot = candidate_screening_printability.pivot_table(
    index="Formulation_label",
    columns="Pressure_kPa",
    values="Predicted_SR",
    aggfunc="mean"
)

qm_pivot = candidate_screening_printability.pivot_table(
    index="Formulation_label",
    columns="Pressure_kPa",
    values="Predicted_Qm",
    aggfunc="mean"
)

af_pivot = af_pressure_summary.pivot_table(
    index="Formulation_label",
    columns="Pressure_kPa",
    values="AF_mean_pred",
    aggfunc="mean"
)

fig, axes = plt.subplots(2, 2, figsize=(15, 9))

plot_heatmap(
    axes[0, 0],
    pr_pivot,
    "Predicted printability ratio (Pr)",
    "Predicted Pr"
)

plot_heatmap(
    axes[0, 1],
    sr_pivot,
    "Predicted spreading ratio (SR)",
    "Predicted SR"
)

plot_heatmap(
    axes[1, 0],
    qm_pivot,
    "Predicted mass flow rate (Qm)",
    "Predicted Qm (mg/s)"
)

plot_heatmap(
    axes[1, 1],
    af_pivot,
    "Mean predicted angle fidelity (AF)",
    "Mean predicted AF"
)

plt.tight_layout()
candidate_screening_heatmap_png, candidate_screening_heatmap_pdf = save_current_figure(
    "figure_candidate_screening_prediction_heatmaps"
)
plt.show()

# -------------------------------------------------------------------------
# Figure 2: Recommended and strict candidate map
# -------------------------------------------------------------------------

candidate_plot_df = angle_specific_candidates.copy()

formulation_order = (
    candidate_plot_df[["Formulation_label", "Alg_mg_ml", "HA_mg_ml"]]
    .drop_duplicates()
    .sort_values(["Alg_mg_ml", "HA_mg_ml"])
    ["Formulation_label"]
    .tolist()
)

formulation_to_y = {
    formulation: idx
    for idx, formulation in enumerate(formulation_order)
}

candidate_plot_df["Y_position"] = candidate_plot_df["Formulation_label"].map(formulation_to_y)

recommended_plot_df = candidate_plot_df[
    candidate_plot_df["Recommended_candidate"] == True
].copy()

strict_plot_df = candidate_plot_df[
    candidate_plot_df["Strict_candidate"] == True
].copy()

plt.figure(figsize=(11, 5))

plt.scatter(
    candidate_plot_df["Pressure_kPa"],
    candidate_plot_df["Y_position"],
    s=35,
    alpha=0.30,
    label="All candidate screening points"
)

plt.scatter(
    recommended_plot_df["Pressure_kPa"],
    recommended_plot_df["Y_position"],
    s=70,
    marker="s",
    alpha=0.75,
    label="Recommended candidates"
)

plt.scatter(
    strict_plot_df["Pressure_kPa"],
    strict_plot_df["Y_position"],
    s=140,
    marker="*",
    alpha=0.95,
    label="Strict candidates"
)

plt.yticks(
    ticks=np.arange(len(formulation_order)),
    labels=formulation_order
)

plt.xlabel("Pressure (kPa)")
plt.ylabel("Formulation")
plt.title("XAI-guided candidate screening candidates")
plt.legend(loc="best")
plt.tight_layout()

candidate_map_png, candidate_map_pdf = save_current_figure(
    "figure_candidate_screening_candidate_map"
)
plt.show()

# -------------------------------------------------------------------------
# Figure 3: Top strict candidates ranked by combined error
# -------------------------------------------------------------------------

top_strict_for_plot = strict_candidates.copy().head(17)

top_strict_for_plot["Candidate_label"] = (
    top_strict_for_plot["Formulation_label"].astype(str)
    + " | "
    + top_strict_for_plot["Pressure_kPa"].astype(int).astype(str)
    + " kPa | "
    + top_strict_for_plot["Best_angle_deg"].astype(int).astype(str)
    + " deg"
)

plot_df = top_strict_for_plot.sort_values(
    "Angle_specific_combined_error",
    ascending=True
)

plt.figure(figsize=(9, 7))
plt.barh(
    plot_df["Candidate_label"],
    plot_df["Angle_specific_combined_error"]
)

plt.xlabel("Combined error from ideal Pr and best-angle AF")
plt.ylabel("Strict candidate")
plt.title("Top strict design candidates")
plt.tight_layout()

top_candidates_png, top_candidates_pdf = save_current_figure(
    "figure_top_strict_design_candidates"
)
plt.show()

# -------------------------------------------------------------------------
# Figure 4: Best-angle distribution among recommended candidates
# -------------------------------------------------------------------------

plt.figure(figsize=(6, 4))
plt.hist(
    recommended_candidates["Best_angle_deg"].dropna(),
    bins=len(sorted(recommended_candidates["Best_angle_deg"].dropna().unique()))
)

plt.xlabel("Best angle (degrees)")
plt.ylabel("Number of recommended candidates")
plt.title("Best-angle distribution for recommended candidates")
plt.tight_layout()

best_angle_png, best_angle_pdf = save_current_figure(
    "figure_best_angle_distribution"
)
plt.show()

# -------------------------------------------------------------------------
# Save final candidate table for manuscript
# -------------------------------------------------------------------------

final_top_candidate_table = strict_candidates[[
    "Formulation_label",
    "Alg_mg_ml",
    "HA_mg_ml",
    "Pressure_kPa",
    "Best_angle_deg",
    "Predicted_Pr",
    "Predicted_SR",
    "Predicted_Qm",
    "Best_angle_predicted_AF",
    "Angle_specific_combined_error"
]].copy()

final_top_candidate_table_path = TABLES_DIR / "final_top_strict_design_candidates.csv"
final_top_candidate_table_excel_path = TABLES_DIR / "final_top_strict_design_candidates.xlsx"

final_top_candidate_table.to_csv(final_top_candidate_table_path, index=False)
final_top_candidate_table.to_excel(final_top_candidate_table_excel_path, index=False)

# -------------------------------------------------------------------------
# Figure manifest
# -------------------------------------------------------------------------

figure_manifest = pd.DataFrame([
    {
        "Figure_name": "Prediction heatmaps",
        "Description": "Four-panel candidate screening showing predicted Pr, SR, Qm, and mean AF across supported formulation-pressure space.",
        "PNG_file": candidate_screening_heatmap_png,
        "PDF_file": candidate_screening_heatmap_pdf
    },
    {
        "Figure_name": "Candidate map",
        "Description": "Map of all candidate screening points, recommended candidates, and strict candidates.",
        "PNG_file": candidate_map_png,
        "PDF_file": candidate_map_pdf
    },
    {
        "Figure_name": "Top strict candidates",
        "Description": "Strict candidate ranking based on combined Pr and best-angle AF error.",
        "PNG_file": top_candidates_png,
        "PDF_file": top_candidates_pdf
    },
    {
        "Figure_name": "Best-angle distribution",
        "Description": "Distribution of best predicted angle among recommended candidates.",
        "PNG_file": best_angle_png,
        "PDF_file": best_angle_pdf
    }
])

figure_manifest_path = TABLES_DIR / "candidate_screening_figure_manifest.csv"
figure_manifest_excel_path = TABLES_DIR / "candidate_screening_figure_manifest.xlsx"

figure_manifest.to_csv(figure_manifest_path, index=False)
figure_manifest.to_excel(figure_manifest_excel_path, index=False)

print("Final candidate screening figures saved:")
display(figure_manifest)

print("\nFinal strict candidate table saved:")
print(final_top_candidate_table_path)
print(final_top_candidate_table_excel_path)

print("\nFinal top strict design candidates:")
display(final_top_candidate_table.head(17))


# %% Cell 26

# Cell 24D: Verify restored modeling datasets before sensitivity analysis

import os
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path("/content/bioink_xai_project")
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

files_to_check = {
    "Pr": PROCESSED_DIR / "modeling_dataset_Pr.csv",
    "SR": PROCESSED_DIR / "modeling_dataset_SR.csv",
    "Qm": PROCESSED_DIR / "modeling_dataset_Qm.csv",
    "AF": PROCESSED_DIR / "modeling_dataset_AF.csv"
}

important_columns = [
    "Alg_mg_ml",
    "HA_mg_ml",
    "Pressure_kPa",
    "Angle_deg",
    "TP",
    "P_over_TP",
    "P_over_Alg",
    "P_over_HA",
    "Alg_x_P",
    "HA_x_P",
    "TP_x_P",
    "TP_x_Angle",
    "Pr_mean",
    "SR_mean",
    "Qm_mean_mg_s",
    "AF_mean"
]

for target, file_path in files_to_check.items():
    print("=" * 80)
    print(f"Target: {target}")
    print(f"File exists: {file_path.exists()}")
    print(f"File path: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    existing = [col for col in important_columns if col in df.columns]
    missing = [col for col in important_columns if col not in df.columns]

    print("\nExisting important columns:")
    print(existing)

    print("\nMissing important columns:")
    print(missing)

    print("\nFirst 5 columns:")
    print(df.columns[:5].tolist())

print("\nVerification completed.")


# %% Cell 27

# Cell 24E: Engineered-feature ablation sensitivity analysis

import os
import warnings
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

PROCESSED_DIR = str(PROJECT_DIR / "data" / "processed")
RESULTS_DIR = str(PROJECT_DIR / "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

os.makedirs(TABLES_DIR, exist_ok=True)

RANDOM_STATE = 42

target_info = {
    "Pr": {
        "file": "modeling_dataset_Pr.csv",
        "target": "Pr_mean",
        "core": ["Alg_mg_ml", "HA_mg_ml", "Pressure_kPa"]
    },
    "SR": {
        "file": "modeling_dataset_SR.csv",
        "target": "SR_mean",
        "core": ["Alg_mg_ml", "HA_mg_ml", "Pressure_kPa"]
    },
    "Qm": {
        "file": "modeling_dataset_Qm.csv",
        "target": "Qm_mean_mg_s",
        "core": ["Alg_mg_ml", "HA_mg_ml", "Pressure_kPa"]
    },
    "AF": {
        "file": "modeling_dataset_AF.csv",
        "target": "AF_mean",
        "core": ["Alg_mg_ml", "HA_mg_ml", "Pressure_kPa", "Angle_deg"]
    }
}

engineered_general = [
    "TP",
    "TP_pct_wv",
    "HF",
    "Alg_fraction_feature",
    "Alg_HA_ratio_safe",
    "P_over_TP",
    "P_over_Alg",
    "P_over_HA",
    "Alg_x_P",
    "HA_x_P",
    "TP_x_P",
    "HF_x_P",
    "Alg_x_HA",
    "Alg_fraction_x_HA_fraction"
]

engineered_angle = [
    "Angle_rad",
    "sin_angle",
    "cos_angle",
    "P_x_Angle",
    "Alg_x_Angle",
    "HA_x_Angle",
    "TP_x_Angle"
]

def get_existing(df, cols):
    return [col for col in cols if col in df.columns]

def make_groups(df):
    if "Formulation_label" in df.columns:
        return df["Formulation_label"].astype(str)

    alg = (df["Alg_mg_ml"] / 10).round().astype(int).astype(str)
    ha = (df["HA_mg_ml"] / 10).round().astype(int).astype(str)
    return alg + "ALG" + ha + "HA"

def safe_spearman(y_true, y_pred):
    value = spearmanr(y_true, y_pred).correlation
    if pd.isna(value):
        return np.nan
    return value

def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Spearman": safe_spearman(y_true, y_pred)
    }

def build_models():
    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "SVR_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10.0, epsilon=0.05))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            min_samples_leaf=1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        )
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1
        )
    except Exception as error:
        print("XGBoost skipped:", error)

    return models

def evaluate_repeated_kfold(X, y, model):
    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=20,
        random_state=RANDOM_STATE
    )

    rows = []

    for train_idx, test_idx in rkf.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rows.append(metrics(y_test, y_pred))

    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()

def evaluate_lofo(X, y, groups, model):
    logo = LeaveOneGroupOut()

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

    return metrics(np.array(y_true_all), np.array(y_pred_all))

all_rows = []

for target_name, info in target_info.items():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, info["file"]))
    df = df.dropna(subset=[info["target"]]).reset_index(drop=True)

    core_features = get_existing(df, info["core"])

    engineered_features = core_features + get_existing(df, engineered_general)

    if target_name == "AF":
        engineered_features = engineered_features + get_existing(df, engineered_angle)

    engineered_features = list(dict.fromkeys(engineered_features))

    feature_sets = {
        "core": core_features,
        "engineered": engineered_features
    }

    y = df[info["target"]]
    groups = make_groups(df)

    for feature_set_name, feature_list in feature_sets.items():
        X = df[feature_list].copy()

        for model_name, model in build_models().items():
            repeated = evaluate_repeated_kfold(X, y, model)
            lofo = evaluate_lofo(X, y, groups, model)

            all_rows.append({
                "Target": target_name,
                "Feature_set": feature_set_name,
                "Model": model_name,
                "Validation": "Repeated_KFold",
                "N": len(df),
                "N_features": len(feature_list),
                "R2": repeated["R2"],
                "MAE": repeated["MAE"],
                "RMSE": repeated["RMSE"],
                "Spearman": repeated["Spearman"],
                "Feature_list": ", ".join(feature_list)
            })

            all_rows.append({
                "Target": target_name,
                "Feature_set": feature_set_name,
                "Model": model_name,
                "Validation": "LOFO",
                "N": len(df),
                "N_features": len(feature_list),
                "R2": lofo["R2"],
                "MAE": lofo["MAE"],
                "RMSE": lofo["RMSE"],
                "Spearman": lofo["Spearman"],
                "Feature_list": ", ".join(feature_list)
            })

ablation_all = pd.DataFrame(all_rows)

all_path = os.path.join(TABLES_DIR, "engineered_feature_ablation_all_models.csv")
all_xlsx_path = os.path.join(TABLES_DIR, "engineered_feature_ablation_all_models.xlsx")

ablation_all.to_csv(all_path, index=False)
ablation_all.to_excel(all_xlsx_path, index=False)

best_rows = []

for (target, validation, feature_set), group in ablation_all.groupby(["Target", "Validation", "Feature_set"]):
    best_row = group.sort_values("RMSE", ascending=True).iloc[0]
    best_rows.append(best_row)

best_by_set = pd.DataFrame(best_rows)

summary_rows = []

for (target, validation), group in best_by_set.groupby(["Target", "Validation"]):
    core = group[group["Feature_set"] == "core"].iloc[0]
    engineered = group[group["Feature_set"] == "engineered"].iloc[0]

    summary_rows.append({
        "Target": target,
        "Validation": validation,
        "Best_core_model": core["Model"],
        "Core_N_features": core["N_features"],
        "Core_R2": core["R2"],
        "Core_MAE": core["MAE"],
        "Core_RMSE": core["RMSE"],
        "Core_Spearman": core["Spearman"],
        "Best_engineered_model": engineered["Model"],
        "Engineered_N_features": engineered["N_features"],
        "Engineered_R2": engineered["R2"],
        "Engineered_MAE": engineered["MAE"],
        "Engineered_RMSE": engineered["RMSE"],
        "Engineered_Spearman": engineered["Spearman"],
        "Delta_R2_engineered_minus_core": engineered["R2"] - core["R2"],
        "Delta_MAE_core_minus_engineered": core["MAE"] - engineered["MAE"],
        "Delta_RMSE_core_minus_engineered": core["RMSE"] - engineered["RMSE"],
        "Delta_Spearman_engineered_minus_core": engineered["Spearman"] - core["Spearman"],
        "Engineered_improved_RMSE": engineered["RMSE"] < core["RMSE"],
        "Engineered_improved_Spearman": engineered["Spearman"] > core["Spearman"]
    })

ablation_summary = pd.DataFrame(summary_rows)

summary_path = os.path.join(TABLES_DIR, "engineered_feature_ablation_best_summary.csv")
summary_xlsx_path = os.path.join(TABLES_DIR, "engineered_feature_ablation_best_summary.xlsx")

ablation_summary.to_csv(summary_path, index=False)
ablation_summary.to_excel(summary_xlsx_path, index=False)

print("Engineered-feature sensitivity analysis completed.")
print("Saved:")
print(all_path)
print(summary_path)

display(
    ablation_summary.sort_values(["Target", "Validation"]).reset_index(drop=True)
)
