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
_candidates = [_cwd, _cwd.parent]
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
    df["HF"] = df["HA_fraction"]
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

    # Angle-related features for angular fidelity model
    if include_angle and "Angle_deg" in df.columns:

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
    "TP", "HF", "Alg_HA_ratio_safe",
    "P_over_TP", "P_over_Alg", "P_over_HA",
    "Alg_x_HA", "Alg_x_P", "HA_x_P", "TP_x_P", "HF_x_P"
]
print(new_cols_printability)

display(printability_ml[[
    "Formulation_label", "Alg_mg_ml", "HA_mg_ml", "Pressure_kPa",
    "Pr_mean", "Qm_mean_mg_s", "SR_mean"
] + new_cols_printability].head())

print("\nNew ML feature columns in angle fidelity dataset:")
new_cols_angle = new_cols_printability + [
    "Angle_deg", "P_x_Angle", "Alg_x_Angle", "HA_x_Angle", "TP_x_Angle"
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

    # No renaming is required after redundant fraction aliases were removed.
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
        "Feature_name": "HF",
        "Symbol": "HF",
        "Group": "Engineered composition feature",
        "Definition": "Hyaluronic acid fraction, calculated as HA / (Alg + HA)",
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
    # Reviewer-responsive nonredundant engineered representation.
    # Exact scale aliases and deterministic fraction duplicates are excluded.
    "TP",
    "HF",
    "Alg_HA_ratio_safe",
    "P_over_TP",
    "P_over_Alg",
    "P_over_HA",
    "Alg_x_HA",
    "Alg_x_P",
    "HA_x_P",
    "TP_x_P",
    "HF_x_P"
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
    # Use the experimentally sampled angle directly; redundant reparameterizations
    # (radian, sine, cosine) are excluded to avoid fragmented attribution.
    "Angle_deg",
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

# Reviewer-responsive XAI analysis for Comments 8-R1 and 17-18-R2.
# The prior in-sample permutation/consensus-score workflow has been retired.
# This script computes held-out permutation importance, reports SHAP separately,
# excludes Pr from rank-agreement summaries, and writes replacement XAI artifacts.

import subprocess
import sys

reviewer_xai_script = PROJECT_DIR / "src" / "revision_comments_8_R1_17_18_R2_xai.py"
subprocess.run([sys.executable, str(reviewer_xai_script)], check=True)

reviewer_xai_dir = RESULTS_DIR / "revision_comments_8_R1_17_18_R2"
permutation_importance_df = pd.read_csv(reviewer_xai_dir / "OOF_Permutation_Repeated_Summary_Comments_18-R2.csv")
permutation_importance_lofo_df = pd.read_csv(reviewer_xai_dir / "OOF_Permutation_LOFO_Summary_Comments_18-R2.csv")
shap_summary_df = pd.read_csv(reviewer_xai_dir / "SHAP_Separate_Summary_Comments_17-R2_8-R1.csv")
xai_comparison_df = pd.read_csv(reviewer_xai_dir / "Separate_XAI_Ranks_Comments_17-R2_8-R1.csv")
rank_agreement_df = pd.read_csv(reviewer_xai_dir / "Rank_Agreement_Excluding_Pr_Comment_8-R1.csv")
design_rules_df = pd.read_csv(reviewer_xai_dir / "Table3_Separate_XAI_Comments_17-R2_8-R1.csv")

print("Reviewer-responsive XAI outputs loaded from:", reviewer_xai_dir)
print("Consensus XAI score: removed")
print("Permutation importance: held-out only")
print("Pr rank agreement: excluded")

display(rank_agreement_df)
display(design_rules_df)


# %% Cell 22

# Cell 21-fix: Redefine ML feature function with corrected feature name

def add_ml_features(df, include_angle=False):
    df = df.copy()

    # Core composition features
    df["TP"] = df["Total_polymer_mg_ml"]
    df["HF"] = df["HA_fraction"]
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

    # Angle-related features for angular fidelity model
    if include_angle and "Angle_deg" in df.columns:

        if "Pressure_kPa" in df.columns:
            df["P_x_Angle"] = df["Pressure_kPa"] * df["Angle_deg"]
            df["Alg_x_Angle"] = df["Alg_mg_ml"] * df["Angle_deg"]
            df["HA_x_Angle"] = df["HA_mg_ml"] * df["Angle_deg"]
            df["TP_x_Angle"] = df["TP"] * df["Angle_deg"]

    return df

print("add_ml_features function has been corrected.")
print("Now rerun Cell 21 from the beginning.")


# %% Reviewer-responsive candidate-set screening: Comments 2-R1 and 6-7-R2

# The prior model-ranked candidate workflow has been retired. Reviewer 1 Comment 2
# and Reviewer 2 Comments 6-7 identified two linked limitations: the previous
# within-set ordering was finer than the predictive error, and predicted Pr was
# too weak to serve as a narrow screening gate. The revised analysis therefore
# uses only jointly observed formulation-pressure conditions, applies the Pr and
# AF criteria to measured source-study values, and reports an unordered candidate
# set. No combined-error score or scientific rank is calculated.

import subprocess
import sys

candidate_revision_script = PROJECT_DIR / "src" / "revision_comments_2_R1_6_7_R2_candidate_set.py"
subprocess.run([sys.executable, str(candidate_revision_script)], check=True)

candidate_revision_dir = RESULTS_DIR / "revision_comments_2_R1_6_7_R2"
measured_candidate_set = pd.read_csv(
    candidate_revision_dir / "Table4_Unordered_Candidate_Set_Comments_2-R1_6-7-R2.csv"
)
candidate_screening_audit = pd.read_csv(
    candidate_revision_dir / "Candidate_Screening_Audit_Comments_2-R1_6-7-R2.csv"
)

print("Reviewer-responsive candidate-set analysis loaded from:", candidate_revision_dir)
print("Predicted Pr gate: removed")
print("Within-set ranking: removed")
print("Candidate-set size:", len(measured_candidate_set))

display(candidate_screening_audit)
display(measured_candidate_set)


# %% Cell 26

# Cell 24D: Verify restored modeling datasets before sensitivity analysis

import os
import pandas as pd
from pathlib import Path

# Reuse the repository root resolved in the setup section.
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
    "HF",
    "Alg_HA_ratio_safe",
    "P_over_TP",
    "P_over_Alg",
    "P_over_HA",
    "Alg_x_P",
    "HA_x_P",
    "TP_x_P",
    "HF_x_P",
    "Alg_x_HA"
]

engineered_angle = [
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
