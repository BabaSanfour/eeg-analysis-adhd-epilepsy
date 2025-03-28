"""
analysis_design.py

This script reads a demographic CSV file and a features CSV file that share the same "Study ID".
It then filters the merged dataset based on user-provided combinations for:
    - sex (F, M, or combined)
    - age_groups (1, 2, or combined)
    - ADHD (with, without, or combined)
    - TSA (with, without, or combined)
    - Epilepsy (with, without, or combined)

Based on the specified analysis type:
    - "medications": Only subjects with psychostimulant_category 1 or 2 are kept and are labeled as "Med1" or "Med2".
    - "general": Subjects are classified as control (0) or psychostimulant (1) based on their "Psychostimulant (y/n)" status.

Usage:
    python analysis_design.py --demographic_csv path/to/demographic.csv \
        --features_csv path/to/features.csv --sex F --age_groups 1 --ADHD with --TSA combined \
        --Epilepsy without --analysis_type medications
"""

import os
import sys
import argparse
import logging
import warnings
import pandas as pd
import numpy as np

# Configure logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Update system path to import configuration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import csv_dir, MAPPING_PSYCHOSTIMULANT


def load_csv(file_path):
    """
    Load a CSV file into a DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {os.path.basename(file_path)} with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        sys.exit(1)


def merge_data(demo_df, features_df):
    """
    Merge demographic and features DataFrames on 'Study ID'.

    Args:
        demo_df (pd.DataFrame): Demographic DataFrame.
        features_df (pd.DataFrame): Features DataFrame.
        
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    if 'Study ID' not in demo_df.columns or 'Study ID' not in features_df.columns:
        logging.error("Both CSV files must contain a 'Study ID' column.")
        sys.exit(1)
    merged_df = pd.merge(demo_df, features_df, on="Study ID", how="inner")
    logging.info(f"Merged data has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")
    return merged_df


def compute_age_groups(df):
    """
    Compute age groups from an Age column if not already present.

    The age groups are defined as:
        - Group 1: Ages [0, 12)
        - Group 2: Ages [12, 19)

    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with a new 'age_groups' column.
    """
    if "age_groups" not in df.columns:
        df["age_groups"] = pd.cut(df["Age"], bins=[0, 12, 19], labels=[1, 2], right=False)
        logging.info("Computed 'age_groups' from 'Age' column.")
    return df


def filter_by_sex(df, sex):
    """
    Filter DataFrame by sex.

    Args:
        df (pd.DataFrame): Input DataFrame.
        sex (str): Filter condition ("F", "M", or "combined").
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if sex.lower() in ['f', 'm']:
        filtered = df[df['Sex'] == sex.upper()]
        logging.info(f"Filtered by Sex = {sex.upper()}: {filtered.shape[0]} rows remain.")
        return filtered
    logging.info("Sex filter set to 'combined'; no filtering applied.")
    return df


def filter_by_age_group(df, age_group):
    """
    Filter DataFrame by age group.

    Args:
        df (pd.DataFrame): Input DataFrame.
        age_group (str): Age group filter ("1", "2", or "combined").
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = compute_age_groups(df)
    if age_group.lower() != "combined":
        try:
            age_group_val = int(age_group)
            filtered = df[df['age_groups'] == age_group_val]
            logging.info(f"Filtered by age_group = {age_group_val}: {filtered.shape[0]} rows remain.")
            return filtered
        except ValueError:
            logging.error("Age group must be 1, 2, or 'combined'.")
            sys.exit(1)
    logging.info("Age group filter set to 'combined'; no filtering applied.")
    return df


def filter_diagnosis(df, diag_col, condition, include_potential=True):
    """
    Filter DataFrame based on a diagnosis condition.

    Args:
        df (pd.DataFrame): Input DataFrame.
        diag_col (str): Column name for the diagnosis.
        condition (str): Diagnosis filter ("with", "without", or "combined").
        include_potential (bool): If True, include potential diagnoses when condition is "with".
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if condition.lower() == "combined":
        logging.info(f"No filtering applied for {diag_col} (combined condition).")
        return df

    if condition.lower() == "with":
        valid_values = ['1', '0 (potentiel)'] if include_potential else ['1']
    elif condition.lower() == "without":
        valid_values = ['0'] if include_potential else ['0', '0 (potentiel)']
    else:
        logging.warning(f"Unrecognized diagnosis filter for {diag_col}: {condition}. No filtering applied.")
        return df

    filtered = df[df[diag_col].isin(valid_values)]
    logging.info(f"Filtered {diag_col} with condition '{condition}': {filtered.shape[0]} rows remain.")
    return filtered


def create_target_column(df, analysis_type):
    """
    Create a target column based on the specified analysis type.

    For "medications":
        - Keep rows with psychostimulant_category 1 or 2.
        - Create target: "Med1" if psychostimulant_category == 1, "Med2" if equal to 2.
    For "general":
        - Use the "Psychostimulant (y/n)" column.
        - Define target: 0 for control (value 0 or NaN), 1 for psychostimulant (any other value).

    Args:
        df (pd.DataFrame): Input DataFrame.
        analysis_type (str): "medications" or "general".
        
    Returns:
        pd.DataFrame: DataFrame with an added "target" column.
    """
    if analysis_type.lower() == "medications":
        # Ensure psychostimulant_category exists; compute from description if needed
        if 'psychostimulant_category' not in df.columns:
            if 'psychostimulant_description' in df.columns:
                df['psychostimulant_category'] = df['psychostimulant_description'].map(MAPPING_PSYCHOSTIMULANT)
                logging.info("Computed 'psychostimulant_category' from 'psychostimulant_description'.")
            else:
                logging.error("For 'medications' analysis, either 'psychostimulant_category' or 'psychostimulant_description' must be present.")
                sys.exit(1)
        # Filter to keep only categories 1 and 2
        df = df[df['psychostimulant_category'].isin([1, 2])].copy()
        df['target'] = df['psychostimulant_category'].apply(lambda x: 'Med1' if x == 1 else 'Med2')
        logging.info("Created target column for 'medications' analysis.")
    
    elif analysis_type.lower() == "general":
        if "Psychostimulant (y/n)" not in df.columns:
            logging.error("For 'general' analysis, the 'Psychostimulant (y/n)' column is required.")
            sys.exit(1)
        df['target'] = df['Psychostimulant (y/n)'].apply(lambda x: 0 if pd.isna(x) or x == 0 else 1)
        logging.info("Created target column for 'general' analysis (0: Control, 1: Psychostimulant).")
    
    else:
        logging.error("Analysis type must be either 'medications' or 'general'.")
        sys.exit(1)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Merge demographic and features CSVs, filter by given variables, and create a target column for classification."
    )
    parser.add_argument("--demographic_csv", type=str, required=True,
                        help="Path to the demographic CSV file.")
    parser.add_argument("--features_csv", type=str, required=True,
                        help="Path to the features CSV file.")
    parser.add_argument("--sex", type=str, choices=["F", "M", "combined"], required=True,
                        help="Sex filter: F, M, or combined.")
    parser.add_argument("--age_groups", type=str, choices=["1", "2", "combined"], required=True,
                        help="Age group filter: 1, 2, or combined.")
    parser.add_argument("--ADHD", type=str, choices=["with", "without", "combined"], required=True,
                        help="ADHD diagnosis filter: with, without, or combined.")
    parser.add_argument("--TSA", type=str, choices=["with", "without", "combined"], required=True,
                        help="TSA diagnosis filter: with, without, or combined.")
    parser.add_argument("--Epilepsy", type=str, choices=["with", "without", "combined"], required=True,
                        help="Epilepsy diagnosis filter: with, without, or combined.")
    parser.add_argument("--analysis_type", type=str, choices=["medications", "general"], required=True,
                        help="Analysis type: 'medications' (for med1_vs_med2) or 'general' (for ctrl vs psychostimulant).")
    args = parser.parse_args()

    # Load both CSV files
    demo_df = load_csv(args.demographic_csv)
    features_df = load_csv(args.features_csv)

    # Merge on 'Study ID'
    merged_df = merge_data(demo_df, features_df)

    # Apply filtering based on provided parameters
    filtered_df = filter_by_sex(merged_df, args.sex)
    filtered_df = filter_by_age_group(filtered_df, args.age_groups)
    filtered_df = filter_diagnosis(filtered_df, "ADHD", args.ADHD, include_potential=True)
    filtered_df = filter_diagnosis(filtered_df, "TSA", args.TSA, include_potential=True)
    filtered_df = filter_diagnosis(filtered_df, "Epilepsy", args.Epilepsy, include_potential=True)

    # Create the target column based on analysis type
    final_df = create_target_column(filtered_df, args.analysis_type)

    # Optionally, save the filtered and annotated data for further analysis
    output_file = f"{args.analysis_type}_classification_results.csv"
    final_df.to_csv(output_file, index=False)
    logging.info(f"Final results saved to {output_file}")


if __name__ == "__main__":
    # Display a warning if needed
    warnings.warn("Ensure that both CSV files have the required columns and that 'Study ID' exists in both.", UserWarning)
    main()
