"""
analysis_exploration.py

This script is part of the analysis-design repository. It reads an input CSV file, 
performs data cleaning and transformation, and then generates various analysis CSV files 
and a PDF report for quality control. The output files are designed for use in ML pipelines.

Usage:
    python analysis_exploration.py --csv_file <filename> [--potential_in_with]

Arguments:
    --csv_file           Path to the CSV file (default is set from the config csv_dir)
    --potential_in_with  Flag to include potential diagnoses in the 'with' condition
"""

import os
import sys
import argparse
import itertools
import warnings
import logging
import numpy as np
import pandas as pd
from fpdf import FPDF

# Update system path to import configuration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import csv_dir, MAPPING_PSYCHOSTIMULANT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ----------------------------------------------------------------------
# Data Loading and Report Generation Functions
# ----------------------------------------------------------------------
def load_csv_file(file_path):
    """
    Load a CSV file into a DataFrame, log its statistics, and generate a PDF report.
    
    Args:
        file_path (str): Full path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    report_lines = []

    base_name = os.path.basename(file_path)
    header_message = f"Loaded file: {base_name} with {df.shape[0]} rows and {df.shape[1]} columns"
    logging.info(header_message)
    report_lines.append(header_message)

    # Log key details about the dataset
    info_messages = [
        f"First 5 rows:\n{df.head()}",
        f"Columns:\n{df.columns.tolist()}",
        f"Missing values:\n{df.isnull().sum()}",
        f"Duplicate rows: {df.duplicated().sum()}",
        f"Unique values per column:\n{df.nunique()}"
    ]
    for msg in info_messages:
        logging.info(msg)
        report_lines.append(msg)
    
    # Log unique values for each column (excluding the first column)
    for column in df.columns[1:]:
        unique_msg = f"Unique values for '{column}':\n{df[column].unique()}"
        logging.info(unique_msg)
        report_lines.append(unique_msg)
    
    # Log age range details if 'Age' column exists
    if 'Age' in df.columns:
        age_msg = f"Age range: {df['Age'].min()} - {df['Age'].max()}"
        logging.info(age_msg)
        report_lines.append(age_msg)
    
    # Log value counts for each column (excluding the first column)
    for column in df.columns[1:]:
        count_msg = f"Value counts for '{column}':\n{df[column].value_counts()}"
        logging.info(count_msg)
        report_lines.append(count_msg)
    
    # Generate PDF report of the logged information
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        for line in report_lines:
            for subline in str(line).split("\n"):
                pdf.cell(0, 10, txt=subline, ln=True)
        pdf_file_name = file_path.replace(".csv", "_report.pdf")
        pdf.output(pdf_file_name)
        logging.info(f"PDF report saved to {pdf_file_name}")
    except Exception as e:
        logging.error("Failed to generate PDF report. Ensure 'fpdf' is installed. Error: " + str(e))
    
    return df

def clean_data(df):
    """
    Clean and transform the input DataFrame:
      - Map 'psychostimulant_description' to a numeric category.
      - Create age groups using defined bins.
    
    Args:
        df (pd.DataFrame): Original DataFrame.
    
    Returns:
        pd.DataFrame: Cleaned and transformed DataFrame.
    """
    df["psychostimulant_category"] = df["psychostimulant_description"].map(MAPPING_PSYCHOSTIMULANT)
    df["age_groups"] = pd.cut(df["Age"], bins=[0, 12, 19], labels=[1, 2], right=False)
    return df

# ----------------------------------------------------------------------
# Helper Filtering Functions
# ----------------------------------------------------------------------
# Define filters for Sex and Age
sex_filters = {
    'F': lambda d: d[d['Sex'] == 'F'],
    'M': lambda d: d[d['Sex'] == 'M'],
    'Combined': lambda d: d  # No filtering for combined
}

age_filters = {
    1: lambda d: d[d['age_groups'] == 1],
    2: lambda d: d[d['age_groups'] == 2],
    'Combined': lambda d: d  # No filtering for combined
}

def apply_diagnosis_filter(df, diag_col, condition, include_potential=True):
    """
    Filter the DataFrame based on a diagnosis condition.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        diag_col (str): Column name containing diagnosis data.
        condition (str): Filter condition: 'with', 'without', or 'combined'.
        include_potential (bool): If True, include potential diagnoses in the 'with' condition.
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if condition == 'combined':
        return df

    if condition == 'with':
        valid_values = ['1', '0 (potentiel)'] if include_potential else ['1']
    elif condition == 'without':
        valid_values = ['0'] if include_potential else ['0', '0 (potentiel)']
    else:
        return df

    return df[df[diag_col].isin(valid_values)]

# Define diagnosis filter options and target diagnosis columns
diag_filter_options = ['with', 'without', 'combined']
diagnosis_columns = ['TDAH', 'Epilepsy', 'TSA']

# ----------------------------------------------------------------------
# Analysis Functions
# ----------------------------------------------------------------------
def get_counts_by_med_analysis(df, sex_key, age_key, diag_filters, analysis_type, include_potential=True):
    """
    Calculate subject counts based on medication analysis type and filters.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        sex_key (str): Key for sex filter ('F', 'M', or 'Combined').
        age_key (int or str): Key for age filter (1, 2, or 'Combined').
        diag_filters (dict): Dictionary mapping diagnosis to filter condition.
        analysis_type (str): Analysis type ('ctrl_vs_all', 'med1_vs_med2', etc.).
        include_potential (bool): Flag to include potential diagnoses.
        
    Returns:
        tuple: (count_group1, count_group2, label_group1, label_group2)
    """
    # Apply sex and age filters
    filtered_df = sex_filters[sex_key](df)
    filtered_df = age_filters[age_key](filtered_df)
    
    # Apply each diagnosis filter
    for diag in diagnosis_columns:
        filtered_df = apply_diagnosis_filter(filtered_df, diag, diag_filters.get(diag, 'combined'), include_potential)

    # Define control group: rows with 'Psychostimulant (y/n)' equal to 0 or missing
    is_control = filtered_df['Psychostimulant (y/n)'].apply(lambda x: pd.isna(x) or x == 0)
    
    if analysis_type == 'ctrl_vs_all':
        count_control = filtered_df[is_control].shape[0]
        count_med = filtered_df[~is_control].shape[0]
        return count_control, count_med, 'Control', 'Med'

    elif analysis_type == 'med1_vs_med2':
        med_df = filtered_df[filtered_df['psychostimulant_category'].isin([1, 2])]
        count_med1 = med_df[med_df['psychostimulant_category'] == 1].shape[0]
        count_med2 = med_df[med_df['psychostimulant_category'] == 2].shape[0]
        return count_med1, count_med2, 'Med1', 'Med2'

    elif analysis_type == 'ctrl_vs_med1':
        sub_df = filtered_df[filtered_df['psychostimulant_category'].isin([0, 1])].copy()
        count_control = sub_df[sub_df['psychostimulant_category'].apply(lambda x: pd.isna(x) or x == 0)].shape[0]
        count_med1 = sub_df[sub_df['psychostimulant_category'] == 1].shape[0]
        return count_control, count_med1, 'Control', 'Med1'

    elif analysis_type == 'ctrl_vs_med2':
        sub_df = filtered_df[filtered_df['psychostimulant_category'].isin([0, 2])].copy()
        count_control = sub_df[sub_df['psychostimulant_category'].apply(lambda x: pd.isna(x) or x == 0)].shape[0]
        count_med2 = sub_df[sub_df['psychostimulant_category'] == 2].shape[0]
        return count_control, count_med2, 'Control', 'Med2'
    
    return None, None, None, None

def create_analysis_dataframe(df, analysis_type, include_potential=True):
    """
    Create a summary DataFrame with counts for each combination of filters.
    
    Args:
        df (pd.DataFrame): Cleaned input DataFrame.
        analysis_type (str): Type of medication analysis.
        include_potential (bool): Flag to include potential diagnoses.
        
    Returns:
        pd.DataFrame: DataFrame summarizing counts across filter combinations.
    """
    results = []
    # Iterate over sex and age filter options
    for sex_key in sex_filters.keys():
        for age_key in age_filters.keys():
            # Generate all combinations of diagnosis filter conditions (3^3 = 27 combinations)
            for diag_combo in itertools.product(diag_filter_options, repeat=3):
                diag_filters = dict(zip(diagnosis_columns, diag_combo))
                count1, count2, label1, label2 = get_counts_by_med_analysis(
                    df, sex_key, age_key, diag_filters, analysis_type, include_potential
                )
                
                sub_df = df.copy()
                sub_df = sex_filters[sex_key](sub_df)
                sub_df = age_filters[age_key](sub_df)
                for diag in diagnosis_columns:
                    sub_df = apply_diagnosis_filter(sub_df, diag, diag_filters.get(diag, 'combined'), include_potential)
                
                M_count = sub_df[sub_df['Sex'] == 'M'].shape[0]
                F_count = sub_df[sub_df['Sex'] == 'F'].shape[0]
                
                overall_age_mean = sub_df['Age'].mean() if not sub_df.empty else np.nan
                overall_age_std = sub_df['Age'].std() if not sub_df.empty else np.nan
                
                female_subset = sub_df[sub_df['Sex'] == 'F']
                female_age_mean = female_subset['Age'].mean() if not female_subset.empty else np.nan
                female_age_std = female_subset['Age'].std() if not female_subset.empty else np.nan

                male_subset = sub_df[sub_df['Sex'] == 'M']
                male_age_mean = male_subset['Age'].mean() if not male_subset.empty else np.nan
                male_age_std = male_subset['Age'].std() if not male_subset.empty else np.nan

                # Calculate age statistics for group1 and group2 separately.
                # Recreate the splitting logic as per the analysis type.
                if analysis_type == 'ctrl_vs_all':
                    is_control = sub_df['Psychostimulant (y/n)'].apply(lambda x: pd.isna(x) or x == 0)
                    group1_df = sub_df[is_control]
                    group2_df = sub_df[~is_control]
                elif analysis_type == 'med1_vs_med2':
                    med_df = sub_df[sub_df['psychostimulant_category'].isin([1, 2])]
                    group1_df = med_df[med_df['psychostimulant_category'] == 1]
                    group2_df = med_df[med_df['psychostimulant_category'] == 2]
                elif analysis_type == 'ctrl_vs_med1':
                    sub_sub_df = sub_df[sub_df['psychostimulant_category'].isin([0, 1])]
                    group1_df = sub_sub_df[sub_sub_df['psychostimulant_category'].apply(lambda x: pd.isna(x) or x == 0)]
                    group2_df = sub_sub_df[sub_sub_df['psychostimulant_category'] == 1]
                elif analysis_type == 'ctrl_vs_med2':
                    sub_sub_df = sub_df[sub_df['psychostimulant_category'].isin([0, 2])]
                    group1_df = sub_sub_df[sub_sub_df['psychostimulant_category'].apply(lambda x: pd.isna(x) or x == 0)]
                    group2_df = sub_sub_df[sub_sub_df['psychostimulant_category'] == 2]
                else:
                    group1_df = pd.DataFrame()
                    group2_df = pd.DataFrame()

                # Compute age statistics for group1
                age_mean_group1 = group1_df['Age'].mean() if not group1_df.empty else np.nan
                age_std_group1  = group1_df['Age'].std()  if not group1_df.empty else np.nan
                # Compute age statistics for group2
                age_mean_group2 = group2_df['Age'].mean() if not group2_df.empty else np.nan
                age_std_group2  = group2_df['Age'].std()  if not group2_df.empty else np.nan

                results.append({
                    'med_analysis': analysis_type,
                    'sex': sex_key,
                    'age_group': age_key,
                    'TDAH_filter': diag_filters['TDAH'],
                    'Epilepsy_filter': diag_filters['Epilepsy'],
                    'TSA_filter': diag_filters['TSA'],
                    'M_count': M_count,
                    'F_count': F_count,
                    'age_mean_overall': overall_age_mean,
                    'age_std_overall': overall_age_std,
                    'age_mean_female': female_age_mean,
                    'age_std_female': female_age_std,
                    'age_mean_male': male_age_mean,
                    'age_std_male': male_age_std,
                    'age_mean_group1': age_mean_group1,
                    'age_std_group1': age_std_group1,
                    'age_mean_group2': age_mean_group2,
                    'age_std_group2': age_std_group2,
                    label1: count1,
                    label2: count2
                })
    return pd.DataFrame(results)

def remove_small_count_rows(df, min_count=20):
    """
    Filter out rows where either of the last two count columns is less than min_count.
    
    Args:
        df (pd.DataFrame): DataFrame with count columns.
        min_count (int): Minimum threshold for counts.
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    return df[(df.iloc[:, -2] >= min_count) & (df.iloc[:, -1] >= min_count)]

# ----------------------------------------------------------------------
# Main Execution Flow
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate analysis CSV files and PDF report from input CSV data.")
    parser.add_argument(
        "--csv_file",
        type=str,
        default=os.path.join(csv_dir, "patients_controls_new.csv"),
        help="Name of the CSV file to process (located in the csv_dir)"
    )
    parser.add_argument(
        "--potential_in_with",
        action="store_true",
        help="Include potential diagnoses in the 'with' condition"
    )
    args = parser.parse_args()
    
    # Log the flag status
    warnings.warn(f"Potential in with flag set to: {args.potential_in_with}", UserWarning)
    
    # Construct full file path
    input_file = os.path.join(csv_dir, args.csv_file)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} does not exist")
    
    # Load and preprocess the data
    df = load_csv_file(input_file)
    df = clean_data(df)
    
    # Define analysis types to process
    med_analyses = ['ctrl_vs_all', 'med1_vs_med2']
    
    for analysis in med_analyses:
        logging.info(f"Starting analysis: {analysis}")
        analysis_df = create_analysis_dataframe(df.copy(), analysis, include_potential=args.potential_in_with)
        
        # Save full analysis results
        output_csv = os.path.join(csv_dir, f"{analysis}_results.csv")
        analysis_df.to_csv(output_csv, index=False)
        logging.info(f"Results saved to {output_csv}")
        
        # Filter out rows with counts below threshold and save
        filtered_df = remove_small_count_rows(analysis_df)
        filtered_csv = os.path.join(csv_dir, f"{analysis}_filtered_results.csv")
        filtered_df.to_csv(filtered_csv, index=False)
        logging.info(f"Filtered results saved to {filtered_csv}")

if __name__ == "__main__":
    main()
