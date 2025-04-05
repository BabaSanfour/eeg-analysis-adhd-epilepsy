import os 
import numpy as np

user = os.path.expanduser('~')
# Path to the directory where the data is stored
data_dir = os.path.join(user, 'Projects/data/EEG_psychostimulant_data/EEG_psychostimulants_2025-02')
source_dirs = {"control": "Controls", "patients": "patients", }
csv_dir = os.path.join(data_dir, 'csv')
bids_dir = os.path.join(data_dir, 'BIDS')
derivatives_dir = os.path.join(data_dir, 'derivatives')
sensors_to_keep = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "A1", "T3", "C3", "Cz",
                "C4", "T4", "A2", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]

results_dir = os.path.join(data_dir, 'results')
n_subjects = 252


# Mapping to compute psychostimulant category based on description
MAPPING_PSYCHOSTIMULANT = {
    np.nan: 0,
    'Lisdexamfetamine': 1,
    'Lisdexamfetamine (d/c)': 1,
    'Methylphenidate (d/c)': 2,
    'Methylphenidate': 2,
    'Methylphenidate, Methylphenidate': 2,
    'Lisdexamfetamine, Methylphenidate': 3,
    'Dextroamphetamine (Dexedrine)': 4,
    'Amphetamine/dextroamphetamine salt (Adderall)': 5,
}

