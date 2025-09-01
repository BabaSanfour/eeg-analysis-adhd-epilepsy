import os
import numpy as np


# ---------------------------------------------------------------------------
# Environment-based configuration
# ---------------------------------------------------------------------------

def _get_env_path(var_name: str, default: str | None = None) -> str:
    """Return the path from an environment variable or a default value.

    Args:
        var_name: Name of the environment variable to read.
        default: Default path to use if the environment variable is unset.

    Returns:
        The resolved path as a string.

    Raises:
        EnvironmentError: If the variable is not set and no default is provided.
    """

    value = os.environ.get(var_name, default)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{var_name}' is not set."
        )
    return value


# Base directory for EEG data
data_dir = _get_env_path("EEG_DATA_DIR")

# Subdirectories with sensible defaults that can also be overridden
embeddings_dir = _get_env_path("EEG_EMBEDDINGS_DIR", os.path.join(data_dir, "embeddings"))
results_dir = _get_env_path("EEG_RESULTS_DIR", os.path.join(data_dir, "results"))
csv_dir = _get_env_path("EEG_CSV_DIR", os.path.join(data_dir, "csv"))
bids_dir = _get_env_path("EEG_BIDS_DIR", os.path.join(data_dir, "BIDS"))
derivatives_dir = _get_env_path("EEG_DERIVATIVES_DIR", os.path.join(data_dir, "derivatives"))

source_dirs = {"control": "Controls", "patients": "patients"}

sensors_to_keep = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "A1", "T3", "C3", "Cz",
    "C4", "T4", "A2", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"
]

n_subjects = 253


# Mapping to compute psychostimulant category based on description
MAPPING_PSYCHOSTIMULANT = {
    'no psychostimulants': 0,
    'Lisdexamfetamine': 1,
    'Lisdexamfetamine (d/c)': 1,
    'Methylphenidate (d/c)': 2,
    'Methylphenidate': 2,
    'Methylphenidate, Methylphenidate': 2,
    'Lisdexamfetamine, Methylphenidate': 3,
    'Dextroamphetamine (Dexedrine)': 4,
    'Amphetamine/dextroamphetamine salt (Adderall)': 5,
}

