#!/usr/bin/env python3
"""
Check Data Script

Verifies that processed data files exist and can be loaded correctly,
performing basic checks on the loaded data format and properties.
"""

import os
import numpy as np
import logging
import sys

# Configure basic logging for the check script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Just print to console
)
logger = logging.getLogger(__name__)

# --- Configuration ---
PROCESSED_DIR = "data/processed"
EXPECTED_FILES = {
    "train_digits": ["train_digits_data.npy", "train_digits_labels.npy"],
    "test_digits": ["test_digits_data.npy", "test_digits_labels.npy"],
    # Add other datasets if needed (e.g., intersections)
    # "train_intersections": ["train_intersections_data.npy", "train_intersections_points.npy"],
}
EXPECTED_IMAGE_SHAPE = (28, 28, 1) # Expected shape for each image sample (H, W, C)
EXPECTED_DTYPE = np.float32
EXPECTED_MIN_VALUE = 0.0
EXPECTED_MAX_VALUE = 1.0
# --- End Configuration ---

def check_file_existence(base_dir, dataset_name, file_list):
    """Checks if all expected files for a dataset exist."""
    logger.info(f"--- Checking file existence for '{dataset_name}' dataset ---")
    all_exist = True
    for filename in file_list:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            logger.info(f"  [ OK ] Found: {path}")
        else:
            logger.error(f"  [FAIL] Missing: {path}")
            all_exist = False
    if not all_exist:
         logger.error(f"Missing required files for '{dataset_name}'. Run data preparation first.")
    return all_exist

def check_loaded_data(data, labels, dataset_name):
    """Performs checks on loaded data and labels."""
    logger.info(f"--- Checking loaded data for '{dataset_name}' dataset ---")
    passed = True

    # Check types
    if not isinstance(data, np.ndarray):
        logger.error(f"  [FAIL] Data is not a numpy array (type: {type(data)})")
        passed = False
    else:
        logger.info(f"  [ OK ] Data is a numpy array.")

    if not isinstance(labels, np.ndarray):
        logger.error(f"  [FAIL] Labels is not a numpy array (type: {type(labels)})")
        passed = False
    else:
         logger.info(f"  [ OK ] Labels is a numpy array.")

    # If types are wrong, further checks might fail, so return early
    if not passed:
        return False

    # Check number of samples match
    if data.shape[0] != labels.shape[0]:
        logger.error(f"  [FAIL] Mismatch in number of samples: Data={data.shape[0]}, Labels={labels.shape[0]}")
        passed = False
    else:
        logger.info(f"  [ OK ] Number of samples match ({data.shape[0]}).")

    # Check data shape (assuming images)
    # Check shape of the first sample, ignoring batch dimension
    if data.shape[1:] != EXPECTED_IMAGE_SHAPE:
        logger.warning(f"  [WARN] Data shape {data.shape[1:]} does not match expected {EXPECTED_IMAGE_SHAPE}. Check if channel dim is missing or size is different.")
        # Don't mark as fail, but warn
    else:
        logger.info(f"  [ OK ] Data shape matches expected {EXPECTED_IMAGE_SHAPE} (excluding batch size).")

    # Check data type
    if data.dtype != EXPECTED_DTYPE:
        logger.warning(f"  [WARN] Data dtype is {data.dtype}, expected {EXPECTED_DTYPE}. Normalization might be needed.")
        # Don't mark as fail, but warn
    else:
        logger.info(f"  [ OK ] Data dtype is {EXPECTED_DTYPE}.")

    # Check data range (only if float)
    if np.issubdtype(data.dtype, np.floating):
        min_val, max_val = np.min(data), np.max(data)
        if not (EXPECTED_MIN_VALUE <= min_val <= max_val <= EXPECTED_MAX_VALUE):
             logger.warning(f"  [WARN] Data range [{min_val:.2f}, {max_val:.2f}] is outside expected [{EXPECTED_MIN_VALUE}, {EXPECTED_MAX_VALUE}]. Check normalization.")
             # Don't mark as fail, but warn
        else:
             logger.info(f"  [ OK ] Data range [{min_val:.2f}, {max_val:.2f}] is within expected [{EXPECTED_MIN_VALUE}, {EXPECTED_MAX_VALUE}].")
    else:
        logger.info(f"  Skipping range check for non-float dtype ({data.dtype}).")

    # Check label shape (should be 1D)
    if labels.ndim != 1:
        logger.error(f"  [FAIL] Labels array is not 1-dimensional (shape: {labels.shape})")
        passed = False
    else:
        logger.info(f"  [ OK ] Labels array is 1-dimensional.")

    # Check label range (assuming 0-9 digits)
    min_label, max_label = np.min(labels), np.max(labels)
    if not (0 <= min_label <= max_label <= 9):
         logger.warning(f"  [WARN] Label range [{min_label}, {max_label}] seems unusual for digits 0-9.")
         # Don't mark as fail, but warn
    else:
         logger.info(f"  [ OK ] Label range [{min_label}, {max_label}] seems reasonable.")

    return passed


def main():
    """Main function to run checks."""
    logger.info("Starting data verification script...")

    # Dynamically import load_dataset from the project structure
    try:
        # Assumes check_data.py is in the project root alongside 'utils'
        from utils.data_preparation import load_dataset
        logger.info("Successfully imported 'load_dataset' from utils.data_preparation")
    except ImportError as e:
        logger.error(f"Failed to import 'load_dataset': {e}")
        logger.error("Ensure this script is run from the project root directory and 'utils/data_preparation.py' exists.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during import: {e}")
        sys.exit(1)

    overall_success = True

    for dataset_name, file_list in EXPECTED_FILES.items():
        logger.info(f"\n=== Verifying Dataset: {dataset_name} ===")

        # 1. Check file existence
        if not check_file_existence(PROCESSED_DIR, dataset_name, file_list):
            overall_success = False
            continue # Skip loading if files are missing

        # 2. Attempt to load data
        logger.info(f"--- Attempting to load '{dataset_name}' using load_dataset ---")
        try:
            data, labels = load_dataset(PROCESSED_DIR, dataset_name)
            logger.info(f"  Successfully loaded '{dataset_name}'.")
        except FileNotFoundError:
             logger.error(f"  [FAIL] load_dataset failed: FileNotFoundError. Check implementation or paths.")
             overall_success = False
             continue
        except Exception as e:
            logger.error(f"  [FAIL] load_dataset failed for '{dataset_name}': {e}", exc_info=True)
            overall_success = False
            continue

        # 3. Check loaded data properties
        if not check_loaded_data(data, labels, dataset_name):
            overall_success = False
            logger.error(f"Checks failed for '{dataset_name}'. Please review warnings/errors above.")
        else:
             logger.info(f"All checks passed for '{dataset_name}'.")

    logger.info("\n=== Verification Summary ===")
    if overall_success:
        logger.info("[ SUCCESS ] All checks passed successfully!")
        sys.exit(0)
    else:
        logger.error("[ FAILED ] Some checks failed. Please review the log messages.")
        sys.exit(1)


if __name__ == "__main__":
    main()
