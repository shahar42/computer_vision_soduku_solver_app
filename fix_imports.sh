#!/bin/bash

# Script to fix specific relative imports in data_preparation.py

TARGET_FILE="utils/data_preparation.py"

# --- Safety Check ---
# Check if the target file exists before proceeding
if [ ! -f "$TARGET_FILE" ]; then
  echo "ERROR: Target file '$TARGET_FILE' not found."
  echo "Make sure you are in the 'Git_soduku' root directory."
  exit 1
fi

echo "Applying changes to $TARGET_FILE..."

# --- Perform the substitutions ---

# Change 'from ..config' to 'from config'
# Using '#' as separator because paths might contain '/'
sed -i 's#from \.\.config import get_settings#from config import get_settings#' "$TARGET_FILE"

# Change 'from ..utils.error_handling' to 'from utils.error_handling'
sed -i 's#from \.\.utils\.error_handling import SudokuRecognizerError#from utils.error_handling import SudokuRecognizerError#' "$TARGET_FILE"

# --- Confirmation ---
echo "Done. Relative imports should be fixed."
echo "Verify the changes in $TARGET_FILE if needed."

exit 0
