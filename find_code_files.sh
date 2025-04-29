#!/bin/bash

# Script to find non-empty files (excluding images) within the project root
# and its immediate subdirectories (max depth = 1 level down),
# outputting only the base filenames.

# Define the target project root directory
ROOT_DIR="$HOME/git2/Git_soduku"

# --- Safety Check ---
if [ ! -d "$ROOT_DIR" ]; then
  echo "ERROR: Root directory '$ROOT_DIR' not found."
  exit 1
fi

echo "Searching for non-empty files (excluding images) in '$ROOT_DIR' and its immediate subdirectories..."
echo "Outputting only filenames."
echo "------------------------------------------"

# Find files within ROOT_DIR that are:
# -maxdepth 2 => Look in ROOT_DIR itself (depth 1) and only one level down
#                into its direct subdirectories (depth 2).
# -type f     => Must be a regular file.
# -size +0    => Size must be greater than 0 bytes (not empty).
# -not \( ... \)=> EXCLUDE common image extensions (case-insensitive).
#                 The '\( ... \)' groups the '-o' (OR) conditions.
# -printf '%f\n'=> Print ONLY the base filename component of the path,
#                 followed by a newline.

find "$ROOT_DIR" -maxdepth 2 -type f -size +0 \
    -not \( \
        -iname '*.jpg' -o \
        -iname '*.jpeg' -o \
        -iname '*.png' -o \
        -iname '*.gif' -o \
        -iname '*.bmp' -o \
        -iname '*.svg' -o \
        -iname '*.webp' -o \
        -iname '*.ico' -o \
        -iname '*.tiff' \
    \) \
    -printf '%f\n'

echo "------------------------------------------"
echo "Search complete."

exit 0
