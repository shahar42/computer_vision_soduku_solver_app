# utils/validation.py
"""
Validation functions for Sudoku processing pipeline.
"""

import numpy as np
import logging
from typing import List, Union

# Configure logging
logger = logging.getLogger(__name__)

# Define types
ImageType = np.ndarray
GridType = List[List[int]]

def validate_cell_image(cell: Union[ImageType, None], min_std_dev: float = 5.0, min_size: int = 5) -> bool:
    """
    Performs basic validation on an image patch intended to be a Sudoku cell.

    Checks if the input is a valid NumPy array, has a minimum size, and
    has a standard deviation above a threshold (to filter out blank patches).

    Args:
        cell: The input image (NumPy array) or None.
        min_std_dev: Minimum standard deviation of pixel values required
                     to be considered potentially non-blank. Adjust based on
                     image normalization (e.g., use a smaller value for 0-1 floats).
        min_size: Minimum number of pixels required (height * width).

    Returns:
        True if the cell image seems valid, False otherwise.
    """
    if cell is None:
        logger.debug("Cell image is None, validation failed.")
        return False

    if not isinstance(cell, np.ndarray):
        logger.debug(f"Cell image is not a NumPy array (type: {type(cell)}), validation failed.")
        return False

    if cell.size < min_size * min_size:
        logger.debug(f"Cell image size ({cell.shape}) is too small (< {min_size}x{min_size}), validation failed.")
        return False

    # Check if the image has some variation (not just a solid color/blank)
    # Adjust threshold based on expected pixel value range (e.g., 0-255 or 0-1)
    std_dev = np.std(cell)
    if std_dev < min_std_dev:
        logger.debug(f"Cell image standard deviation ({std_dev:.2f}) is below threshold ({min_std_dev}), likely blank. Validation failed.")
        return False

    # If all checks pass
    return True

def validate_image(image: Union[ImageType, None], min_size: int = 10) -> None:
    """
    Performs basic validation on an input image.

    Checks if the input is a valid NumPy array and has a minimum size.

    Args:
        image: The input image (NumPy array) or None.
        min_size: Minimum required height and width.

    Raises:
        ValueError: If the image is invalid.
    """
    if image is None:
        msg = "Input image cannot be None."
        logger.error(msg)
        raise ValueError(msg)

    if not isinstance(image, np.ndarray):
        msg = f"Input image must be a NumPy array, got {type(image)}."
        logger.error(msg)
        raise ValueError(msg)

    if image.ndim < 2 or image.shape[0] < min_size or image.shape[1] < min_size:
        msg = f"Input image dimensions ({image.shape}) are too small or invalid (min required: {min_size}x{min_size})."
        logger.error(msg)
        raise ValueError(msg)

    # Add any other basic checks if needed (e.g., max size, specific dtype)
    logger.debug("Input image passed basic validation.")

def validate_grid_values(grid: GridType) -> None:
    """
    Validate that the grid contains only integers between 0 and 9.

    Args:
        grid: 9x9 grid representing the Sudoku puzzle.

    Raises:
        ValueError: If the grid has invalid dimensions or contains invalid values.
    """
    if not isinstance(grid, list) or len(grid) != 9 or not all(isinstance(row, list) and len(row) == 9 for row in grid):
        raise ValueError("Invalid grid dimensions. Must be a 9x9 list of lists.")

    for r in range(9):
        for c in range(9):
            val = grid[r][c]
            if not isinstance(val, int) or not (0 <= val <= 9):
                raise ValueError(f"Invalid value '{val}' at grid position ({r}, {c}). Must be integer 0-9.")

# --- Add any other validation functions needed below ---
# Example: validate_sudoku_rules (if needed elsewhere, currently defined in solver.py)
# def validate_sudoku_rules(grid: GridType) -> None:
#     # Implementation to check row/col/box uniqueness
#     pass
