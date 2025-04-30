"""
Input validation utilities.

This module provides comprehensive validation functions for all system inputs,
ensuring robustness against invalid or unexpected inputs.
"""

import os
import re
import math
import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from utils.error_handling import (
    ImageFormatError, ImageLoadError, SudokuRecognizerError, InvalidPuzzleError
)

# Define common types
ImageType = np.ndarray  # OpenCV image (numpy array)
PointType = Tuple[int, int]  # (x, y) coordinates
GridType = List[List[int]]  # 9x9 grid of integers (0 = empty)


def validate_file_exists(file_path: str) -> bool:
    """
    Validate that a file exists and is readable.
    
    Args:
        file_path: Path to file to check
        
    Returns:
        True if file exists and is readable
        
    Raises:
        SudokuRecognizerError: If file doesn't exist or isn't readable
    """
    if not file_path:
        raise SudokuRecognizerError("File path is empty")
        
    if not os.path.exists(file_path):
        raise SudokuRecognizerError(f"File not found: {file_path}")
        
    if not os.path.isfile(file_path):
        raise SudokuRecognizerError(f"Not a file: {file_path}")
        
    if not os.access(file_path, os.R_OK):
        raise SudokuRecognizerError(f"File not readable: {file_path}")
        
    return True


def validate_image_file(file_path: str, allowed_extensions: Optional[List[str]] = None) -> bool:
    """
    Validate that a file is a valid image file.
    
    Args:
        file_path: Path to image file
        allowed_extensions: List of allowed file extensions
        
    Returns:
        True if file is a valid image file
        
    Raises:
        ImageFormatError: If file is not a valid image file
    """
    # Validate file exists
    validate_file_exists(file_path)
    
    # Validate file extension
    if allowed_extensions:
        ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        if ext not in allowed_extensions:
            raise ImageFormatError(
                f"Unsupported image format: {ext}. Supported formats: {', '.join(allowed_extensions)}"
            )
    
    try:
        # Try to read image
        img = cv2.imread(file_path)
        if img is None:
            raise ImageLoadError(f"Failed to load image: {file_path}")
            
        # Validate image dimensions
        if img.shape[0] < 10 or img.shape[1] < 10:
            raise ImageFormatError(f"Image too small: {img.shape[1]}x{img.shape[0]}")
            
        return True
    except Exception as e:
        if isinstance(e, (ImageFormatError, ImageLoadError)):
            raise
        raise ImageLoadError(f"Error loading image {file_path}: {str(e)}")


def validate_image(image: ImageType) -> bool:
    """
    Validate that an image is valid.
    
    Args:
        image: OpenCV image
        
    Returns:
        True if image is valid
        
    Raises:
        ImageFormatError: If image is invalid
    """
    if image is None:
        raise ImageFormatError("Image is None")
        
    if not isinstance(image, np.ndarray):
        raise ImageFormatError(f"Invalid image type: {type(image).__name__}")
        
    if image.size == 0:
        raise ImageFormatError("Empty image")
        
    if len(image.shape) < 2:
        raise ImageFormatError(f"Invalid image shape: {image.shape}")
        
    if image.shape[0] < 10 or image.shape[1] < 10:
        raise ImageFormatError(f"Image too small: {image.shape[1]}x{image.shape[0]}")
        
    return True


def validate_grid_size(grid: GridType) -> bool:
    """
    Validate Sudoku grid size (should be 9x9).
    
    Args:
        grid: Sudoku grid
        
    Returns:
        True if grid size is valid
        
    Raises:
        InvalidPuzzleError: If grid size is invalid
    """
    if not grid:
        raise InvalidPuzzleError("Empty grid")
        
    if len(grid) != 9:
        raise InvalidPuzzleError(f"Invalid grid rows: {len(grid)}, expected 9")
        
    for i, row in enumerate(grid):
        if len(row) != 9:
            raise InvalidPuzzleError(f"Invalid grid columns in row {i}: {len(row)}, expected 9")
            
    return True


def validate_grid_values(grid: GridType) -> bool:
    """
    Validate that a Sudoku grid contains only valid values (0-9).
    
    Args:
        grid: Sudoku grid
        
    Returns:
        True if grid values are valid
        
    Raises:
        InvalidPuzzleError: If grid contains invalid values
    """
    validate_grid_size(grid)
    
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            if not isinstance(value, int):
                raise InvalidPuzzleError(f"Non-integer value at position ({i}, {j}): {value}")
                
            if value < 0 or value > 9:
                raise InvalidPuzzleError(f"Invalid value at position ({i}, {j}): {value}")
                
    return True


def validate_sudoku_rules(grid: GridType) -> bool:
    """
    Validate that a Sudoku grid follows Sudoku rules (no duplicates).
    
    Args:
        grid: Sudoku grid
        
    Returns:
        True if grid follows Sudoku rules
        
    Raises:
        InvalidPuzzleError: If grid violates Sudoku rules
    """
    validate_grid_values(grid)
    
    # Check rows
    for i, row in enumerate(grid):
        row_values = [val for val in row if val != 0]
        if len(row_values) != len(set(row_values)):
            raise InvalidPuzzleError(f"Duplicate values in row {i}")
            
    # Check columns
    for j in range(9):
        col_values = [grid[i][j] for i in range(9) if grid[i][j] != 0]
        if len(col_values) != len(set(col_values)):
            raise InvalidPuzzleError(f"Duplicate values in column {j}")
            
    # Check 3x3 boxes
    for box_i in range(3):
        for box_j in range(3):
            box_values = []
            for i in range(3):
                for j in range(3):
                    val = grid[box_i*3 + i][box_j*3 + j]
                    if val != 0:
                        box_values.append(val)
            
            if len(box_values) != len(set(box_values)):
                raise InvalidPuzzleError(f"Duplicate values in box ({box_i}, {box_j})")
                
    return True


def validate_points(points: List[PointType], image_shape: Tuple[int, int]) -> bool:
    """
    Validate that points are within image boundaries.
    
    Args:
        points: List of (x, y) points
        image_shape: Image dimensions (height, width)
        
    Returns:
        True if all points are valid
        
    Raises:
        SudokuRecognizerError: If points are invalid
    """
    if not points:
        raise SudokuRecognizerError("Empty points list")
        
    height, width = image_shape[:2]
    
    for i, (x, y) in enumerate(points):
        if not (0 <= x < width and 0 <= y < height):
            raise SudokuRecognizerError(
                f"Point {i} ({x}, {y}) is outside image boundaries (width={width}, height={height})"
            )
            
    return True


def validate_homography_matrix(matrix: np.ndarray) -> bool:
    """
    Validate that a matrix is a valid homography matrix.
    
    Args:
        matrix: 3x3 homography matrix
        
    Returns:
        True if matrix is valid
        
    Raises:
        SudokuRecognizerError: If matrix is invalid
    """
    if matrix is None:
        raise SudokuRecognizerError("Homography matrix is None")
        
    if not isinstance(matrix, np.ndarray):
        raise SudokuRecognizerError(f"Invalid matrix type: {type(matrix).__name__}")
        
    if matrix.shape != (3, 3):
        raise SudokuRecognizerError(f"Invalid matrix shape: {matrix.shape}, expected (3, 3)")
        
    # Check if the matrix is invertible
    if abs(np.linalg.det(matrix)) < 1e-6:
        raise SudokuRecognizerError("Homography matrix is singular (non-invertible)")
        
    return True


def validate_cell_image(cell_image: ImageType) -> bool:
    """
    Validate that a cell image is valid for digit recognition.
    
    Args:
        cell_image: Cell image
        
    Returns:
        True if cell image is valid
        
    Raises:
        ImageFormatError: If cell image is invalid
    """
    validate_image(cell_image)
    
    # Check if image is too small for reliable digit recognition
    if cell_image.shape[0] < 10 or cell_image.shape[1] < 10:
        raise ImageFormatError(f"Cell image too small: {cell_image.shape[1]}x{cell_image.shape[0]}")
        
    # For grayscale images, check if there's enough contrast
    if len(cell_image.shape) == 2 or cell_image.shape[2] == 1:
        min_val = np.min(cell_image)
        max_val = np.max(cell_image)
        if max_val - min_val < 20:
            # Not raising an error here, as low contrast might be valid for empty cells
            return False
            
    return True


def is_valid_intersection_point(point: PointType, image_shape: Tuple[int, int], min_distance: int = 5) -> bool:
    """
    Check if a point is a valid intersection point.
    
    Args:
        point: (x, y) coordinates
        image_shape: Image dimensions (height, width)
        min_distance: Minimum distance from image borders
        
    Returns:
        True if point is valid
    """
    x, y = point
    height, width = image_shape[:2]
    
    # Check if point is within image boundaries with minimum distance from borders
    return (
        min_distance <= x < width - min_distance and
        min_distance <= y < height - min_distance
    )


def is_valid_grid(
    grid_points: List[PointType],
    image_shape: Tuple[int, int],
    min_points: int = 60,
    min_coverage: float = 0.3
) -> bool:
    """
    Check if detected grid points form a valid grid.
    
    Args:
        grid_points: List of grid intersection points
        image_shape: Image dimensions (height, width)
        min_points: Minimum number of grid points
        min_coverage: Minimum image area coverage fraction
        
    Returns:
        True if grid is valid
    """
    if len(grid_points) < min_points:
        return False
        
    # Calculate bounding box
    x_vals = [p[0] for p in grid_points]
    y_vals = [p[1] for p in grid_points]
    
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    
    grid_width = max_x - min_x
    grid_height = max_y - min_y
    
    # Check if grid covers enough of the image
    img_height, img_width = image_shape[:2]
    img_area = img_width * img_height
    grid_area = grid_width * grid_height
    
    coverage = grid_area / img_area
    
    return coverage >= min_coverage


def is_confidence_sufficient(confidences: List[float], threshold: float = 0.7) -> bool:
    """
    Check if confidence scores are high enough.
    
    Args:
        confidences: List of confidence scores
        threshold: Minimum average confidence threshold
        
    Returns:
        True if confidence is sufficient
    """
    if not confidences:
        return False
        
    avg_confidence = sum(confidences) / len(confidences)
    return avg_confidence >= threshold


def is_puzzle_solvable(grid: GridType) -> bool:
    """
    Check if a Sudoku puzzle is solvable.
    
    Args:
        grid: Sudoku grid
        
    Returns:
        True if puzzle is solvable
    """
    try:
        validate_sudoku_rules(grid)
        return True
    except InvalidPuzzleError:
        return False


def normalize_image_size(
    image: ImageType,
    min_size: int = 300,
    max_size: int = 1600
) -> Tuple[ImageType, float]:
    """
    Normalize image size within bounds, preserving aspect ratio.
    
    Args:
        image: Input image
        min_size: Minimum dimension size
        max_size: Maximum dimension size
        
    Returns:
        Tuple of (normalized image, scale factor)
    """
    validate_image(image)
    
    height, width = image.shape[:2]
    
    # Calculate scale factors
    scale_up = min_size / min(height, width)
    scale_down = max_size / max(height, width)
    
    # Determine final scale
    if min(height, width) < min_size:
        scale = scale_up
    elif max(height, width) > max_size:
        scale = scale_down
    else:
        # No scaling needed
        return image, 1.0
        
    # Apply scaling
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    
    return resized, scale
