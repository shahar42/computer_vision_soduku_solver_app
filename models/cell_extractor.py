"""
Cell Extractor Module.

This module implements cell extraction from a Sudoku grid image with robust
error handling and image enhancement techniques.
"""

import os
import numpy as np
import cv2
import pickle
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from . import CellExtractorBase
from config.settings import get_settings
from utils.error_handling import (
    CellExtractionError, retry, fallback, robust_method, safe_execute
)
from utils.validation import validate_image, validate_points, validate_cell_image

# Define types
ImageType = np.ndarray
PointType = Tuple[int, int]
GridPointsType = List[List[PointType]]

# Configure logging
logger = logging.getLogger(__name__)

class PerspectiveCellExtractor(CellExtractorBase):
    """
    Perspective transform-based cell extractor with minimal processing.

    This class extracts cell images from a Sudoku grid using perspective
    transformation to correct distortion and normalize cell size while
    preserving digit features.
    """

    def __init__(self):
        """Initialize perspective cell extractor with gentle parameters."""
        self.settings = get_settings().get_nested("cell_extractor")

        # Cell parameters
        self.cell_size = self.settings.get("cell_size", 28)
        self.border_padding = self.settings.get("border_padding", 0.07)  # Minimal border padding

        # Processing flags - most disabled by default for better feature preservation
        self.perspective_correction = self.settings.get("perspective_correction", True)
        self.contrast_enhancement = self.settings.get("contrast_enhancement", False)  # Disabled
        self.noise_reduction = self.settings.get("noise_reduction", False)  # Disabled
        self.adaptive_thresholding = self.settings.get("adaptive_thresholding", False)  # Disabled
        self.histogram_equalization = self.settings.get("histogram_equalization", False)  # Disabled


        # Extraction mode - 'preserve' keeps original grayscale values
        self.extraction_mode = self.settings.get("extraction_mode", "preserve")

    def load(self, model_path: str) -> bool:
        """Load model parameters from file."""
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    params = pickle.load(f)

                # Update parameters
                for key, value in params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

                logger.info(f"Loaded perspective cell extractor parameters from {model_path}")
                return True
            else:
                logger.warning(f"Parameter file {model_path} not found, using defaults")
                return False

        except Exception as e:
            logger.error(f"Error loading perspective cell extractor parameters: {str(e)}")
            return False

    def save(self, model_path: str) -> bool:
        """Save model parameters to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Collect parameters
            params = {
                'cell_size': self.cell_size,
                'border_padding': self.border_padding,
                'perspective_correction': self.perspective_correction,
                'contrast_enhancement': self.contrast_enhancement,
                'noise_reduction': self.noise_reduction,
                'adaptive_thresholding': self.adaptive_thresholding,
                'histogram_equalization': self.histogram_equalization,
                'extraction_mode': self.extraction_mode
            }

            # Save parameters
            with open(model_path, 'wb') as f:
                pickle.dump(params, f)

            logger.info(f"Saved perspective cell extractor parameters to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving perspective cell extractor parameters: {str(e)}")
            return False

    @robust_method(max_retries=2, timeout_sec=30.0)
    def extract(self, image: ImageType, grid_points: GridPointsType) -> List[List[ImageType]]:
        """
        Extract cell images from grid with minimal processing.

        Args:
            image: Original image
            grid_points: 2D list of ordered grid points

        Returns:
            2D list of cell images (9x9 for standard Sudoku)
        """
        try:
            # Validate inputs
            validate_image(image)

            if len(grid_points) != 10 or any(len(row) != 10 for row in grid_points):
                raise CellExtractionError(
                    f"Invalid grid points: {len(grid_points)} rows, expected 10"
                )

            # Convert to grayscale if necessary
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply very minimal global preprocessing
            if self.noise_reduction:
                # Use smallest possible kernel for noise reduction
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # Extract cells
            cell_images = []
            for i in range(9):  # 9 rows
                row_cells = []
                for j in range(9):  # 9 columns
                    # Define the four corners of this cell
                    corners = [
                        grid_points[i][j],      # Top-left
                        grid_points[i][j+1],    # Top-right
                        grid_points[i+1][j+1],  # Bottom-right
                        grid_points[i+1][j]     # Bottom-left
                    ]

                    # Extract cell with minimal processing
                    try:
                        cell = self._extract_cell(gray, corners)
                        row_cells.append(cell)
                    except Exception as e:
                        logger.warning(f"Error extracting cell ({i},{j}): {str(e)}")
                        # Create an empty cell as fallback
                        empty_cell = np.zeros((self.cell_size, self.cell_size), dtype=np.uint8)
                        row_cells.append(empty_cell)

                cell_images.append(row_cells)

            return cell_images

        except Exception as e:
            if isinstance(e, CellExtractionError):
                raise
            raise CellExtractionError(f"Error in cell extraction: {str(e)}")

    def _extract_cell(self, image: ImageType, corners: List[PointType]) -> ImageType:
        """
        Extract a single cell using perspective transform with minimal processing.

        Args:
            image: Grayscale image
            corners: Four corners of the cell [top-left, top-right, bottom-right, bottom-left]

        Returns:
            Extracted cell image with digit features preserved
        """
        # Check if all corners are valid
        invalid_corners = [p for p in corners if p[0] < 0 or p[1] < 0]
        if invalid_corners:
            raise ValueError(f"Invalid corners: {invalid_corners}")

        # Convert corners to numpy array
        corners_array = np.array(corners, dtype=np.float32)

        # Calculate output size
        output_size = self.cell_size

        # Define output corners (destination points)
        dst_corners = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype=np.float32)

        # Calculate perspective transform matrix
        if self.perspective_correction:
            transform_matrix = cv2.getPerspectiveTransform(corners_array, dst_corners)

            # Apply perspective transformation
            cell = cv2.warpPerspective(image, transform_matrix, (output_size, output_size))
        else:
            # Use simpler affine transform
            transform_matrix = cv2.getAffineTransform(
                corners_array[:3],  # Only need 3 points for affine
                dst_corners[:3]     # Only need 3 points for affine
            )

            # Apply affine transformation
            cell = cv2.warpAffine(image, transform_matrix, (output_size, output_size))

        # Apply border removal (very minimal)
        if self.border_padding > 0:
            padding = int(output_size * self.border_padding)
            if padding > 0:
                # Compute inner borders with minimal padding
                start_idx = padding
                end_idx = output_size - padding
                # Extract the center portion
                cell = cell[start_idx:end_idx, start_idx:end_idx]
                # Resize back to output_size
                cell = cv2.resize(cell, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

        # Apply optional enhancements based on extraction mode
        if self.extraction_mode == "preserve":
            # Return the cell with minimal processing
            return cell
        elif self.extraction_mode == "enhance":
            # Apply gentle enhancements
            return self._enhance_cell(cell)
        elif self.extraction_mode == "threshold":
            # Apply thresholding for binary output
            return self._threshold_cell(cell)
        else:
            # Default to preserve mode
            return cell

    def _enhance_cell(self, cell: ImageType) -> ImageType:
        """Apply gentle enhancements to improve digit visibility."""
        # Make a copy to avoid modifying the original
        enhanced = cell.copy()

        # Calculate basic statistics
        cell_min, cell_max = np.min(enhanced), np.max(enhanced)
        cell_contrast = cell_max - cell_min

        # Only apply enhancements if there's enough contrast
        if cell_contrast > 20:
            # Apply histogram normalization (stretches contrast, preserves details)
            if self.histogram_equalization:
                enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

            # Apply very gentle contrast enhancement
            if self.contrast_enhancement:
                alpha = 1.1  # Slight contrast increase
                beta = -5    # Slight brightness decrease
                enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

        return enhanced

    def _threshold_cell(self, cell: ImageType) -> ImageType:
        """Apply thresholding for binary output (only when specifically requested)."""
        # Make a copy to avoid modifying the original
        binary = cell.copy()

        # Calculate basic statistics
        cell_min, cell_max = np.min(binary), np.max(binary)
        cell_contrast = cell_max - cell_min

        # Only apply thresholding if there's enough contrast
        if cell_contrast > 20:
            if self.adaptive_thresholding:
                # Use larger block size and constant for gentler thresholding
                binary = cv2.adaptiveThreshold(
                    binary,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    21,  # Large block size
                    5    # Higher constant
                )
            else:
                # Use Otsu's method for global thresholding
                _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return binary


class CannyEdgeCellExtractor(CellExtractorBase):
    """
    Canny edge-based cell extractor.

    This class extracts cell images using edge detection to identify
    digit boundaries more reliably.
    """

    def __init__(self):
        """Initialize Canny edge cell extractor with default parameters."""
        self.settings = get_settings().get_nested("cell_extractor")

        # Cell parameters
        self.cell_size = self.settings.get("cell_size", 28)
        self.border_padding = self.settings.get("border_padding", 0.07)

        # Edge detection parameters
        self.canny_low = 50
        self.canny_high = 150

    def load(self, model_path: str) -> bool:
        """
        Load model parameters from file.

        Args:
            model_path: Path to parameter file

        Returns:
            True if successful
        """
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    params = pickle.load(f)

                # Update parameters
                for key, value in params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

                logger.info(f"Loaded Canny edge cell extractor parameters from {model_path}")
                return True
            else:
                logger.warning(f"Parameter file {model_path} not found, using defaults")
                return False

        except Exception as e:
            logger.error(f"Error loading Canny edge cell extractor parameters: {str(e)}")
            return False

    def save(self, model_path: str) -> bool:
        """
        Save model parameters to file.

        Args:
            model_path: Path to save parameter file

        Returns:
            True if successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Collect parameters
            params = {
                'cell_size': self.cell_size,
                'border_padding': self.border_padding,
                'canny_low': self.canny_low,
                'canny_high': self.canny_high
            }

            # Save parameters
            with open(model_path, 'wb') as f:
                pickle.dump(params, f)

            logger.info(f"Saved Canny edge cell extractor parameters to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving Canny edge cell extractor parameters: {str(e)}")
            return False

    @robust_method(max_retries=2, timeout_sec=30.0)
    def extract(self, image: ImageType, grid_points: GridPointsType) -> List[List[ImageType]]:
        """
        Extract cell images from grid using edge detection.

        Args:
            image: Original image
            grid_points: 2D list of ordered grid points

        Returns:
            2D list of cell images (9x9 for standard Sudoku)

        Raises:
            CellExtractionError: If extraction fails
        """
        try:
            # Validate inputs
            validate_image(image)

            if len(grid_points) != 10 or any(len(row) != 10 for row in grid_points):
                raise CellExtractionError(
                    f"Invalid grid points: {len(grid_points)} rows, expected 10"
                )

            # Convert to grayscale if necessary
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply global preprocessing
            # Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply Canny edge detection
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

            # Dilate edges to connect discontinuities
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)

            # Extract cells
            cell_images = []
            for i in range(9):  # 9 rows
                row_cells = []
                for j in range(9):  # 9 columns
                    # Define the four corners of this cell
                    corners = [
                        grid_points[i][j],      # Top-left
                        grid_points[i][j+1],    # Top-right
                        grid_points[i+1][j+1],  # Bottom-right
                        grid_points[i+1][j]     # Bottom-left
                    ]

                    # Extract and process cell
                    try:
                        # Extract cell from both original and edge images
                        cell_gray = self._extract_cell_patch(gray, corners)
                        cell_edges = self._extract_cell_patch(dilated_edges, corners)

                        # Combine information from both sources
                        cell = self._process_cell(cell_gray, cell_edges)
                        row_cells.append(cell)
                    except Exception as e:
                        logger.warning(f"Error extracting cell ({i},{j}): {str(e)}")
                        # Create an empty cell as fallback
                        empty_cell = np.zeros((self.cell_size, self.cell_size), dtype=np.uint8)
                        row_cells.append(empty_cell)

                cell_images.append(row_cells)

            return cell_images

        except Exception as e:
            if isinstance(e, CellExtractionError):
                raise
            raise CellExtractionError(f"Error in edge-based cell extraction: {str(e)}")

    def _extract_cell_patch(self, image: ImageType, corners: List[PointType]) -> ImageType:
        """
        Extract a single cell patch using perspective transform.

        Args:
            image: Input image
            corners: Four corners of the cell [top-left, top-right, bottom-right, bottom-left]

        Returns:
            Extracted cell patch
        """
        # Check if all corners are valid
        invalid_corners = [p for p in corners if p[0] < 0 or p[1] < 0]
        if invalid_corners:
            raise ValueError(f"Invalid corners: {invalid_corners}")

        # Convert corners to numpy array
        corners_array = np.array(corners, dtype=np.float32)

        # Calculate output size
        output_size = self.cell_size

        # Define output corners (destination points)
        dst_corners = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype=np.float32)

        # Calculate perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(corners_array, dst_corners)

        # Apply perspective transformation
        cell = cv2.warpPerspective(image, transform_matrix, (output_size, output_size), flags=cv2.INTER_CUBIC)

        # Apply border removal
        if self.border_padding > 0:
            padding = int(output_size * self.border_padding)
            if padding > 0:
                cell = cell[padding:-padding, padding:-padding]
                cell = cv2.resize(cell, (output_size, output_size))

        return cell

    def _process_cell(self, cell_gray: ImageType, cell_edges: ImageType) -> ImageType:
        """
        Process cell images to enhance digit visibility.

        Args:
            cell_gray: Grayscale cell image
            cell_edges: Edge-detected cell image

        Returns:
            Processed cell image
        """
        # If edge image has significant edges, use it to enhance grayscale image
        edge_pixels = np.count_nonzero(cell_edges)
        edge_ratio = edge_pixels / (cell_edges.shape[0] * cell_edges.shape[1])

        if edge_ratio > 0.01 and edge_ratio < 0.2:  # Reasonable amount of edges
            # Threshold grayscale image
            _, thresh = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Combine thresholded image with edges
            combined = cv2.bitwise_or(thresh, cell_edges)

            # Clean up combined image
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

            return cleaned
        else:
            # If edge detection didn't work well, use adaptive thresholding
            cell_min, cell_max = np.min(cell_gray), np.max(cell_gray)
            if cell_max - cell_min > 30:  # Only apply if there's reasonable contrast
                return cv2.adaptiveThreshold(
                    cell_gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    11,
                    2
                )
            else:
                # If low contrast, use Otsu's thresholding
                _, thresh = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                return thresh


class RobustCellExtractor(CellExtractorBase):
    """
    Robust cell extractor with multiple methods and fallback mechanisms.

    This class combines perspective and edge-based extractors for
    robustness, with intelligent method selection and fallback strategies.
    """

    def __init__(self):
        """Initialize robust cell extractor with multiple methods."""
        self.settings = get_settings().get_nested("cell_extractor")

        # Initialize extractors
        self.perspective_extractor = PerspectiveCellExtractor()
        self.edge_extractor = CannyEdgeCellExtractor()

        # Settings
        self.use_multiple_extractors = self.settings.get("use_multiple_extractors", False)
        self.cell_size = self.settings.get("cell_size", 28)

    def load(self, model_path: str) -> bool:
        """
        Load models from files.

        Args:
            model_path: Base path for model files

        Returns:
            True if at least one model was loaded successfully
        """
        # Determine model paths
        perspective_path = os.path.splitext(model_path)[0] + "_perspective.pkl"
        edge_path = os.path.splitext(model_path)[0] + "_edge.pkl"

        # Load models
        perspective_loaded = self.perspective_extractor.load(perspective_path)
        edge_loaded = self.edge_extractor.load(edge_path)

        # Log results
        if perspective_loaded:
            logger.info("Perspective cell extractor loaded successfully")
        else:
            logger.warning("Failed to load perspective cell extractor")

        if edge_loaded:
            logger.info("Edge cell extractor loaded successfully")
        else:
            logger.warning("Failed to load edge cell extractor")

        # Return True if at least one model was loaded
        return perspective_loaded or edge_loaded

    def save(self, model_path: str) -> bool:
        """
        Save models to files.

        Args:
            model_path: Base path for model files

        Returns:
            True if both models were saved successfully
        """
        # Determine model paths
        perspective_path = os.path.splitext(model_path)[0] + "_perspective.pkl"
        edge_path = os.path.splitext(model_path)[0] + "_edge.pkl"

        # Save models
        perspective_saved = self.perspective_extractor.save(perspective_path)
        edge_saved = self.edge_extractor.save(edge_path)

        # Log results
        if perspective_saved:
            logger.info(f"Perspective cell extractor saved to {perspective_path}")
        else:
            logger.warning(f"Failed to save perspective cell extractor to {perspective_path}")

        if edge_saved:
            logger.info(f"Edge cell extractor saved to {edge_path}")
        else:
            logger.warning(f"Failed to save edge cell extractor to {edge_path}")

        # Return True only if both models were saved
        return perspective_saved and edge_saved

    @robust_method(max_retries=3, timeout_sec=60.0)
    def extract(self, image: ImageType, grid_points: GridPointsType) -> List[List[ImageType]]:
        """
        Extract cell images from grid using multiple methods with fallback.

        Args:
            image: Original image
            grid_points: 2D list of ordered grid points

        Returns:
            2D list of cell images (9x9 for standard Sudoku)

        Raises:
            CellExtractionError: If all extraction methods fail
        """
        try:
            # Validate inputs
            validate_image(image)

            if len(grid_points) != 10 or any(len(row) != 10 for row in grid_points):
                raise CellExtractionError(
                    f"Invalid grid points: {len(grid_points)} rows, expected 10"
                )

            # If not using multiple extractors, just use perspective method
            if not self.use_multiple_extractors:
                return self.perspective_extractor.extract(image, grid_points)

            # Try perspective method first
            perspective_cells = None
            edge_cells = None

            try:
                perspective_cells = self.perspective_extractor.extract(image, grid_points)
            except Exception as e:
                logger.warning(f"Perspective extraction failed: {str(e)}")

            # Try edge method
            try:
                edge_cells = self.edge_extractor.extract(image, grid_points)
            except Exception as e:
                logger.warning(f"Edge extraction failed: {str(e)}")

            # If both methods failed, use fallback method
            if perspective_cells is None and edge_cells is None:
                logger.warning("Both extraction methods failed, using fallback")
                return self._extract_fallback(image, grid_points)

            # If only one method succeeded, return its results
            if perspective_cells is None:
                return edge_cells # type: ignore
            if edge_cells is None:
                return perspective_cells # type: ignore

            # If both methods succeeded, select the best cells from each
            return self._combine_extraction_results(perspective_cells, edge_cells)

        except Exception as e:
            if isinstance(e, CellExtractionError):
                raise
            raise CellExtractionError(f"Error in robust cell extraction: {str(e)}")

    def _extract_fallback(self, image: ImageType, grid_points: GridPointsType) -> List[List[ImageType]]:
        """
        Fallback cell extraction method.

        Args:
            image: Original image
            grid_points: 2D list of ordered grid points

        Returns:
            2D list of cell images
        """
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Create empty cells
        cell_images = []
        for i in range(9):  # 9 rows
            row_cells = []
            for j in range(9):  # 9 columns
                try:
                    # Define the four corners of this cell
                    corners = [
                        grid_points[i][j],      # Top-left
                        grid_points[i][j+1],    # Top-right
                        grid_points[i+1][j+1],  # Bottom-right
                        grid_points[i+1][j]     # Bottom-left
                    ]

                    # Simple extraction using direct warping
                    corners_array = np.array(corners, dtype=np.float32)

                    # Define output corners
                    dst_corners = np.array([
                        [0, 0],
                        [self.cell_size - 1, 0],
                        [self.cell_size - 1, self.cell_size - 1],
                        [0, self.cell_size - 1]
                    ], dtype=np.float32)

                    # Calculate perspective transform matrix
                    transform_matrix = cv2.getPerspectiveTransform(corners_array, dst_corners)

                    # Apply perspective transformation
                    cell = cv2.warpPerspective(gray, transform_matrix, (self.cell_size, self.cell_size))

                    # Apply simple thresholding
                    _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                    row_cells.append(cell)
                except Exception as e:
                    logger.warning(f"Fallback extraction failed for cell ({i},{j}): {str(e)}")
                    # Create an empty cell
                    empty_cell = np.zeros((self.cell_size, self.cell_size), dtype=np.uint8)
                    row_cells.append(empty_cell)

            cell_images.append(row_cells)

        return cell_images

    def _combine_extraction_results(
        self,
        cells1: List[List[ImageType]],
        cells2: List[List[ImageType]]
    ) -> List[List[ImageType]]:
        """
        Combine extraction results from multiple methods.

        Args:
            cells1: Cell images from first method
            cells2: Cell images from second method

        Returns:
            Combined cell images
        """
        # Validate input dimensions
        if (len(cells1) != 9 or len(cells2) != 9 or
            any(len(row) != 9 for row in cells1) or
            any(len(row) != 9 for row in cells2)):
            raise ValueError("Invalid cell dimensions")

        # Combine cell images by selecting the best one for each cell
        combined_cells = []
        for i in range(9):
            row_cells = []
            for j in range(9):
                cell1 = cells1[i][j]
                cell2 = cells2[i][j]

                # Use cell quality metrics to determine which is better
                quality1 = self._cell_quality(cell1)
                quality2 = self._cell_quality(cell2)

                # Use the cell with higher quality
                if quality1 >= quality2:
                    row_cells.append(cell1)
                else:
                    row_cells.append(cell2)

            combined_cells.append(row_cells)

        return combined_cells

    def _cell_quality(self, cell: ImageType) -> float:
        """
        Calculate quality metric for a cell image with preference for grayscale.

        Args:
            cell: Cell image

        Returns:
            Quality score (higher is better)
        """
        # If cell is empty or invalid, return low score
        if cell is None or cell.size == 0:
            return 0.0

        # Check if cell is binary (just black and white)
        unique_values = len(np.unique(cell))
        if unique_values <= 2:
            # Penalize binary images to prefer grayscale
            return 0.5

        # Calculate various quality metrics for grayscale images

        # Contrast: difference between max and min values
        min_val, max_val = np.min(cell), np.max(cell)
        contrast = (max_val - min_val) / 255.0

        # Sharpness: variance of Laplacian
        laplacian = cv2.Laplacian(cell, cv2.CV_64F)
        sharpness = np.var(laplacian) / 10000.0  # Normalize

        # Noise: inverse of standard deviation in homogeneous regions
        # Get the most common value (background)
        hist = cv2.calcHist([cell], [0], None, [256], [0, 256])
        background_val = np.argmax(hist)

        # Create mask of background pixels
        mask = np.abs(cell.astype(np.float32) - background_val) < 10
        if np.sum(mask) > 0:
            # Calculate standard deviation in background regions
            noise = 1.0 - min(np.std(cell[mask]) / 30.0, 1.0)
        else:
            noise = 0.5  # Default if no background found

        # Balance of gray levels (for grayscale images)
        white_ratio = np.sum(cell > 127) / cell.size
        balance = 1.0 - abs(white_ratio - 0.2) / 0.2  # Optimal ~20% white

        # Combine metrics with weights
        weights = [0.3, 0.25, 0.2, 0.25]  # Contrast, sharpness, noise, balance
        score = (contrast * weights[0] +
                 sharpness * weights[1] +
                 noise * weights[2] +
                 balance * weights[3])

        # Boost grayscale score
        return min(score * 1.5, 1.0)  # Ensure score is at most 1.0
