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

# --- MODULE VERSION PRINT STATEMENT ---
print("--- Loading Custom version do power!! Cell Extractor Module (Version: INTER_CUBIC Test on original structure) ---")
print(f"--- Timestamp: {__import__('datetime').datetime.now()} ---")


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
# Ensure basic logging is configured if not done elsewhere in your project entry point
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class PerspectiveCellExtractor(CellExtractorBase):
    """
    Perspective transform-based cell extractor with minimal processing.

    This class extracts cell images from a Sudoku grid using perspective
    transformation to correct distortion and normalize cell size while
    preserving digit features.
    """

    def __init__(self):
        """Initialize perspective cell extractor with gentle parameters."""
        print("PerspectiveCellExtractor: Initializing... (INTER_CUBIC Test Version - from original structure)")
        self.settings = get_settings().get_nested("cell_extractor")

        # Cell parameters
        self.cell_size = self.settings.get("cell_size", 28)
        self.border_padding = self.settings.get("border_padding", 0.07)  # This will be overridden by your patch script

        # Processing flags - most disabled by default for better feature preservation
        self.perspective_correction = self.settings.get("perspective_correction", True)
        self.contrast_enhancement = self.settings.get("contrast_enhancement", False)  # Disabled
        self.noise_reduction = self.settings.get("noise_reduction", False)  # Disabled
        self.adaptive_thresholding = self.settings.get("adaptive_thresholding", False)  # Disabled
        self.histogram_equalization = self.settings.get("histogram_equalization", False)  # Disabled

        # Extraction mode - 'preserve' keeps original grayscale values
        self.extraction_mode = self.settings.get("extraction_mode", "preserve")
        print(f"PerspectiveCellExtractor: Initial settings: cell_size={self.cell_size}, border_padding={self.border_padding}, perspective_correction={self.perspective_correction}, extraction_mode='{self.extraction_mode}'")


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
        # Your patch script will modify self.border_padding, self.extraction_mode, etc. on the instance
        # logger.info(f"PerspectiveCellExtractor.extract called. Current border_padding: {self.border_padding}, extraction_mode: {self.extraction_mode}")
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
            if self.noise_reduction: # This will be false due to your patch
                # logger.info("PerspectiveCellExtractor.extract: Applying GaussianBlur for noise reduction.")
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

                    # --- ADD THIS IF CONDITION ---
                    if j == 4: # Check if it's the 5th column (0-indexed)
                        print(f"DEBUG: Corners for cell ({i},{j}) in PerspectiveCellExtractor: {corners}")
                    # --- END OF ADDITION ---

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

                print(f"DEBUG: PerspectiveCellExtractor.extract - Total rows in cell_images: {len(cell_images)}")
                for idx, r_cells in enumerate(cell_images):
                    print(f"DEBUG: PerspectiveCellExtractor.extract - Row {idx} has {len(r_cells)} cells.")
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

        interpolation_flag_to_use = cv2.INTER_CUBIC
        interpolation_method_name = "INTER_CUBIC" # For the print statement

        # Calculate perspective transform matrix
        if self.perspective_correction: # This will be true by default, and not changed by your patch
            print(f"PerspectiveCellExtractor._extract_cell: Applying cv2.warpPerspective with flags={interpolation_method_name}")
            transform_matrix = cv2.getPerspectiveTransform(corners_array, dst_corners)
            # Apply perspective transformation
            cell = cv2.warpPerspective(image, transform_matrix, (output_size, output_size), flags=interpolation_flag_to_use) # MODIFIED HERE
        else:
            print(f"PerspectiveCellExtractor._extract_cell: Applying cv2.warpAffine with flags={interpolation_method_name}")
            # Use simpler affine transform
            transform_matrix = cv2.getAffineTransform(
                corners_array[:3],  # Only need 3 points for affine
                dst_corners[:3]     # Only need 3 points for affine
            )
            # Apply affine transformation
            cell = cv2.warpAffine(image, transform_matrix, (output_size, output_size), flags=interpolation_flag_to_use) # MODIFIED HERE

        # Apply border removal (very minimal) - your patch sets self.border_padding to 0.01
        if self.border_padding > 0:
            padding = int(output_size * self.border_padding)
            if padding > 0:
                # Compute inner borders with minimal padding
                start_idx = padding
                end_idx = output_size - padding
                # Ensure indices are valid and do not result in an empty slice
                if start_idx < end_idx and start_idx >= 0 and end_idx <= output_size :
                    cell_cropped = cell[start_idx:end_idx, start_idx:end_idx]
                    # Ensure the cropped cell is not empty before resizing
                    if cell_cropped.size > 0:
                        cell = cv2.resize(cell_cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
                    # else: If cropping results in an empty image, 'cell' remains the uncropped warped image
                # else: If padding is too large, 'cell' remains the uncropped warped image

        # Apply optional enhancements based on extraction mode - your patch sets mode to "preserve"
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
        if cell_contrast > 20: # This will be false due to your patch
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
        if cell_contrast > 20: # This will be false due to your patch
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
        self.canny_low = self.settings.get("canny_low", 50) # Using get() for these specific params
        self.canny_high = self.settings.get("canny_high", 150)


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
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            params = {
                'cell_size': self.cell_size,
                'border_padding': self.border_padding,
                'canny_low': self.canny_low,
                'canny_high': self.canny_high
            }
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
            validate_image(image)
            if len(grid_points) != 10 or any(len(row) != 10 for row in grid_points):
                raise CellExtractionError(f"Invalid grid points: {len(grid_points)} rows, expected 10")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)

            cell_images = []
            for i in range(9):
                row_cells = []
                for j in range(9):
                    corners = [
                        grid_points[i][j], grid_points[i][j+1],
                        grid_points[i+1][j+1], grid_points[i+1][j]
                    ]
                    try:
                        cell_gray = self._extract_cell_patch(gray, corners)
                        cell_edges = self._extract_cell_patch(dilated_edges, corners)
                        cell = self._process_cell(cell_gray, cell_edges)
                        row_cells.append(cell)
                    except Exception as e:
                        logger.warning(f"Error extracting cell ({i},{j}) in Canny: {str(e)}")
                        empty_cell = np.zeros((self.cell_size, self.cell_size), dtype=np.uint8)
                        row_cells.append(empty_cell)
                cell_images.append(row_cells)
            return cell_images
        except Exception as e:
            if isinstance(e, CellExtractionError): raise
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
        invalid_corners = [p for p in corners if p[0] < 0 or p[1] < 0]
        if invalid_corners:
            raise ValueError(f"Invalid corners: {invalid_corners}")

        corners_array = np.array(corners, dtype=np.float32)
        output_size = self.cell_size
        dst_corners = np.array([
            [0, 0], [output_size - 1, 0],
            [output_size - 1, output_size - 1], [0, output_size - 1]
        ], dtype=np.float32)

        transform_matrix = cv2.getPerspectiveTransform(corners_array, dst_corners)
        # This already had INTER_CUBIC in your original, so I'm preserving it.
        cell = cv2.warpPerspective(image, transform_matrix, (output_size, output_size), flags=cv2.INTER_CUBIC)

        if self.border_padding > 0:
            padding = int(output_size * self.border_padding)
            if padding > 0:
                if output_size - 2 * padding > 0:
                    cell_cropped = cell[padding:output_size-padding, padding:output_size-padding]
                    if cell_cropped.size > 0:
                        cell = cv2.resize(cell_cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR) # Original interpolation for resize
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
        edge_pixels = np.count_nonzero(cell_edges)
        edge_ratio = edge_pixels / (cell_edges.size if cell_edges.size > 0 else 1.0) # Avoid division by zero

        if 0.01 < edge_ratio < 0.2:  # Reasonable amount of edges
            _, thresh = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            combined = cv2.bitwise_or(thresh, cell_edges)
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            return cleaned
        else:
            cell_min, cell_max = np.min(cell_gray), np.max(cell_gray)
            if cell_max - cell_min > 30:  # Only apply if there's reasonable contrast
                return cv2.adaptiveThreshold(
                    cell_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
            else:
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
        self.perspective_extractor = PerspectiveCellExtractor()
        self.edge_extractor = CannyEdgeCellExtractor()
        self.use_multiple_extractors = self.settings.get("use_multiple_extractors", False)
        self.cell_size = self.settings.get("cell_size", 28)

    def load(self, model_path: str) -> bool:
        perspective_path = os.path.splitext(model_path)[0] + "_perspective.pkl"
        edge_path = os.path.splitext(model_path)[0] + "_edge.pkl"
        perspective_loaded = self.perspective_extractor.load(perspective_path)
        edge_loaded = self.edge_extractor.load(edge_path)
        if perspective_loaded: logger.info("Perspective cell extractor loaded successfully")
        else: logger.warning("Failed to load perspective cell extractor")
        if edge_loaded: logger.info("Edge cell extractor loaded successfully")
        else: logger.warning("Failed to load edge cell extractor")
        return perspective_loaded or edge_loaded

    def save(self, model_path: str) -> bool:
        perspective_path = os.path.splitext(model_path)[0] + "_perspective.pkl"
        edge_path = os.path.splitext(model_path)[0] + "_edge.pkl"
        perspective_saved = self.perspective_extractor.save(perspective_path)
        edge_saved = self.edge_extractor.save(edge_path)
        if perspective_saved: logger.info(f"Perspective cell extractor saved to {perspective_path}")
        else: logger.warning(f"Failed to save perspective cell extractor to {perspective_path}")
        if edge_saved: logger.info(f"Edge cell extractor saved to {edge_path}")
        else: logger.warning(f"Failed to save edge cell extractor to {edge_path}")
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
        # Your patch script will force self.use_multiple_extractors = False
        # and directly call self.perspective_extractor.extract(...)
        try:
            validate_image(image)
            if len(grid_points) != 10 or any(len(row) != 10 for row in grid_points):
                raise CellExtractionError(f"Invalid grid points: {len(grid_points)} rows, expected 10")

            if not self.use_multiple_extractors: # This will be True due to your monkey patch
                return self.perspective_extractor.extract(image, grid_points)

            # Fallback logic if multiple extractors were to be used (not hit by your current patch)
            perspective_cells, edge_cells = None, None
            try:
                perspective_cells = self.perspective_extractor.extract(image, grid_points)
            except Exception as e: logger.warning(f"Perspective extraction failed: {str(e)}")
            try:
                edge_cells = self.edge_extractor.extract(image, grid_points)
            except Exception as e: logger.warning(f"Edge extraction failed: {str(e)}")

            if perspective_cells is None and edge_cells is None:
                logger.warning("Both extraction methods failed, using fallback")
                return self._extract_fallback(image, grid_points)
            if perspective_cells is None: return cast(List[List[ImageType]], edge_cells)
            if edge_cells is None: return cast(List[List[ImageType]], perspective_cells)

            return self._combine_extraction_results(perspective_cells, edge_cells)
        except Exception as e:
            if isinstance(e, CellExtractionError): raise
            raise CellExtractionError(f"Error in robust cell extraction: {str(e)}")

    def _extract_fallback(self, image: ImageType, grid_points: GridPointsType) -> List[List[ImageType]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        cell_images = []
        for i in range(9):
            row_cells = []
            for j in range(9):
                try:
                    corners = [
                        grid_points[i][j], grid_points[i][j+1],
                        grid_points[i+1][j+1], grid_points[i+1][j]
                    ]
                    corners_array = np.array(corners, dtype=np.float32)
                    dst_corners = np.array([
                        [0,0], [self.cell_size-1,0],
                        [self.cell_size-1,self.cell_size-1], [0,self.cell_size-1]
                    ], dtype=np.float32)
                    transform_matrix = cv2.getPerspectiveTransform(corners_array, dst_corners)
                    cell = cv2.warpPerspective(gray, transform_matrix, (self.cell_size, self.cell_size)) # Default interpolation for fallback
                    _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    row_cells.append(cell)
                except Exception as e:
                    logger.warning(f"Fallback extraction failed for cell ({i},{j}): {str(e)}")
                    empty_cell = np.zeros((self.cell_size, self.cell_size), dtype=np.uint8)
                    row_cells.append(empty_cell)
            cell_images.append(row_cells)
        return cell_images

    def _combine_extraction_results(
        self, cells1: List[List[ImageType]], cells2: List[List[ImageType]]
    ) -> List[List[ImageType]]:
        if not (len(cells1) == 9 and len(cells2) == 9 and
                all(len(row) == 9 for row in cells1) and
                all(len(row) == 9 for row in cells2)):
            raise ValueError("Invalid cell dimensions")

        combined_cells = []
        for i in range(9):
            row_cells = []
            for j in range(9):
                cell1, cell2 = cells1[i][j], cells2[i][j]
                quality1, quality2 = self._cell_quality(cell1), self._cell_quality(cell2)
                row_cells.append(cell1 if quality1 >= quality2 else cell2)
            combined_cells.append(row_cells)
        return combined_cells

    def _cell_quality(self, cell: ImageType) -> float:
        if cell is None or cell.size == 0: return 0.0
        if len(np.unique(cell)) <= 2: return 0.5

        min_val, max_val = np.min(cell), np.max(cell)
        contrast = (max_val - min_val) / 255.0
        laplacian = cv2.Laplacian(cell, cv2.CV_64F)
        sharpness = np.var(laplacian) / 10000.0
        hist = cv2.calcHist([cell], [0], None, [256], [0, 256])
        background_val = np.argmax(hist)
        mask = np.abs(cell.astype(np.float32) - background_val) < 10
        noise = 1.0 - min(np.std(cell[mask]) / 30.0, 1.0) if np.sum(mask) > 0 else 0.5
        white_ratio = np.sum(cell > 127) / (cell.size if cell.size > 0 else 1.0) # Avoid division by zero
        balance = 1.0 - abs(white_ratio - 0.2) / 0.2
        weights = [0.3, 0.25, 0.2, 0.25]
        score = sum(m * w for m, w in zip([contrast, sharpness, noise, balance], weights))
        return min(score * 1.5, 1.0)
