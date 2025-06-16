"""
Sudoku Recognizer Pipeline Module. good version

This module implements the main pipeline that orchestrates all components
with robust error handling, recovery strategies, and fallback mechanisms.
"""

import os
import cv2
import numpy as np
import logging
import time
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from models.intersection_detector import RobustIntersectionDetector
from models.grid_reconstructor import RobustGridReconstructor
from models.cell_extractor import RobustCellExtractor
from models.digit_recognizer import RobustDigitRecognizer
from models.solver import RobustSolver
from models.board_detector import BoardDetector
from config.settings import get_settings
from utils.error_handling import (
    SudokuRecognizerError, ImageLoadError, DetectionError, IntersectionDetectionError,
    GridReconstructionError, CellExtractionError, DigitRecognitionError, SolverError,
    InvalidPuzzleError, PipelineError, RecoveryResult, get_error_handler,
    retry, fallback, robust_method, safe_execute
)
from utils.validation import (
    validate_file_exists, validate_image_file, validate_image,
    normalize_image_size, validate_grid_values, is_puzzle_solvable
)

from utils.tf_compatibility import setup_tensorflow_compatibility
# Define types
ImageType = np.ndarray
PointType = Tuple[int, int]
GridPointsType = List[List[PointType]]
GridType = List[List[int]]

# Configure logging
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SudokuRecognizerPipeline:
    """
    Main pipeline for Sudoku recognition.

    This class orchestrates all components of the Sudoku recognition system
    with robust error handling, recovery strategies, and fallback mechanisms.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Sudoku recognizer pipeline.

        Args:
            config_path: Path to configuration file (optional)
        """
        # Load settings
        self.settings = get_settings()
        setup_tensorflow_compatibility()
        self.board_detector = None
        self.use_board_detection = True

        # Pipeline settings
        self.pipeline_settings = self.settings.get_nested("pipeline")
        self.save_intermediates = self.pipeline_settings.get("save_intermediates", False)
        self.parallel_processing = self.pipeline_settings.get("parallel_processing", True)
        self.max_workers = self.pipeline_settings.get("max_workers", 4)
        self.retry_failed_stages = self.pipeline_settings.get("retry_failed_stages", True)
        self.verification_at_each_stage = self.pipeline_settings.get("verification_at_each_stage", True)

        # Initialize components
        self.intersection_detector = RobustIntersectionDetector()
        self.grid_reconstructor = RobustGridReconstructor()
        self.cell_extractor = RobustCellExtractor()
        self.digit_recognizer = RobustDigitRecognizer()
        self.solver = RobustSolver()

        # States and results
        self.current_state = {
            "image": None,
            "intersections": None,
            "grid_points": None,
            "cell_images": None,
            "digit_grid": None,
            "confidence_grid": None,
            "solved_grid": None
        }

        self.error_handler = get_error_handler()

        # Register recovery strategies
        self._register_recovery_strategies()

    def _register_recovery_strategies(self) -> None:
        """Register recovery strategies for different error types."""
        # Grid detection recovery strategies
        self.error_handler.register_recovery(
            IntersectionDetectionError,
            self._recover_intersection_detection_by_enhancing_contrast
        )
        self.error_handler.register_recovery(
            IntersectionDetectionError,
            self._recover_intersection_detection_by_trying_alternative_method
        )

        # Grid reconstruction recovery strategies
        self.error_handler.register_recovery(
            GridReconstructionError,
            self._recover_grid_reconstruction_by_relaxing_parameters
        )
        self.error_handler.register_recovery(
            GridReconstructionError,
            self._recover_grid_reconstruction_by_trying_alternative_method
        )

        # Cell extraction recovery strategies
        self.error_handler.register_recovery(
            CellExtractionError,
            self._recover_cell_extraction_by_refining_grid
        )
        self.error_handler.register_recovery(
            CellExtractionError,
            self._recover_cell_extraction_by_adjusting_cell_boundaries
        )

        # Digit recognition recovery strategies
        self.error_handler.register_recovery(
            DigitRecognitionError,
            self._recover_digit_recognition_by_enhancing_contrast
        )
        self.error_handler.register_recovery(
            DigitRecognitionError,
            self._recover_digit_recognition_by_trying_alternative_model
        )

        # Solving recovery strategies
        self.error_handler.register_recovery(
            SolverError,
            self._recover_solving_by_validating_digits
        )
        self.error_handler.register_recovery(
            SolverError,
            self._recover_solving_by_relaxing_constraints
        )
        self.error_handler.register_recovery(
            InvalidPuzzleError,
            self._recover_solving_by_correcting_invalid_puzzle
        )

    def load_models(self, model_dir: str) -> bool:
        """
        Load all models with improved error handling and compatibility.
        
        Args:
            model_dir: Directory containing model files
            
        Returns:
            True if critical models were loaded successfully
        """
        try:
            # Create model paths
            intersection_model_path = os.path.join(model_dir, "intersection_detector.h5")
            digit_model_path = os.path.join(model_dir, "digit_recognizer.h5")
            board_model_path = os.path.join(model_dir, "board_detector.h5")
            
            # Track loading success
            models_loaded = {
                'intersection': False,
                'digit': False,
                'board': False
            }
            
            # Load intersection detector (critical)
            try:
                models_loaded['intersection'] = self.intersection_detector.load(intersection_model_path)
                if models_loaded['intersection']:
                    logger.info("✅ Intersection detector loaded successfully")
                else:
                    logger.warning("⚠️ Intersection detector failed to load")
            except Exception as e:
                logger.error(f"❌ Intersection detector loading error: {str(e)}")
            
            # Load digit recognizer (critical)
            try:
                models_loaded['digit'] = self.digit_recognizer.load(digit_model_path)
                if models_loaded['digit']:
                    logger.info("✅ Digit recognizer loaded successfully")
                else:
                    logger.warning("⚠️ Digit recognizer failed to load")
            except Exception as e:
                logger.error(f"❌ Digit recognizer loading error: {str(e)}")
            
            # Load board detector (optional but recommended)
            try:
                if os.path.exists(board_model_path):
                    from models.board_detector import BoardDetector
                    self.board_detector = BoardDetector(board_model_path)
                    models_loaded['board'] = True
                    logger.info("✅ Board detector loaded successfully")
                    self.use_board_detection = True
                else:
                    logger.info("📝 Board detector not found, proceeding without it")
                    self.use_board_detection = False
            except Exception as e:
                logger.warning(f"⚠️ Board detector loading error: {str(e)}")
                self.use_board_detection = False
            
            # Load other models (they have fallback mechanisms)
            try:
                grid_model_path = os.path.join(model_dir, "grid_reconstructor.pkl")
                self.grid_reconstructor.load(grid_model_path)
                logger.info("📝 Grid reconstructor parameters loaded (optional)")
            except Exception as e:
                logger.info("📝 Grid reconstructor using default parameters")
                
            try:
                cell_model_path = os.path.join(model_dir, "cell_extractor.pkl")
                self.cell_extractor.load(cell_model_path)
                logger.info("📝 Cell extractor parameters loaded (optional)")
            except Exception as e:
                logger.info("📝 Cell extractor using default parameters")
            
            # Determine success criteria
            critical_models_loaded = models_loaded['intersection'] or models_loaded['digit']
            
            if models_loaded['intersection'] and models_loaded['digit']:
                logger.info("🎉 All critical models loaded successfully!")
                return True
            elif critical_models_loaded:
                logger.warning("⚠️ Some critical models loaded, pipeline will use fallbacks")
                return True
            else:
                logger.error("❌ No critical models loaded, pipeline may not work properly")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in model loading process: {str(e)}")
            return False
    @robust_method(max_retries=2, timeout_sec=60.0)
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a Sudoku image to detect and solve the puzzle.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with results from each pipeline stage

        Raises:
            PipelineError: If processing fails
        """
        try:
            # Validate input file
            validate_file_exists(image_path)
            validate_image_file(image_path)

            # Start timing
            start_time = time.time()

            # Reset current state
            self.current_state = {
                "image": None,
                "intersections": None,
                "grid_points": None,
                "cell_images": None,
                "digit_grid": None,
                "confidence_grid": None,
                "solved_grid": None
            }

            # Load image
            try:
                image = cv2.imread(image_path)
                if image is None:
                    raise ImageLoadError(f"Failed to load image: {image_path}")

                # Validate image
                validate_image(image)

                # Store in current state
                self.current_state["image"] = image
                logger.info(f"Image loaded: {image_path}")

            except Exception as e:
                logger.error(f"Error loading image: {str(e)}")
                raise ImageLoadError(f"Failed to load image: {str(e)}")

            # Execute pipeline stages
            stages = self.pipeline_settings.get("stages", [
                "detect_grid", "extract_cells", "recognize_digits", "solve"
            ])

            results = {}

            for stage in stages:
                try:
                    if stage == "detect_grid":
                        results["grid_detection"] = self._detect_grid()
                    elif stage == "extract_cells":
                        results["cell_extraction"] = self._extract_cells()
                    elif stage == "recognize_digits":
                        results["digit_recognition"] = self._recognize_digits()
                    elif stage == "solve":
                        results["solving"] = self._solve_puzzle()
                    else:
                        logger.warning(f"Unknown pipeline stage: {stage}")

                except Exception as e:
                    logger.error(f"Error in pipeline stage {stage}: {str(e)}")

                    # Try to recover
                    if self.retry_failed_stages:
                        context = {"stage": stage, "image_path": image_path}
                        recovery_result = self.error_handler.handle_error(e, context)

                        if recovery_result.success:
                            logger.info(f"Successfully recovered from error in stage {stage}")
                            results[f"{stage}_recovered"] = True
                        else:
                            logger.error(f"Failed to recover from error in stage {stage}")
                            results[f"{stage}_error"] = str(e)

                            # If this is a critical stage that we can't proceed without,
                            # raise an error
                            if stage in ["detect_grid", "extract_cells"]:
                                raise PipelineError(f"Critical stage {stage} failed: {str(e)}")
                    else:
                        results[f"{stage}_error"] = str(e)

                        # If this is a critical stage that we can't proceed without,
                        # raise an error
                        if stage in ["detect_grid", "extract_cells"]:
                            raise PipelineError(f"Critical stage {stage} failed: {str(e)}")

            # Calculate processing time
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time

            logger.info(f"Image processed successfully in {processing_time:.2f} seconds")

            # Prepare final results
            final_results = {
                "success": True,
                "processing_time": processing_time,
                "grid_detected": self.current_state["grid_points"] is not None,
                "cells_extracted": self.current_state["cell_images"] is not None,
                "digits_recognized": self.current_state["digit_grid"] is not None,
                "puzzle_solved": self.current_state["solved_grid"] is not None,
                "digit_grid": self.current_state["digit_grid"],
                "confidence_grid": self.current_state["confidence_grid"],
                "solved_grid": self.current_state["solved_grid"],
                "stage_results": results
            }

            return final_results

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")

            if isinstance(e, PipelineError):
                # Rethrow pipeline errors
                raise
            else:
                # Wrap other errors
                raise PipelineError(f"Failed to process image: {str(e)}")

    def process_image_data(self, image_data: ImageType) -> Dict[str, Any]:
        """
        Process a Sudoku image from image data.

        Args:
            image_data: Image data as numpy array

        Returns:
            Dictionary with results from each pipeline stage

        Raises:
            PipelineError: If processing fails
        """
        try:
            # Validate input image
            validate_image(image_data)

            # Start timing
            start_time = time.time()

            # Reset current state
            self.current_state = {
                "image": None,
                "intersections": None,
                "grid_points": None,
                "cell_images": None,
                "digit_grid": None,
                "confidence_grid": None,
                "solved_grid": None
            }

            # Store image in current state
            self.current_state["image"] = image_data
            logger.info("Image data loaded")

            # Execute pipeline stages
            stages = self.pipeline_settings.get("stages", [
                "detect_grid", "extract_cells", "recognize_digits", "solve"
            ])

            results = {}

            for stage in stages:
                try:
                    if stage == "detect_grid":
                        results["grid_detection"] = self._detect_grid()
                    elif stage == "extract_cells":
                        results["cell_extraction"] = self._extract_cells()
                    elif stage == "recognize_digits":
                        results["digit_recognition"] = self._recognize_digits()
                    elif stage == "solve":
                        results["solving"] = self._solve_puzzle()
                    else:
                        logger.warning(f"Unknown pipeline stage: {stage}")

                except Exception as e:
                    logger.error(f"Error in pipeline stage {stage}: {str(e)}")

                    # Try to recover
                    if self.retry_failed_stages:
                        context = {"stage": stage}
                        recovery_result = self.error_handler.handle_error(e, context)

                        if recovery_result.success:
                            logger.info(f"Successfully recovered from error in stage {stage}")
                            results[f"{stage}_recovered"] = True
                        else:
                            logger.error(f"Failed to recover from error in stage {stage}")
                            results[f"{stage}_error"] = str(e)

                            # If this is a critical stage that we can't proceed without,
                            # raise an error
                            if stage in ["detect_grid", "extract_cells"]:
                                raise PipelineError(f"Critical stage {stage} failed: {str(e)}")
                    else:
                        results[f"{stage}_error"] = str(e)

                        # If this is a critical stage that we can't proceed without,
                        # raise an error
                        if stage in ["detect_grid", "extract_cells"]:
                            raise PipelineError(f"Critical stage {stage} failed: {str(e)}")

            # Calculate processing time
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time

            logger.info(f"Image processed successfully in {processing_time:.2f} seconds")

            # Prepare final results
            final_results = {
                "success": True,
                "processing_time": processing_time,
                "grid_detected": self.current_state["grid_points"] is not None,
                "cells_extracted": self.current_state["cell_images"] is not None,
                "digits_recognized": self.current_state["digit_grid"] is not None,
                "puzzle_solved": self.current_state["solved_grid"] is not None,
                "digit_grid": self.current_state["digit_grid"],
                "confidence_grid": self.current_state["confidence_grid"],
                "solved_grid": self.current_state["solved_grid"],
                "stage_results": results
            }

            return final_results

        except Exception as e:
            logger.error(f"Error processing image data: {str(e)}")

            if isinstance(e, PipelineError):
                # Rethrow pipeline errors
                raise
            else:
                # Wrap other errors
                raise PipelineError(f"Failed to process image data: {str(e)}")

    def _detect_grid(self) -> Dict[str, Any]:
        """
        Detect Sudoku grid in image using board detection when available.
        Now includes minimum intersections per line reconstruction method.

        Returns:
            Dictionary with grid detection results
        """
        try:
            # Get image from current state
            image = self.current_state["image"]
            if image is None:
                raise PipelineError("No image to detect grid in")

            # Initialize variables
            board_detected = False
            board_bbox = None
            filtered_intersections = None

            # Step 1: Try board detection if available
            if self.use_board_detection and self.board_detector is not None:
                try:
                    logger.info("🎯 Running board detection...")
                    detection_result = self.board_detector.detect(image)
                    if detection_result is not None:
                        x1, y1, x2, y2, confidence = detection_result
                        board_bbox = (x1, y1, x2, y2)
                    else:
                        board_bbox = None
                        confidence = 0.0
                    if board_bbox is not None and confidence >= 0.5:
                        logger.info(f"✅ Board detected with confidence {confidence:.3f}")
                        board_detected = True
                    else:
                        logger.info(f"Board detection failed or low confidence ({confidence:.3f}), falling back to intersection-only method")
                except Exception as e:
                    logger.warning(f"Board detection error: {str(e)}. Falling back to intersection-only method")

            # In pipeline.py, after board detection:
            if board_bbox is not None:
                x1, y1, x2, y2 = board_bbox
                board_width = x2 - x1
                board_height = y2 - y1
                board_area_ratio = (board_width * board_height) / (image.shape[0] * image.shape[1])
                
                logger.info(f"Board dimensions: {board_width}x{board_height}")
                logger.info(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
                logger.info(f"Board covers {board_area_ratio:.1%} of image")
                
                # If board covers more than 90% of image, something's wrong
                if board_area_ratio > 0.9:
                    logger.warning("Board detection may be incorrect - covers too much of image")
            # Step 2: Detect intersections (always needed)
            intersections = self.intersection_detector.detect(image)

            # Verify we have enough intersections
            if len(intersections) < 20:
                raise IntersectionDetectionError(
                    f"Insufficient intersections detected: {len(intersections)}"
                )

            logger.info(f"Detected {len(intersections)} grid intersections")

            # Store in current state
            self.current_state["intersections"] = intersections

            # Step 3: Filter intersections if board was detected
            if board_detected and board_bbox is not None:
                x1, y1, x2, y2 = board_bbox
                # Calculate board diagonal length
                diagonal = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                # Calculate margin (diagonal/14 as per requirements)
                margin = diagonal / 14

                # Expand bounding box by margin
                x1_expanded = max(0, x1 - margin)
                y1_expanded = max(0, y1 - margin)
                x2_expanded = min(image.shape[1] - 1, x2 + margin)
                y2_expanded = min(image.shape[0] - 1, y2 + margin)

                # Filter intersections to those within expanded bbox
                filtered_intersections = []
                for point in intersections:
                    x, y = point
                    if (x1_expanded <= x <= x2_expanded and y1_expanded <= y <= y2_expanded):
                        filtered_intersections.append(point)

                logger.info(f"Filtered intersections: {len(intersections)} → {len(filtered_intersections)} (kept {len(filtered_intersections)/len(intersections)*100:.1f}%)")

                # Check if we have enough filtered intersections
                if len(filtered_intersections) < 78:  # As per requirements
                    logger.warning(f"Too few filtered intersections ({len(filtered_intersections)}), falling back to all intersections")
                    filtered_intersections = None

            # Step 4: Reconstruct grid with new intelligent method selection
            grid_points = None
            reconstruction_methods = self.settings.get_nested("grid_reconstructor").get(
                "reconstruction_method_order",
                ["min_intersections", "board_detection", "standard"]
            )

            for method in reconstruction_methods:
                try:
                    if method == "min_intersections" and board_bbox is not None:
                        # Use the new minimum intersections per line method
                        use_min_intersections = self.settings.get_nested("grid_reconstructor").get(
                            "use_min_intersections_method", True
                        )

                        if use_min_intersections:
                            min_points_per_line = self.settings.get_nested("grid_reconstructor").get(
                                "min_points_per_line", 7
                            )

                            logger.info(f"Trying minimum intersections per line method (min={min_points_per_line})")

                            grid_points = self.grid_reconstructor.reconstruct_with_min_intersections_per_line(
                                intersections,  # All points for context
                                image.shape,
                                board_bbox,
                                filtered_intersections,  # Filtered points if available
                                min_points_per_line=min_points_per_line
                            )

                            logger.info("✅ Grid reconstructed using minimum intersections per line method")
                            break

                    elif method == "board_detection" and board_detected and filtered_intersections is not None:
                        # Use board-aware reconstruction with 82% requirement
                        logger.info("Trying board-aware reconstruction with 82% requirement")

                        grid_points = self.grid_reconstructor.reconstruct_with_board_detection(
                            intersections, image.shape, board_bbox, filtered_intersections
                        )

                        logger.info("✅ Grid reconstructed using board detection method")
                        break

                    elif method == "standard":
                        # Use standard reconstruction
                        logger.info("Trying standard grid reconstruction")

                        grid_points = self.grid_reconstructor.reconstruct(intersections, image.shape)

                        logger.info("✅ Grid reconstructed using standard method")
                        break

                except Exception as e:
                    logger.warning(f"Method '{method}' failed: {str(e)}")
                    continue

            # If no method succeeded, raise error
            if grid_points is None:
                raise GridReconstructionError("All grid reconstruction methods failed")

            # Verify grid dimensions
            if len(grid_points) != 10 or any(len(row) != 10 for row in grid_points):
                raise GridReconstructionError(
                    f"Invalid grid dimensions: {len(grid_points)}x{len(grid_points[0]) if grid_points else 0}"
                )

            logger.info("Grid reconstructed successfully")

            # Store in current state
            self.current_state["grid_points"] = grid_points

            # Save intermediates if enabled
            if self.save_intermediates:
                self._save_grid_detection_intermediates(image, intersections, grid_points)

            return {
                "intersections": len(intersections),
                "filtered_intersections": len(filtered_intersections) if filtered_intersections else None,
                "board_detected": board_detected,
                "grid_dimensions": (len(grid_points), len(grid_points[0])),
                "reconstruction_method": method  # Track which method succeeded
            }

        except Exception as e:
            logger.error(f"Error detecting grid: {str(e)}")

            if isinstance(e, (IntersectionDetectionError, GridReconstructionError)):
                # Rethrow specific errors
                raise
            else:
                # Wrap other errors
                raise DetectionError(f"Failed to detect grid: {str(e)}")
    def _extract_cells(self) -> Dict[str, Any]:
        """
        Extract cells from detected grid.

        Returns:
            Dictionary with cell extraction results
        """
        try:
            # Get image and grid points from current state
            image = self.current_state["image"]
            grid_points = self.current_state["grid_points"]

            if image is None:
                raise PipelineError("No image to extract cells from")

            if grid_points is None:
                raise PipelineError("No grid points to extract cells from")

            # Extract cells
            cell_images = self.cell_extractor.extract(image, grid_points)

            # Verify cell dimensions
            if len(cell_images) != 9 or any(len(row) != 9 for row in cell_images):
                raise CellExtractionError(
                    f"Invalid cell dimensions: {len(cell_images)}x{len(cell_images[0]) if cell_images else 0}"
                )

            logger.info("Cells extracted successfully")

            # Store in current state
            self.current_state["cell_images"] = cell_images

            # Verify cell images if required
            if self.verification_at_each_stage:
                self._verify_cell_images(cell_images)

            # Save intermediates if enabled
            if self.save_intermediates:
                self._save_cell_extraction_intermediates(cell_images)

            return {
                "cell_dimensions": (len(cell_images), len(cell_images[0]))
            }

        except Exception as e:
            logger.error(f"Error extracting cells: {str(e)}")

            if isinstance(e, CellExtractionError):
                # Rethrow specific errors
                raise
            else:
                # Wrap other errors
                raise CellExtractionError(f"Failed to extract cells: {str(e)}")

    def _recognize_digits(self) -> Dict[str, Any]:
        """
        Recognize digits in extracted cells.

        Returns:
            Dictionary with digit recognition results
        """
        try:
            # Get cell images from current state
            cell_images = self.current_state["cell_images"]

            if cell_images is None:
                raise PipelineError("No cell images to recognize digits in")

            # Recognize digits
            digit_grid, confidence_grid = self.digit_recognizer.recognize(cell_images)

            # Verify grid dimensions
            if len(digit_grid) != 9 or any(len(row) != 9 for row in digit_grid):
                raise DigitRecognitionError(
                    f"Invalid digit grid dimensions: {len(digit_grid)}x{len(digit_grid[0]) if digit_grid else 0}"
                )

            logger.info("Digits recognized successfully")

            # Store in current state
            self.current_state["digit_grid"] = digit_grid
            self.current_state["confidence_grid"] = confidence_grid

            # Verify digit grid if required
            if self.verification_at_each_stage:
                self._verify_digit_grid(digit_grid, confidence_grid)

            # Calculate recognition statistics
            num_digits = sum(1 for row in digit_grid for digit in row if digit > 0)
            avg_confidence = sum(
                confidence_grid[i][j] for i in range(9) for j in range(9) if digit_grid[i][j] > 0
            ) / max(1, num_digits)

            # Save intermediates if enabled
            if self.save_intermediates:
                self._save_digit_recognition_intermediates(digit_grid, confidence_grid)

            return {
                "num_digits": num_digits,
                "avg_confidence": avg_confidence
            }

        except Exception as e:
            logger.error(f"Error recognizing digits: {str(e)}")

            if isinstance(e, DigitRecognitionError):
                # Rethrow specific errors
                raise
            else:
                # Wrap other errors
                raise DigitRecognitionError(f"Failed to recognize digits: {str(e)}")

    def _solve_puzzle(self) -> Dict[str, Any]:
        """
        Solve recognized Sudoku puzzle.

        Returns:
            Dictionary with solving results
        """
        try:
            # Get digit grid from current state
            digit_grid = self.current_state["digit_grid"]

            if digit_grid is None:
                raise PipelineError("No digit grid to solve")

            # Check if puzzle is solvable
            if not is_puzzle_solvable(digit_grid):
                raise InvalidPuzzleError("Recognized puzzle is not solvable (rule violation)")

            # Solve puzzle
            solved_grid = self.solver.solve(digit_grid)

            # Verify solution
            if len(solved_grid) != 9 or any(len(row) != 9 for row in solved_grid):
                raise SolverError(
                    f"Invalid solution dimensions: {len(solved_grid)}x{len(solved_grid[0]) if solved_grid else 0}"
                )

            # Check if solution is complete
            if any(0 in row for row in solved_grid):
                raise SolverError("Incomplete solution")

            logger.info("Puzzle solved successfully")

            # Store in current state
            self.current_state["solved_grid"] = solved_grid

            # Save intermediates if enabled
            if self.save_intermediates:
                self._save_solving_intermediates(solved_grid)

            return {
                "solution_complete": not any(0 in row for row in solved_grid)
            }

        except Exception as e:
            logger.error(f"Error solving puzzle: {str(e)}")

            if isinstance(e, (SolverError, InvalidPuzzleError)):
                # Rethrow specific errors
                raise
            else:
                # Wrap other errors
                raise SolverError(f"Failed to solve puzzle: {str(e)}")

    def _verify_cell_images(self, cell_images: List[List[ImageType]]) -> None:
        """
        Verify extracted cell images.

        Args:
            cell_images: 2D list of cell images

        Raises:
            CellExtractionError: If verification fails
        """
        # Check dimensions
        if len(cell_images) != 9 or any(len(row) != 9 for row in cell_images):
            raise CellExtractionError(
                f"Invalid cell dimensions: {len(cell_images)}x{len(cell_images[0]) if cell_images else 0}"
            )

        # Check if any cell is empty or invalid
        for i in range(9):
            for j in range(9):
                cell = cell_images[i][j]

                if cell is None or cell.size == 0:
                    raise CellExtractionError(f"Empty cell at position ({i}, {j})")

                if not isinstance(cell, np.ndarray) or len(cell.shape) < 2:
                    raise CellExtractionError(f"Invalid cell format at position ({i}, {j})")

    def _verify_digit_grid(self, digit_grid: GridType, confidence_grid: List[List[float]]) -> None:
        """
        Verify recognized digit grid.

        Args:
            digit_grid: Grid of recognized digits
            confidence_grid: Grid of confidence scores

        Raises:
            DigitRecognitionError: If verification fails
        """
        # Check dimensions
        if len(digit_grid) != 9 or any(len(row) != 9 for row in digit_grid):
            raise DigitRecognitionError(
                f"Invalid digit grid dimensions: {len(digit_grid)}x{len(digit_grid[0]) if digit_grid else 0}"
            )

        if len(confidence_grid) != 9 or any(len(row) != 9 for row in confidence_grid):
            raise DigitRecognitionError(
                f"Invalid confidence grid dimensions: {len(confidence_grid)}x{len(confidence_grid[0]) if confidence_grid else 0}"
            )

        # Check if any digit is out of range
        for i in range(9):
            for j in range(9):
                digit = digit_grid[i][j]
                confidence = confidence_grid[i][j]

                if not (0 <= digit <= 9):
                    raise DigitRecognitionError(f"Invalid digit at position ({i}, {j}): {digit}")

                if not (0.0 <= confidence <= 1.0):
                    raise DigitRecognitionError(f"Invalid confidence at position ({i}, {j}): {confidence}")

    def _save_grid_detection_intermediates(
        self,
        image: ImageType,
        intersections: List[PointType],
        grid_points: GridPointsType
    ) -> None:
        """
        Save grid detection intermediate results.

        Args:
            image: Input image
            intersections: Detected intersection points
            grid_points: Reconstructed grid points
        """
        # Create output directory
        os.makedirs("intermediates", exist_ok=True)

        # Save visualization of intersections
        intersection_viz = image.copy()
        for x, y in intersections:
            cv2.circle(intersection_viz, (x, y), 5, (0, 0, 255), -1)

        cv2.imwrite("intermediates/intersections.jpg", intersection_viz)

        # Save visualization of grid
        grid_viz = image.copy()

        # Draw grid lines
        for i in range(10):
            # Draw horizontal line
            cv2.polylines(
                grid_viz,
                [np.array(grid_points[i], dtype=np.int32)],
                False,
                (0, 255, 0),
                2
            )

            # Draw vertical line
            points = [grid_points[j][i] for j in range(10)]
            cv2.polylines(
                grid_viz,
                [np.array(points, dtype=np.int32)],
                False,
                (0, 255, 0),
                2
            )

        cv2.imwrite("intermediates/grid.jpg", grid_viz)

    def _save_cell_extraction_intermediates(self, cell_images: List[List[ImageType]]) -> None:
        """
        Save cell extraction intermediate results.

        Args:
            cell_images: Extracted cell images
        """
        # Create output directory
        os.makedirs("intermediates/cells", exist_ok=True)

        # Save individual cells
        for i in range(9):
            for j in range(9):
                cell = cell_images[i][j]
                cv2.imwrite(f"intermediates/cells/cell_{i}_{j}.jpg", cell)

        # Create a grid image
        grid_image = np.ones((9 * 28, 9 * 28), dtype=np.uint8) * 255

        for i in range(9):
            for j in range(9):
                cell = cell_images[i][j]

                # Resize if necessary
                if cell.shape[0] != 28 or cell.shape[1] != 28:
                    cell = cv2.resize(cell, (28, 28))

                # Place in grid
                grid_image[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = cell

        cv2.imwrite("intermediates/cells_grid.jpg", grid_image)

    def _save_digit_recognition_intermediates(
        self,
        digit_grid: GridType,
        confidence_grid: List[List[float]]
    ) -> None:
        """
        Save digit recognition intermediate results.

        Args:
            digit_grid: Grid of recognized digits
            confidence_grid: Grid of confidence scores
        """
        # Create output directory
        os.makedirs("intermediates", exist_ok=True)

        # Save digit grid as text file
        with open("intermediates/digit_grid.txt", "w") as f:
            for row in digit_grid:
                f.write("".join(str(digit) for digit in row) + "\n")

        # Save confidence grid as text file
        with open("intermediates/confidence_grid.txt", "w") as f:
            for row in confidence_grid:
                f.write(" ".join(f"{conf:.2f}" for conf in row) + "\n")

    def _save_solving_intermediates(self, solved_grid: GridType) -> None:
        """
        Save solving intermediate results.

        Args:
            solved_grid: Solved grid
        """
        # Create output directory
        os.makedirs("intermediates", exist_ok=True)

        # Save solved grid as text file
        with open("intermediates/solved_grid.txt", "w") as f:
            for row in solved_grid:
                f.write("".join(str(digit) for digit in row) + "\n")

    def _recover_intersection_detection_by_enhancing_contrast(self, *args: Any, **kwargs: Any) -> List[PointType]:
        """
        Recovery strategy: Enhance image contrast for intersection detection.

        Returns:
            List of detected intersection points
        """
        logger.info("Attempting recovery: Enhancing contrast for intersection detection")

        # Get image from current state
        image = self.current_state["image"]
        if image is None:
            raise PipelineError("No image to detect intersections in")

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Detect intersections in enhanced image
        enhanced_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        intersections = self.intersection_detector.detect(enhanced_image)

        # Verify we have enough intersections
        if len(intersections) < 20:
            raise IntersectionDetectionError(
                f"Insufficient intersections detected after contrast enhancement: {len(intersections)}"
            )

        logger.info(f"Recovery successful: Detected {len(intersections)} intersections after contrast enhancement")

        # Store in current state
        self.current_state["intersections"] = intersections

        return intersections

    def _recover_intersection_detection_by_trying_alternative_method(self, *args: Any, **kwargs: Any) -> List[PointType]:
        """
        Recovery strategy: Try alternative method for intersection detection.

        Returns:
            List of detected intersection points
        """
        logger.info("Attempting recovery: Using alternative method for intersection detection")

        # Get image from current state
        image = self.current_state["image"]
        if image is None:
            raise PipelineError("No image to detect intersections in")

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply morphological operations to enhance grid lines
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        dilated = cv2.dilate(binary, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        # Detect Hough lines
        lines = cv2.HoughLinesP(
            eroded,
            1,
            np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None:
            raise IntersectionDetectionError("No lines detected with Hough transform")

        # Calculate intersections between all pairs of lines
        intersections = []

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]

                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2

                # Calculate intersection point
                d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

                if abs(d) > 1e-6:  # Non-parallel lines
                    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
                    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d

                    # Check if intersection is within image boundaries
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        intersections.append((int(x), int(y)))

        # Cluster similar intersection points
        if not intersections:
            raise IntersectionDetectionError("No intersections found with alternative method")

        # Convert to numpy array
        points_array = np.array(intersections, dtype=np.float32)

        # Apply DBSCAN clustering
        clustering = cv2.partition(points_array, cv2.PARTITION_CLUSTERS)

        # Calculate cluster centers
        clustered_intersections = []
        for cluster_id in range(max(clustering) + 1):
            cluster_points = points_array[clustering == cluster_id]
            if len(cluster_points) > 0:
                center = np.mean(cluster_points, axis=0)
                clustered_intersections.append((int(center[0]), int(center[1])))

        # Verify we have enough intersections
        if len(clustered_intersections) < 20:
            raise IntersectionDetectionError(
                f"Insufficient intersections detected with alternative method: {len(clustered_intersections)}"
            )

        logger.info(f"Recovery successful: Detected {len(clustered_intersections)} intersections with alternative method")

        # Store in current state
        self.current_state["intersections"] = clustered_intersections

        return clustered_intersections

    def _recover_grid_reconstruction_by_relaxing_parameters(self, *args: Any, **kwargs: Any) -> GridPointsType:
        """
        Recovery strategy: Relax parameters for grid reconstruction.

        Returns:
            Reconstructed grid points
        """
        logger.info("Attempting recovery: Relaxing parameters for grid reconstruction")

        # Get intersections from current state
        intersections = self.current_state["intersections"]
        image = self.current_state["image"]

        if intersections is None:
            raise PipelineError("No intersections to reconstruct grid from")

        if image is None:
            raise PipelineError("No image to reconstruct grid in")

        # Create a copy of the grid reconstructor with relaxed parameters
        relaxed_reconstructor = RobustGridReconstructor()

        # Relax parameters
        # - Decrease RANSAC threshold to allow more inliers
        # - Decrease minimum line points to detect more lines
        # - Increase max angle deviation to allow more flexibility
        original_settings = {
            "ransac_threshold": relaxed_reconstructor.ransac_reconstructor.ransac_threshold,
            "min_line_points": relaxed_reconstructor.ransac_reconstructor.min_line_points,
            "max_angle_deviation": relaxed_reconstructor.ransac_reconstructor.max_angle_deviation
        }

        try:
            relaxed_reconstructor.ransac_reconstructor.ransac_threshold *= 1.5
            relaxed_reconstructor.ransac_reconstructor.min_line_points = max(3, relaxed_reconstructor.ransac_reconstructor.min_line_points - 2)
            relaxed_reconstructor.ransac_reconstructor.max_angle_deviation *= 1.5

            # Reconstruct grid with relaxed parameters
            grid_points = relaxed_reconstructor.reconstruct(intersections, image.shape)

            # Verify grid dimensions
            if len(grid_points) != 10 or any(len(row) != 10 for row in grid_points):
                raise GridReconstructionError(
                    f"Invalid grid dimensions after relaxing parameters: {len(grid_points)}x{len(grid_points[0]) if grid_points else 0}"
                )

            logger.info("Recovery successful: Grid reconstructed with relaxed parameters")

            # Store in current state
            self.current_state["grid_points"] = grid_points

            return grid_points
        finally:
            # Restore original parameters
            relaxed_reconstructor.ransac_reconstructor.ransac_threshold = original_settings["ransac_threshold"]
            relaxed_reconstructor.ransac_reconstructor.min_line_points = original_settings["min_line_points"]
            relaxed_reconstructor.ransac_reconstructor.max_angle_deviation = original_settings["max_angle_deviation"]

    def _recover_grid_reconstruction_by_trying_alternative_method(self, *args: Any, **kwargs: Any) -> GridPointsType:
        """
        Recovery strategy: Try alternative method for grid reconstruction.

        Returns:
            Reconstructed grid points
        """
        logger.info("Attempting recovery: Using alternative method for grid reconstruction")

        # Get intersections from current state
        intersections = self.current_state["intersections"]
        image = self.current_state["image"]

        if intersections is None:
            raise PipelineError("No intersections to reconstruct grid from")

        if image is None:
            raise PipelineError("No image to reconstruct grid in")

        # Create grid points directly from the extrema of the point cloud
        points_array = np.array(intersections)
        min_x = np.min(points_array[:, 0])
        max_x = np.max(points_array[:, 0])
        min_y = np.min(points_array[:, 1])
        max_y = np.max(points_array[:, 1])

        # Create a regular grid
        grid_points = []
        for y in range(10):
            row_points = []
            for x in range(10):
                # Calculate point coordinates
                point_x = min_x + (max_x - min_x) * x / 9
                point_y = min_y + (max_y - min_y) * y / 9
                row_points.append((int(point_x), int(point_y)))
            grid_points.append(row_points)

        # Snap grid points to nearby intersections
        for i in range(10):
            for j in range(10):
                grid_x, grid_y = grid_points[i][j]

                # Find nearest intersection
                distances = np.sqrt(np.sum((points_array - np.array([grid_x, grid_y]))**2, axis=1))
                nearest_idx = np.argmin(distances)

                # Snap to nearest intersection if it's close enough
                if distances[nearest_idx] < 20:
                    grid_points[i][j] = (int(points_array[nearest_idx][0]), int(points_array[nearest_idx][1]))

        logger.info("Recovery successful: Grid reconstructed with alternative method")

        # Store in current state
        self.current_state["grid_points"] = grid_points

        return grid_points

    def _recover_cell_extraction_by_refining_grid(self, *args: Any, **kwargs: Any) -> List[List[ImageType]]:
        """
        Recovery strategy: Refine grid points for cell extraction.

        Returns:
            Extracted cell images
        """
        logger.info("Attempting recovery: Refining grid for cell extraction")

        # Get grid points and image from current state
        grid_points = self.current_state["grid_points"]
        image = self.current_state["image"]

        if grid_points is None:
            raise PipelineError("No grid points to refine")

        if image is None:
            raise PipelineError("No image to extract cells from")

        # Refine grid points by making them more regular
        refined_grid = []
        for i in range(10):
            refined_row = []
            for j in range(10):
                # Get current point
                x, y = grid_points[i][j]

                # Calculate expected position based on grid corners
                tl = grid_points[0][0]
                tr = grid_points[0][9]
                bl = grid_points[9][0]
                br = grid_points[9][9]

                # Interpolate
                top_x = tl[0] + (tr[0] - tl[0]) * j / 9
                top_y = tl[1] + (tr[1] - tl[1]) * j / 9
                bottom_x = bl[0] + (br[0] - bl[0]) * j / 9
                bottom_y = bl[1] + (br[1] - bl[1]) * j / 9

                expected_x = top_x + (bottom_x - top_x) * i / 9
                expected_y = top_y + (bottom_y - top_y) * i / 9

                # Blend current and expected positions
                blend_factor = 0.7  # 70% expected, 30% current
                refined_x = int(x * (1 - blend_factor) + expected_x * blend_factor)
                refined_y = int(y * (1 - blend_factor) + expected_y * blend_factor)

                refined_row.append((refined_x, refined_y))
            refined_grid.append(refined_row)

        # Store refined grid
        self.current_state["grid_points"] = refined_grid

        # Extract cells with refined grid
        cell_images = self.cell_extractor.extract(image, refined_grid)

        # Verify cell dimensions
        if len(cell_images) != 9 or any(len(row) != 9 for row in cell_images):
            raise CellExtractionError(
                f"Invalid cell dimensions after grid refinement: {len(cell_images)}x{len(cell_images[0]) if cell_images else 0}"
            )

        logger.info("Recovery successful: Cells extracted with refined grid")

        # Store in current state
        self.current_state["cell_images"] = cell_images

        return cell_images

    def _recover_cell_extraction_by_adjusting_cell_boundaries(self, *args: Any, **kwargs: Any) -> List[List[ImageType]]:
        """
        Recovery strategy: Adjust cell boundaries for extraction.

        Returns:
            Extracted cell images
        """
        logger.info("Attempting recovery: Adjusting cell boundaries for extraction")

        # Get grid points and image from current state
        grid_points = self.current_state["grid_points"]
        image = self.current_state["image"]

        if grid_points is None:
            raise PipelineError("No grid points to adjust")

        if image is None:
            raise PipelineError("No image to extract cells from")

        # Create a copy of the cell extractor with adjusted parameters
        adjusted_extractor = RobustCellExtractor()

        # Adjust parameters
        # - Increase border padding to remove more of the cell border
        # - Try different extraction method
        original_padding = adjusted_extractor.perspective_extractor.border_padding

        try:
            adjusted_extractor.perspective_extractor.border_padding = 0.2  # Increase padding

            # Extract cells with adjusted parameters
            cell_images = adjusted_extractor.extract(image, grid_points)

            # Verify cell dimensions
            if len(cell_images) != 9 or any(len(row) != 9 for row in cell_images):
                raise CellExtractionError(
                    f"Invalid cell dimensions after adjusting boundaries: {len(cell_images)}x{len(cell_images[0]) if cell_images else 0}"
                )

            logger.info("Recovery successful: Cells extracted with adjusted boundaries")

            # Store in current state
            self.current_state["cell_images"] = cell_images

            return cell_images
        finally:
            # Restore original parameters
            adjusted_extractor.perspective_extractor.border_padding = original_padding

    def _recover_digit_recognition_by_enhancing_contrast(self, *args: Any, **kwargs: Any) -> Tuple[GridType, List[List[float]]]:
        """
        Recovery strategy: Enhance cell contrast for digit recognition.

        Returns:
            Tuple of (grid of recognized digits, grid of confidence scores)
        """
        logger.info("Attempting recovery: Enhancing cell contrast for digit recognition")

        # Get cell images from current state
        cell_images = self.current_state["cell_images"]

        if cell_images is None:
            raise PipelineError("No cell images to enhance")

        # Enhance contrast in each cell
        enhanced_cells = []
        for row in cell_images:
            enhanced_row = []
            for cell in row:
                # Apply contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                if len(cell.shape) == 2:  # Grayscale
                    enhanced = clahe.apply(cell)
                else:  # Color
                    enhanced = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                    enhanced = clahe.apply(enhanced)

                enhanced_row.append(enhanced)
            enhanced_cells.append(enhanced_row)

        # Recognize digits in enhanced cells
        digit_grid, confidence_grid = self.digit_recognizer.recognize(enhanced_cells)

        # Verify grid dimensions
        if len(digit_grid) != 9 or any(len(row) != 9 for row in digit_grid):
            raise DigitRecognitionError(
                f"Invalid digit grid dimensions after contrast enhancement: {len(digit_grid)}x{len(digit_grid[0]) if digit_grid else 0}"
            )

        logger.info("Recovery successful: Digits recognized with enhanced contrast")

        # Store in current state
        self.current_state["digit_grid"] = digit_grid
        self.current_state["confidence_grid"] = confidence_grid

        return digit_grid, confidence_grid

    def _recover_digit_recognition_by_trying_alternative_model(self, *args: Any, **kwargs: Any) -> Tuple[GridType, List[List[float]]]:
        """
        Recovery strategy: Try alternative model for digit recognition.

        Returns:
            Tuple of (grid of recognized digits, grid of confidence scores)
        """
        logger.info("Attempting recovery: Using alternative model for digit recognition")

        # Get cell images from current state
        cell_images = self.current_state["cell_images"]

        if cell_images is None:
            raise PipelineError("No cell images to recognize")

        # Try with template matching recognizer directly
        try:
            # Get access to the template recognizer
            template_recognizer = self.digit_recognizer.template_recognizer

            # Recognize digits
            digit_grid, confidence_grid = template_recognizer.recognize(cell_images)

            # Verify grid dimensions
            if len(digit_grid) != 9 or any(len(row) != 9 for row in digit_grid):
                raise DigitRecognitionError(
                    f"Invalid digit grid dimensions with alternative model: {len(digit_grid)}x{len(digit_grid[0]) if digit_grid else 0}"
                )

            logger.info("Recovery successful: Digits recognized with alternative model")

            # Store in current state
            self.current_state["digit_grid"] = digit_grid
            self.current_state["confidence_grid"] = confidence_grid

            return digit_grid, confidence_grid
        except Exception as e:
            logger.error(f"Alternative model also failed: {str(e)}")
            raise

    def _recover_solving_by_validating_digits(self, *args: Any, **kwargs: Any) -> GridType:
        """
        Recovery strategy: Validate and correct digit grid before solving.

        Returns:
            Solved grid
        """
        logger.info("Attempting recovery: Validating digits before solving")

        # Get digit grid and confidence grid from current state
        digit_grid = self.current_state["digit_grid"]
        confidence_grid = self.current_state["confidence_grid"]

        if digit_grid is None:
            raise PipelineError("No digit grid to validate")

        if confidence_grid is None:
            raise PipelineError("No confidence grid to use for validation")

        # Create a corrected grid with only high-confidence digits
        corrected_grid = [[0 for _ in range(9)] for _ in range(9)]

        for i in range(9):
            for j in range(9):
                digit = digit_grid[i][j]
                confidence = confidence_grid[i][j]

                # Keep only high-confidence digits
                if confidence >= 0.8 and digit > 0:
                    corrected_grid[i][j] = digit

        # Check if corrected grid is valid
        try:
            validate_sudoku_rules(corrected_grid)
        except InvalidPuzzleError:
            logger.warning("Corrected grid still violates Sudoku rules")

            # Further correct by removing conflicting digits
            corrected_grid = self._correct_conflicting_digits(corrected_grid, confidence_grid)

        # Solve the corrected grid
        solved_grid = self.solver.solve(corrected_grid)

        logger.info("Recovery successful: Puzzle solved with validated digits")

        # Store in current state
        self.current_state["solved_grid"] = solved_grid

        return solved_grid

    def _correct_conflicting_digits(self, grid: GridType, confidence_grid: List[List[float]]) -> GridType:
        """
        Correct conflicting digits in a grid.

        Args:
            grid: Digit grid
            confidence_grid: Confidence scores

        Returns:
            Corrected grid
        """
        corrected = [[grid[i][j] for j in range(9)] for i in range(9)]

        # Find conflicts in rows
        for i in range(9):
            digits = {}
            for j in range(9):
                digit = corrected[i][j]
                if digit > 0:
                    if digit in digits:
                        # Conflict found, keep the one with higher confidence
                        prev_j = digits[digit]
                        if confidence_grid[i][j] > confidence_grid[i][prev_j]:
                            corrected[i][prev_j] = 0
                        else:
                            corrected[i][j] = 0
                    else:
                        digits[digit] = j

        # Find conflicts in columns
        for j in range(9):
            digits = {}
            for i in range(9):
                digit = corrected[i][j]
                if digit > 0:
                    if digit in digits:
                        # Conflict found, keep the one with higher confidence
                        prev_i = digits[digit]
                        if confidence_grid[i][j] > confidence_grid[prev_i][j]:
                            corrected[prev_i][j] = 0
                        else:
                            corrected[i][j] = 0
                    else:
                        digits[digit] = i

        # Find conflicts in 3x3 boxes
        for box_i in range(3):
            for box_j in range(3):
                digits = {}
                for i in range(3):
                    for j in range(3):
                        row, col = box_i * 3 + i, box_j * 3 + j
                        digit = corrected[row][col]
                        if digit > 0:
                            if digit in digits:
                                # Conflict found, keep the one with higher confidence
                                prev_pos = digits[digit]
                                prev_row, prev_col = prev_pos
                                if confidence_grid[row][col] > confidence_grid[prev_row][prev_col]:
                                    corrected[prev_row][prev_col] = 0
                                else:
                                    corrected[row][col] = 0
                            else:
                                digits[digit] = (row, col)

        return corrected

    def _recover_solving_by_relaxing_constraints(self, *args: Any, **kwargs: Any) -> GridType:
        """
        Recovery strategy: Relax constraints for solving.

        Returns:
            Solved grid
        """
        logger.info("Attempting recovery: Relaxing constraints for solving")

        # Get digit grid from current state
        digit_grid = self.current_state["digit_grid"]

        if digit_grid is None:
            raise PipelineError("No digit grid to solve")

        # Create a copy of the solver with relaxed parameters
        relaxed_solver = RobustSolver()

        # Relax parameters
        # - Increase max solving time
        # - Force use of backtracking (simpler but more reliable)
        original_time = relaxed_solver.max_solving_time

        try:
            relaxed_solver.max_solving_time = 10  # Increase time limit
            relaxed_solver.use_constraint_propagation = False  # Use only backtracking

            # Solve with relaxed parameters
            solved_grid = relaxed_solver.solve(digit_grid)

            logger.info("Recovery successful: Puzzle solved with relaxed constraints")

            # Store in current state
            self.current_state["solved_grid"] = solved_grid

            return solved_grid
        finally:
            # Restore original parameters
            relaxed_solver.max_solving_time = original_time
            relaxed_solver.use_constraint_propagation = True

    def _recover_solving_by_correcting_invalid_puzzle(self, *args: Any, **kwargs: Any) -> GridType:
        """
        Recovery strategy: Correct invalid puzzle.

        Returns:
            Solved grid
        """
        logger.info("Attempting recovery: Correcting invalid puzzle")

        # Get digit grid and confidence grid from current state
        digit_grid = self.current_state["digit_grid"]
        confidence_grid = self.current_state["confidence_grid"]

        if digit_grid is None:
            raise PipelineError("No digit grid to correct")

        if confidence_grid is None:
            raise PipelineError("No confidence grid to use for correction")

        # Start by removing low-confidence digits
        threshold = 0.7
        corrected_grid = [[0 for _ in range(9)] for _ in range(9)]

        for i in range(9):
            for j in range(9):
                digit = digit_grid[i][j]
                confidence = confidence_grid[i][j]

                if confidence >= threshold and digit > 0:
                    corrected_grid[i][j] = digit

        # Try to solve the corrected grid
        try:
            solved_grid = self.solver.solve(corrected_grid)

            logger.info("Recovery successful: Invalid puzzle corrected and solved")

            # Store in current state
            self.current_state["solved_grid"] = solved_grid

            return solved_grid
        except (SolverError, InvalidPuzzleError):
            # If still invalid, reduce threshold and try again
            thresholds = [0.6, 0.5, 0.4]

            for threshold in thresholds:
                logger.info(f"Trying with lower confidence threshold: {threshold}")

                corrected_grid = [[0 for _ in range(9)] for _ in range(9)]

                for i in range(9):
                    for j in range(9):
                        digit = digit_grid[i][j]
                        confidence = confidence_grid[i][j]

                        if confidence >= threshold and digit > 0:
                            corrected_grid[i][j] = digit

                try:
                    solved_grid = self.solver.solve(corrected_grid)

                    logger.info(f"Recovery successful with threshold {threshold}")

                    # Store in current state
                    self.current_state["solved_grid"] = solved_grid

                    return solved_grid
                except (SolverError, InvalidPuzzleError):
                    continue

            # If all thresholds failed, create an empty grid
            logger.warning("All recovery attempts failed, returning empty grid")
            empty_solution = [[0 for _ in range(9)] for _ in range(9)]

            # Store in current state
            self.current_state["solved_grid"] = empty_solution

            return empty_solution
