"""
Default configuration and fallback values for the Sudoku Recognizer system.

This module provides sensible defaults for all system components with
conservative thresholds to maximize robustness.
"""

# Default configuration with fallback values for all components
DEFAULT_CONFIG = {
    # General system settings
    "system": {
        "debug_mode": False,
        "log_level": "INFO",
        "max_retries": 3,
        "timeout_seconds": 70,
        "temp_directory": "/tmp/sudoku_recognizer",
        "max_image_size": 1600,  # Max dimension in pixels
        "min_image_size": 300,   # Min dimension in pixels
    },

    # Intersection detector settings
    "intersection_detector": {
        "model_path": "data/models/intersection_detector.h5",
        "confidence_threshold": 0.8, # Matches notebook
        "nms_threshold": 3,
        "min_intersections": 70,  # Minimum number of intersections to proceed
        "detection_methods": ["cnn", "hough", "adaptive_threshold"], # Matches notebook
        "patch_size": 15,  # Size of intersection patch in pixels, matches notebook
        "fallback_to_cv": True,  # Fallback to OpenCV methods if CNN fails
        "use_ensemble": True,  # Use ensemble of detection methods
        "x_correction": -5, # Updated from -4 to match notebook
        "y_correction": 7,  # Updated from 6 to match notebook
    },

    # Grid reconstructor settings
    "grid_reconstructor": {
        "ransac_iterations": 1070,
        "ransac_threshold": 5,
        "min_line_points": 5,
        "grid_size": 9,
        "min_line_separation": 20,  # Minimum pixels between grid lines
        "max_angle_deviation": 8,  # Maximum angle deviation in degrees
        "use_homography": True,
        "use_grid_refinement": True,
        "max_perspective_distortion": 45,  # Max perspective angle in degrees
        "use_min_intersections_method": True,  # Matches notebook
        "min_points_per_line": 7,  # Minimum intersections per line, matches notebook
        "reconstruction_method_order": ["min_intersections", "board_detection", "standard"], # Matches notebook
        "grid_detection_methods": ["ransac", "hough", "contour"],
        "use_board_aware_reconstruction": True # Added from notebook
    },

    # Cell extractor settings
    "cell_extractor": {
        "cell_size": 28,  # Output cell size in pixels, matches notebook
        "border_padding": 0.08,  # Updated from 0.05 to match notebook
        "perspective_correction": True, # Matches notebook
        "contrast_enhancement": False, # Matches notebook
        "noise_reduction": False, # Matches notebook
        "adaptive_thresholding": False, # Matches notebook
        "histogram_equalization": False,
        "use_multiple_extractors": False,  # Try different extraction methods, matches notebook
        "extraction_mode": "preserve" # Added from notebook
    },

    # Digit recognizer settings
    "digit_recognizer": {
        "model_path": "data/models/digit_recognizer.h5",
        "confidence_threshold": 0.82,
        "use_multiple_models": True,
        "use_ensemble": True,
        "empty_cell_threshold": 0.90,  # Threshold to consider a cell empty
        "augment_at_runtime": True,  # Apply augmentation during recognition
        "fallback_models": ["tflite", "svm", "template_matching"],
        "digit_height_min_ratio": 0.3,  # Minimum height ratio relative to cell
        "digit_width_min_ratio": 0.1,  # Minimum width ratio relative to cell
    },

    # Sudoku solver settings
    "solver": {
        "use_constraint_propagation": True,
        "use_backtracking": True,
        "max_solving_time": 15,  # Max time in seconds
        "use_multiple_solvers": True,
        "validate_solution": True,
        "fallback_to_simpler_solver": True,
    },

    # Board Detector (New section from notebook)
    "board_detector": {
        "confidence_threshold": 0.5,
        "model_input_size": 416,
        "diagonal_margin_factor": 14
    },

    # Pipeline settings
    "pipeline": {
        "save_intermediates": False,
        "parallel_processing": True,
        "max_workers": 4,
        "stages": ["detect_grid", "extract_cells", "recognize_digits", "solve"],
        "retry_failed_stages": True,
        "verification_at_each_stage": True,
        "recovery_strategies": {
            "grid_detection_failed": ["enhance_contrast", "try_alternative_method"],
            "cell_extraction_failed": ["refine_grid", "adjust_cell_boundaries"],
            "digit_recognition_failed": ["enhance_contrast", "try_alternative_model"],
            "solving_failed": ["validate_digits", "relax_constraints"]
        },
        "use_board_detection": True, # Added from notebook
        "min_filtered_intersections": 78, # Added from notebook
        "grid_separation_enabled": True, # Added from notebook
        "min_grid_size": 40 # Added from notebook
    },

    # Training settings
    "training": {
        "batch_size": 32,
        "epochs": 100,
        "validation_split": 0.2,
        "learning_rate": 0.0007,
        "early_stopping_patience": 10,
        "data_augmentation": True,
        "use_transfer_learning": True,
        "save_best_only": True,
    },

    # Evaluation settings
    "evaluation": {
        "test_split": 0.18,
        "metrics": ["accuracy", "precision", "recall", "f1"],
        "confusion_matrix": True,
        "save_error_examples": True,
    },

    # Web application settings
    "web_app": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": False,
        "max_upload_size_mb": 10,
        "allowed_extensions": ["jpg", "jpeg", "png", "bmp"],
        "session_timeout": 3600,
        "rate_limit": 60,  # Requests per minute
    }
}

# Critical fallback thresholds - these override user settings if they are below these values
CRITICAL_THRESHOLDS = {
    "intersection_detector.confidence_threshold": 0.4,
    "intersection_detector.min_intersections": 40,
    "grid_reconstructor.ransac_iterations": 500,
    "digit_recognizer.confidence_threshold": 0.5,
}

# Default error messages
DEFAULT_ERROR_MESSAGES = {
    "image_load_failed": "Could not load image. Please check the file format and try again.",
    "grid_detection_failed": "Could not detect Sudoku grid. Please ensure the grid is clearly visible.",
    "insufficient_intersections": "Not enough grid intersections detected. Try improving lighting or image quality.",
    "cell_extraction_failed": "Failed to extract Sudoku cells. Please ensure the grid is not distorted.",
    "digit_recognition_failed": "Failed to recognize digits. Please ensure digits are clearly written.",
    "solving_failed": "Sorry :( Could not solve the Sudoku puzzle. Please check that the recognized digits are correct.",
    "invalid_puzzle": "The recognized puzzle is invalid. Please check that the image contains a valid Sudoku puzzle.",
    "system_error": "An unexpected error occurred. Please try again with a different image.",
}
