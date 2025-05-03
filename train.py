#!/usr/bin/env python3
"""
Training Script for Sudoku Recognizer Models.

This script trains all models in the Sudoku recognizer system.
"""

import os
import sys
import argparse
import logging
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional

# Assuming utils, config, models, pipeline are accessible
try:
    from config.settings import initialize_settings
    from utils.data_preparation import (
        load_dataset, prepare_training_data, normalize_cell_images
    )
    from utils.error_handling import setup_exception_handling
    from models.intersection_detector import RobustIntersectionDetector
    from models.grid_reconstructor import RobustGridReconstructor
    from models.cell_extractor import RobustCellExtractor
    from models.digit_recognizer import RobustDigitRecognizer
    from models.solver import RobustSolver
    from pipeline import SudokuRecognizerPipeline
except ImportError as e:
     # Basic logging if imports fail
     logging.basicConfig(level=logging.ERROR)
     logger = logging.getLogger(__name__)
     logger.error(f"Import error: {e}. Please ensure all modules are in PYTHONPATH.")
     sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train.log') # Log to file
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Sudoku recognizer models')

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing raw training data (.jpg, .dat files)'
    )

    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Directory to save/load processed training data (.npy files)'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        default='data/models',
        help='Directory to save trained models'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.json',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Only prepare training data, do not train models'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['all', 'digit', 'intersection', 'grid', 'cell', 'solver'],
        default='all',
        help='Which model to train (default: all)'
    )

    parser.add_argument(
        '--augmentation-factor',
        type=int,
        default=3,
        help='Data augmentation factor for digit dataset'
    )

    parser.add_argument(
        '--force-preparation',
        action='store_true',
        help='Force re-preparation of training data even if processed files exist'
    )

    return parser.parse_args()


def prepare_data(args):
    """Prepare training data using utils.data_preparation.prepare_training_data."""
    # Check if processed data already exists and skip if not forced
    if not args.force_preparation:
        # Check for a key output file to determine existence
        train_digits_path = os.path.join(args.processed_dir, "train_digits_data.npy")
        if os.path.exists(train_digits_path):
            logger.info(f"Processed data found in {args.processed_dir}. Skipping preparation.")
            logger.info("Use --force-preparation to re-run preparation.")
            return # Skip preparation

    logger.info("Starting data preparation...")

    # Create pipeline instance - needed for cell extraction during preparation
    # It might load existing models if available to aid detection stages
    try:
        pipeline = SudokuRecognizerPipeline()
        # Load existing models if they exist to help preparation steps like grid detection
        if os.path.exists(args.model_dir) and os.path.isdir(args.model_dir):
            logger.info(f"Attempting to load existing models from {args.model_dir} to assist data preparation.")
            pipeline.load_models(args.model_dir) # Assumes this handles missing models gracefully
        else:
             logger.info("No existing model directory found. Data preparation will use default pipeline components.")

    except NameError:
         logger.error("SudokuRecognizerPipeline class not found. Cannot prepare data.")
         raise # Re-raise the error
    except Exception as e:
         logger.error(f"Error initializing pipeline for data preparation: {e}")
         raise


    # Prepare training data
    try:
        logger.info(f"Running prepare_training_data from raw data dir: {args.data_dir}")
        stats = prepare_training_data(
            args.data_dir,
            args.processed_dir,
            pipeline,
            args.augmentation_factor
            # Add other args like test_ratio, seed if needed by prepare_training_data
        )

        # Log statistics
        logger.info("Data preparation complete.")
        for key, value in stats.items():
             logger.info(f"  {key}: {value}")

        # Save statistics
        stats_path = os.path.join(args.processed_dir, "preparation_stats.json")
        try:
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Preparation statistics saved to {stats_path}")
        except IOError as e:
            logger.error(f"Could not save preparation statistics to {stats_path}: {e}")

    except FileNotFoundError as e:
         logger.error(f"Data preparation failed: Input directory not found? {e}")
         raise
    except Exception as e:
        logger.error(f"An error occurred during data preparation: {e}")
        raise # Stop execution if preparation fails


def train_intersection_detector(args):
    """Train intersection detector model."""
    logger.info("Training intersection detector...")

    # Load training data - Use default list format (as_numpy=False)
    try:
        logger.info(f"Loading intersection data from {args.processed_dir} (prefix: train_intersections)")
        # Keep default behavior (returns lists) unless explicitly changed
        train_images, train_points = load_dataset(args.processed_dir, "train_intersections") # <<< NO as_numpy=True HERE
        if not train_images:
             logger.warning("No training data loaded for intersection detector. Skipping training.")
             return
    except Exception as e:
        logger.error(f"Failed to load intersection training data: {e}")
        return # Cannot train without data

    # Create model
    try:
        detector = RobustIntersectionDetector() # Assumes constructor needs no args
    except NameError:
         logger.error("RobustIntersectionDetector class not found. Cannot train.")
         return
    except Exception as e:
         logger.error(f"Error initializing RobustIntersectionDetector: {e}")
         return

    # Train model
    try:
        logger.info(f"Starting training for intersection detector with {len(train_images)} images...")
        start_time = time.time()
        # Assume detector.train handles list inputs correctly
        detector.train(train_images, train_points)
        training_time = time.time() - start_time
        logger.info(f"Intersection detector training completed in {training_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"An error occurred during intersection detector training: {e}")
        # Optionally log more details, e.g., traceback
        return # Stop if training fails

    # Save model
    try:
        model_path = os.path.join(args.model_dir, "intersection_detector.h5") # Or other format
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        detector.save(model_path)
        logger.info(f"Intersection detector model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save intersection detector model to {model_path}: {e}")


def train_digit_recognizer(args):
    """Train digit recognizer model."""
    logger.info("Training digit recognizer...")

    # Load training data - Request NumPy arrays using as_numpy=True
    try:
        logger.info(f"Loading digit data from {args.processed_dir} (prefix: train_digits)")
        # =============================================
        # MODIFIED LINE BELOW (Option 1 Applied)
        # =============================================
        train_data, train_labels = load_dataset(args.processed_dir, "train_digits", as_numpy=True) # <<< ADDED as_numpy=True
        # =============================================
        if len(train_data) == 0: # Check length using len() which works on numpy arrays too
             logger.warning("No training data loaded for digit recognizer. Skipping training.")
             return
    except Exception as e:
        logger.error(f"Failed to load digit training data: {e}")
        return # Cannot train without data

    # Create model
    try:
        recognizer = RobustDigitRecognizer() # Assumes constructor needs no args
    except NameError:
         logger.error("RobustDigitRecognizer class not found. Cannot train.")
         return
    except Exception as e:
         logger.error(f"Error initializing RobustDigitRecognizer: {e}")
         return

    # Train model
    try:
        logger.info(f"Starting training for digit recognizer with {len(train_data)} samples...")
        start_time = time.time()
        # Recognizer expects NumPy arrays, which load_dataset now provides
        recognizer.train(train_data, train_labels)
        training_time = time.time() - start_time
        logger.info(f"Digit recognizer training completed in {training_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"An error occurred during digit recognizer training: {e}")
        # Optionally log more details, e.g., traceback
        # This is where the original 'list' object error would have occurred
        return # Stop if training fails

    # Save model
    try:
        model_path = os.path.join(args.model_dir, "digit_recognizer.h5") # Or other format
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        recognizer.save(model_path)
        logger.info(f"Digit recognizer model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save digit recognizer model to {model_path}: {e}")


def main():
    """Main entry point for training script."""
    # Parse arguments
    args = parse_args()

    # Set up exception handling
    # setup_exception_handling() # Call this if defined in error_handling utils

    logger.info("Starting training script...")
    logger.info(f"Arguments: {args}")

    # Initialize settings
    try:
        # settings = initialize_settings(args.config) # Use settings if needed
        logger.info(f"Configuration loaded from {args.config}") # Placeholder
    except FileNotFoundError:
         logger.error(f"Configuration file not found at {args.config}. Exiting.")
         sys.exit(1)
    except Exception as e:
         logger.error(f"Error loading configuration from {args.config}: {e}")
         sys.exit(1)

    # Create directories if they don't exist
    try:
        os.makedirs(args.processed_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)
        logger.info(f"Ensured processed data directory exists: {args.processed_dir}")
        logger.info(f"Ensured model directory exists: {args.model_dir}")
    except OSError as e:
        logger.error(f"Error creating directories: {e}. Exiting.")
        sys.exit(1)

    # Step 1: Prepare data (unless skipped)
    try:
        prepare_data(args)
    except Exception as e:
        logger.error(f"Data preparation failed with error: {e}. Stopping.")
        # Depending on severity, might exit or just skip training
        sys.exit(1) # Exit if preparation is critical and failed


    # Stop if only preparing data
    if args.prepare_only:
        logger.info("Data preparation requested only (--prepare-only). Exiting now.")
        return # Exit successfully after preparation

    # Step 2: Train selected models
    logger.info(f"Proceeding to train models selected: {args.model}")

    training_errors = False # Flag to track if any training step failed

    # Train intersection detector if requested
    if args.model in ['all', 'intersection']:
        try:
            train_intersection_detector(args)
        except Exception as e:
            logger.error(f"Failed to train intersection detector: {e}")
            training_errors = True # Mark failure but continue if possible

    # Train digit recognizer if requested
    if args.model in ['all', 'digit']:
        try:
            train_digit_recognizer(args)
        except Exception as e:
            logger.error(f"Failed to train digit recognizer: {e}")
            training_errors = True # Mark failure

    # Note on other components (if they existed and required training)
    if args.model in ['all', 'grid']:
        # Assuming Grid Reconstructor doesn't need explicit training from this script
        logger.info("Grid reconstructor component selected (assumed no training step here).")

    if args.model in ['all', 'cell']:
        # Assuming Cell Extractor doesn't need explicit training from this script
        logger.info("Cell extractor component selected (assumed no training step here).")

    if args.model in ['all', 'solver']:
        # Assuming Solver doesn't need explicit training from this script
        logger.info("Solver component selected (assumed no training step here).")

    # Final status message
    if training_errors:
         logger.warning("Training script completed, but one or more training steps encountered errors.")
    else:
         logger.info("Training script completed successfully.")


if __name__ == "__main__":
    main()
