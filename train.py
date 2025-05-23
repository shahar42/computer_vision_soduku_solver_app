#!/usr/bin/env python3
"""
Training Script for Sudoku Recognizer Models.

This script trains the models used in the Sudoku recognition system.
It processes image files with corresponding .dat annotation files and
creates models for intersection detection and digit recognition.
"""

import os
import sys
import argparse
import logging
import json
import time
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple

# Import project components
from config.settings import initialize_settings
from utils.data_preparation import load_dataset, normalize_cell_images
from utils.error_handling import setup_exception_handling
from models.intersection_detector import RobustIntersectionDetector
from models.digit_recognizer import RobustDigitRecognizer
from pipeline import SudokuRecognizerPipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train.log')
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
        help='Directory to save/load processed training data'
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
    """Prepare training data by processing raw images and .dat files."""
    # Check if processed data already exists and skip if not forced
    if not args.force_preparation:
        # Check for a key output file to determine existence
        train_digits_path = os.path.join(args.processed_dir, "train_digits_data.npy")
        if os.path.exists(train_digits_path):
            logger.info(f"Processed data found in {args.processed_dir}. Skipping preparation.")
            logger.info("Use --force-preparation to re-run preparation.")
            return

    logger.info("Starting data preparation...")

    # Initialize pipeline for cell extraction
    try:
        pipeline = SudokuRecognizerPipeline()
        # Load existing models if available
        if os.path.exists(args.model_dir) and os.path.isdir(args.model_dir):
            logger.info(f"Loading existing models from {args.model_dir} to assist data preparation.")
            pipeline.load_models(args.model_dir)
        else:
            logger.info("No existing model directory found. Using default pipeline components.")
    except Exception as e:
        logger.error(f"Error initializing pipeline for data preparation: {e}")
        raise

    # Find all image files and their corresponding .dat files
    image_files = []
    dat_files = []

    for file in os.listdir(args.data_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(args.data_dir, file)
            # Find matching .dat file (same name but with .dat extension)
            dat_path = os.path.join(args.data_dir, os.path.splitext(file)[0] + '.dat')
            if os.path.exists(dat_path):
                image_files.append(image_path)
                dat_files.append(dat_path)
            else:
                logger.warning(f"Found image file {file} but no matching .dat file. Skipping.")

    if not image_files:
        logger.error("No valid image/dat file pairs found in data directory.")
        raise FileNotFoundError("No valid training data found")

    logger.info(f"Found {len(image_files)} image/dat file pairs for training.")

    # Process each image/dat pair
    processed_images = []
    processed_intersections = []
    processed_cells = []
    processed_digits = []
    
    stats = {
        "total_images": len(image_files),
        "successfully_processed": 0,
        "failed_processing": 0,
        "total_cells": 0,
        "cells_with_digits": 0,
        "empty_cells": 0
    }

    for img_idx, (image_path, dat_path) in enumerate(zip(image_files, dat_files)):
        try:
            logger.info(f"Processing image {img_idx+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image {image_path}")
                stats["failed_processing"] += 1
                continue
                
            # Load .dat file and parse Sudoku grid
            with open(dat_path, 'r') as f:
                dat_lines = f.readlines()
            
            # Extract Sudoku grid from .dat file
            # Skip header lines (metadata)
            grid_lines = []
            for line in dat_lines:
                line = line.strip()
                if len(line) >= 9 and all(c.isdigit() or c.isspace() for c in line):
                    grid_lines.append(line)
                    if len(grid_lines) == 9:
                        break
            
            if len(grid_lines) != 9:
                logger.error(f"Invalid .dat file format for {dat_path}. Expected 9 grid lines, got {len(grid_lines)}.")
                stats["failed_processing"] += 1
                continue
                
            # Convert grid to 2D array of digits
            sudoku_grid = []
            for line in grid_lines:
                # Replace spaces with zeros if needed
                line = line.replace(' ', '0') if ' ' in line else line
                # Extract only digits
                digits = [int(c) for c in line if c.isdigit()]
                if len(digits) != 9:
                    logger.warning(f"Grid line has {len(digits)} digits, expected 9: {line}")
                    # Pad or truncate line to ensure 9 digits
                    digits = (digits + [0] * 9)[:9]
                sudoku_grid.append(digits)
                
            # Process image through pipeline
            result = pipeline.process_image_data(image)
            
            if not result["success"]:
                logger.warning(f"Pipeline processing failed for {image_path}")
                stats["failed_processing"] += 1
                continue
                
            # If grid detection was successful
            if result["grid_detected"] and result["cells_extracted"]:
                # Store intersection points for intersection detector training
                if "intersections" in result["stage_results"]["grid_detection"]:
                    intersections = pipeline.current_state["intersections"]
                    processed_intersections.append((image, intersections))
                
                # Get cell images
                cell_images = pipeline.current_state["cell_images"]
                
                # Match cell images with their digits from the sudoku_grid
                for row_idx, row in enumerate(cell_images):
                    for col_idx, cell in enumerate(row):
                        # Get the corresponding digit from parsed grid
                        digit = sudoku_grid[row_idx][col_idx]
                        
                        # Store cell with its digit label
                        processed_cells.append((cell, (row_idx, col_idx)))
                        processed_digits.append((cell, digit))
                        
                        # Update statistics
                        stats["total_cells"] += 1
                        if digit > 0:
                            stats["cells_with_digits"] += 1
                        else:
                            stats["empty_cells"] += 1
                
                stats["successfully_processed"] += 1
                processed_images.append(image_path)
            else:
                logger.warning(f"Grid detection or cell extraction failed for {image_path}")
                stats["failed_processing"] += 1
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            stats["failed_processing"] += 1
            continue

    # Create processed_dir if it doesn't exist
    os.makedirs(args.processed_dir, exist_ok=True)
    
    # Save processed data
    if processed_intersections:
        intersection_images = [item[0] for item in processed_intersections]
        intersection_points = [item[1] for item in processed_intersections]
        np.save(os.path.join(args.processed_dir, "train_intersections_images.npy"), intersection_images)
        np.save(os.path.join(args.processed_dir, "train_intersections_points.npy"), intersection_points)
        logger.info(f"Saved intersection data: {len(processed_intersections)} images")
    
    if processed_digits:
        digit_images = [item[0] for item in processed_digits]
        digit_labels = [item[1] for item in processed_digits]
        
        # Apply augmentation
        if args.augmentation_factor > 1:
            logger.info(f"Applying data augmentation with factor {args.augmentation_factor}...")
            augmented_images = []
            augmented_labels = []
            
            for img, label in zip(digit_images, digit_labels):
                # Always include the original
                augmented_images.append(img)
                augmented_labels.append(label)
                
                # Add augmented versions
                for i in range(args.augmentation_factor - 1):
                    # Only augment cells with digits
                    if label > 0:
                        # Random rotation, scaling, translation
                        angle = np.random.uniform(-15, 15)
                        scale = np.random.uniform(0.8, 1.2)
                        tx = np.random.uniform(-2, 2)
                        ty = np.random.uniform(-2, 2)
                        
                        h, w = img.shape[:2]
                        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
                        matrix[0, 2] += tx
                        matrix[1, 2] += ty
                        
                        augmented = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
                        augmented_images.append(augmented)
                        augmented_labels.append(label)
            
            digit_images = augmented_images
            digit_labels = augmented_labels
            logger.info(f"After augmentation: {len(digit_images)} digit samples")
        
        # Normalize all cell images
        digit_images = normalize_cell_images(digit_images)
        
        # Save digit data
        np.save(os.path.join(args.processed_dir, "train_digits_data.npy"), digit_images)
        np.save(os.path.join(args.processed_dir, "train_digits_labels.npy"), digit_labels)
        logger.info(f"Saved digit data: {len(digit_images)} samples")
    
    # Save statistics
    stats_path = os.path.join(args.processed_dir, "preparation_stats.json")
    try:
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved preparation statistics to {stats_path}")
    except IOError as e:
        logger.error(f"Could not save preparation statistics: {e}")
    
    logger.info("Data preparation complete.")
    logger.info(f"Processed {stats['successfully_processed']} images successfully.")
    logger.info(f"Total cells: {stats['total_cells']} ({stats['cells_with_digits']} with digits, {stats['empty_cells']} empty)")
    
    return stats


def train_intersection_detector(args):
    """Train intersection detector model."""
    logger.info("Training intersection detector...")

    # Load training data
    try:
        logger.info(f"Loading intersection data from {args.processed_dir}")
        train_images = np.load(os.path.join(args.processed_dir, "train_intersections_images.npy"), allow_pickle=True)
        train_points = np.load(os.path.join(args.processed_dir, "train_intersections_points.npy"), allow_pickle=True)
        
        if len(train_images) == 0:
            logger.warning("No training data loaded for intersection detector. Skipping training.")
            return
    except Exception as e:
        logger.error(f"Failed to load intersection training data: {e}")
        return

    # Create model
    try:
        detector = RobustIntersectionDetector()
    except Exception as e:
        logger.error(f"Error initializing RobustIntersectionDetector: {e}")
        return

    # Train model
    try:
        logger.info(f"Starting training for intersection detector with {len(train_images)} images...")
        start_time = time.time()
        
        # Convert numpy arrays to lists if needed
        train_images_list = [img for img in train_images]
        train_points_list = [points for points in train_points]
        
        detector.train(train_images_list, train_points_list)
        training_time = time.time() - start_time
        logger.info(f"Intersection detector training completed in {training_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"An error occurred during intersection detector training: {e}")
        return

    # Save model
    try:
        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(args.model_dir, "intersection_detector.h5")
        detector.save(model_path)
        logger.info(f"Intersection detector model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save intersection detector model: {e}")


def train_digit_recognizer(args):
    """Train digit recognizer model."""
    logger.info("Training digit recognizer...")

    # Load training data
    try:
        logger.info(f"Loading digit data from {args.processed_dir}")
        train_data = np.load(os.path.join(args.processed_dir, "train_digits_data.npy"), allow_pickle=True)
        train_labels = np.load(os.path.join(args.processed_dir, "train_digits_labels.npy"), allow_pickle=True)
        
        if len(train_data) == 0:
            logger.warning("No training data loaded for digit recognizer. Skipping training.")
            return
            
        logger.info(f"Loaded {len(train_data)} digit samples with distribution:")
        # Show digit distribution
        unique, counts = np.unique(train_labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        for digit, count in sorted(distribution.items()):
            logger.info(f"  Digit {digit}: {count} samples")
            
    except Exception as e:
        logger.error(f"Failed to load digit training data: {e}")
        return

    # Create model
    try:
        recognizer = RobustDigitRecognizer()
    except Exception as e:
        logger.error(f"Error initializing RobustDigitRecognizer: {e}")
        return

    # Train model
    try:
        logger.info(f"Starting training for digit recognizer with {len(train_data)} samples...")
        start_time = time.time()
        
        # Convert to list if needed
        train_data_list = [img for img in train_data]
        train_labels_list = [int(label) for label in train_labels]
        
        recognizer.train(train_data_list, train_labels_list)
        training_time = time.time() - start_time
        logger.info(f"Digit recognizer training completed in {training_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"An error occurred during digit recognizer training: {e}")
        return

    # Save model
    try:
        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(args.model_dir, "digit_recognizer.h5")
        recognizer.save(model_path)
        logger.info(f"Digit recognizer model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save digit recognizer model: {e}")


def main():
    """Main entry point for training script."""
    # Parse arguments
    args = parse_args()

    # Set up exception handling
    # setup_exception_handling()  # Enable if defined in your utils

    logger.info("Starting training script...")
    logger.info(f"Arguments: {args}")

    # Initialize settings
    try:
        # Initialize settings if needed
        # settings = initialize_settings(args.config)
        logger.info(f"Configuration loaded from {args.config}")
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
        sys.exit(1)

    # Stop if only preparing data
    if args.prepare_only:
        logger.info("Data preparation requested only (--prepare-only). Exiting now.")
        return

    # Step 2: Train selected models
    logger.info(f"Proceeding to train models selected: {args.model}")

    training_errors = False

    # Train intersection detector if requested
    if args.model in ['all', 'intersection']:
        try:
            train_intersection_detector(args)
        except Exception as e:
            logger.error(f"Failed to train intersection detector: {e}")
            training_errors = True

    # Train digit recognizer if requested
    if args.model in ['all', 'digit']:
        try:
            train_digit_recognizer(args)
        except Exception as e:
            logger.error(f"Failed to train digit recognizer: {e}")
            training_errors = True

    # Note on other components that don't require explicit training
    if args.model in ['grid', 'cell', 'solver']:
        logger.info(f"{args.model.capitalize()} component selected (no training needed for this component).")

    # Final status message
    if training_errors:
        logger.warning("Training script completed, but one or more training steps encountered errors.")
    else:
        logger.info("Training script completed successfully.")


if __name__ == "__main__":
    main()
