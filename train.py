#!/usr/bin/env python3
"""
Training Script for Sudoku Recognizer Models.
"""

import os
import sys
import argparse
import logging
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional

from config import initialize_settings
from utils.data_preparation import (
    load_image_files, load_training_data, generate_digit_dataset,
    augment_digit_dataset, save_dataset, load_dataset
)
from utils.error_handling import setup_exception_handling
from models.digit_recognizer import RobustDigitRecognizer

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
        help='Directory containing raw training data'
    )
    
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Directory for processed training data'
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
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Only prepare training data, do not train models'
    )
    
    parser.add_argument(
        '--augmentation-factor',
        type=int,
        default=3,
        help='Data augmentation factor'
    )
    
    parser.add_argument(
        '--force-preparation',
        action='store_true',
        help='Force re-preparation of training data even if it exists'
    )
    
    return parser.parse_args()

def prepare_data(args):
    """Prepare training data."""
    # Check if processed data already exists
    if not args.force_preparation:
        train_digits_path = os.path.join(args.processed_dir, "train_digits_data.npy")
        if os.path.exists(train_digits_path):
            logger.info(f"Processed data already exists at {args.processed_dir}")
            logger.info("Use --force-preparation to force re-preparation")
            return
            
    # Load raw image data
    logger.info(f"Loading images from {args.data_dir}")
    image_data_pairs = load_image_files(args.data_dir)
    
    if not image_data_pairs:
        logger.error(f"No valid data found in {args.data_dir}")
        return
        
    # Load training data
    logger.info("Loading training data")
    images, grids = load_training_data(image_data_pairs)
    
    if not images or not grids:
        logger.error("Failed to load training data")
        return
        
    # For now, we'll use a simplified approach for creating cell images
    # In a full implementation, we would use the pipeline to extract cells
    
    # Create a simple dataset of synthetic "cells"
    logger.info("Creating synthetic cell dataset")
    cell_grids = []
    
    # For each grid, create synthetic cells
    for i, grid in enumerate(grids):
        cell_grid = []
        for row in range(9):
            cell_row = []
            for col in range(9):
                # Create a blank cell image
                cell = np.zeros((28, 28), dtype=np.uint8)
                
                # If there's a digit, draw it in the center
                digit = grid[row][col]
                if digit > 0:
                    # Simple representation - just put the digit value in the center
                    cv2.putText(
                        cell, 
                        str(digit), 
                        (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        255, 
                        2
                    )
                    
                cell_row.append(cell)
            cell_grid.append(cell_row)
        cell_grids.append(cell_grid)
        
    # Generate digit dataset
    logger.info("Generating digit dataset")
    digit_images, digit_labels = generate_digit_dataset(cell_grids, grids)
    
    if not digit_images or not digit_labels:
        logger.error("Failed to generate digit dataset")
        return
        
    # Augment digit dataset
    logger.info(f"Augmenting digit dataset (factor: {args.augmentation_factor})")
    augmented_images, augmented_labels = augment_digit_dataset(
        digit_images, digit_labels, args.augmentation_factor
    )
    
    # Split datasets for training and testing
    train_ratio = 0.8
    train_size = int(len(augmented_images) * train_ratio)
    
    train_digits = augmented_images[:train_size]
    train_labels = augmented_labels[:train_size]
    test_digits = augmented_images[train_size:]
    test_labels = augmented_labels[train_size:]
    
    # Save datasets
    os.makedirs(args.processed_dir, exist_ok=True)
    
    logger.info("Saving digit datasets")
    save_dataset(train_digits, train_labels, args.processed_dir, "train_digits")
    save_dataset(test_digits, test_labels, args.processed_dir, "test_digits")
    
    # Prepare stats
    stats = {
        "total_images": len(images),
        "total_cells": len(digit_images),
        "augmented_cells": len(augmented_images),
        "train_digits": len(train_digits),
        "test_digits": len(test_digits)
    }
    
    # Save statistics
    with open(os.path.join(args.processed_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    # Log statistics
    logger.info("Data preparation complete")
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Total cells: {stats['total_cells']}")
    logger.info(f"Augmented cells: {stats['augmented_cells']}")
    logger.info(f"Training digits: {stats['train_digits']}")
    logger.info(f"Testing digits: {stats['test_digits']}")

def train_digit_recognizer(args):
    """Train digit recognizer model."""
    logger.info("Training digit recognizer")
    
    # Load training data
    train_data, train_labels = load_dataset(args.processed_dir, "train_digits")
    
    # Create model
    recognizer = RobustDigitRecognizer()
    
    # Train model
    start_time = time.time()
    recognizer.train(train_data, train_labels)
    training_time = time.time() - start_time
    
    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "digit_recognizer.h5")
    recognizer.save(model_path)
    
    logger.info(f"Digit recognizer trained in {training_time:.2f} seconds")
    logger.info(f"Model saved to {model_path}")

def main():
    """Main entry point for training script."""
    # Parse arguments
    args = parse_args()
    
    # Set up exception handling
    setup_exception_handling()
    
    # Initialize settings
    initialize_settings(args.config)
    
    # Create directories
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Prepare data
    prepare_data(args)
    
    # Stop if only preparing data
    if args.prepare_only:
        logger.info("Data preparation complete, exiting")
        return
        
    # Train models
    train_digit_recognizer(args)
        
    logger.info("Training complete")

if __name__ == "__main__":
    main()
