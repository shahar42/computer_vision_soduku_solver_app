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
            
    # Create pipeline for data preparation
    pipeline = SudokuRecognizerPipeline()
    
    # Load models if available
    model_dir = args.model_dir
    if os.path.exists(model_dir):
        logger.info(f"Loading models from {model_dir}")
        pipeline.load_models(model_dir)
        
    # Prepare training data
    logger.info(f"Preparing training data from {args.data_dir}")
    stats = prepare_training_data(
        args.data_dir,
        args.processed_dir,
        pipeline,
        args.augmentation_factor
    )
    
    # Log statistics
    logger.info("Data preparation complete")
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Total cells: {stats['total_cells']}")
    logger.info(f"Augmented cells: {stats['augmented_cells']}")
    logger.info(f"Training digits: {stats['train_digits']}")
    logger.info(f"Testing digits: {stats['test_digits']}")
    logger.info(f"Training intersections: {stats['train_intersections']}")
    logger.info(f"Testing intersections: {stats['test_intersections']}")
    
    # Save statistics
    with open(os.path.join(args.processed_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def train_intersection_detector(args):
    """Train intersection detector model."""
    logger.info("Training intersection detector")
    
    # Load training data
    train_images, train_points = load_dataset(args.processed_dir, "train_intersections")
    
    # Create model
    detector = RobustIntersectionDetector()
    
    # Train model
    start_time = time.time()
    detector.train(train_images, train_points)
    training_time = time.time() - start_time
    
    # Save model
    model_path = os.path.join(args.model_dir, "intersection_detector.h5")
    detector.save(model_path)
    
    logger.info(f"Intersection detector trained in {training_time:.2f} seconds")
    logger.info(f"Model saved to {model_path}")


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
    settings = initialize_settings(args.config)
    
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
    if args.model in ['all', 'intersection']:
        train_intersection_detector(args)
        
    if args.model in ['all', 'digit']:
        train_digit_recognizer(args)
        
    # Grid reconstructor, cell extractor, and solver don't require training
    if args.model in ['all', 'grid']:
        logger.info("Grid reconstructor does not require training")
        
    if args.model in ['all', 'cell']:
        logger.info("Cell extractor does not require training")
        
    if args.model in ['all', 'solver']:
        logger.info("Solver does not require training")
        
    logger.info("Training complete")


if __name__ == "__main__":
    main()
