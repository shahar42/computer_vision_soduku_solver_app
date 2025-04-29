#!/usr/bin/env python3
"""
Evaluation Script for Sudoku Recognizer Models.

This script evaluates all models in the Sudoku recognizer system.
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
from utils.data_preparation import load_dataset
from utils.error_handling import setup_exception_handling
from utils.metrics import (
    evaluate_intersection_detector, evaluate_digit_recognizer, 
    evaluate_sudoku_solver, evaluate_full_pipeline, generate_evaluation_report
)
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
        logging.FileHandler('evaluate.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Sudoku recognizer models')
    
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Directory containing processed test data'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='data/models',
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.json',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--component',
        type=str,
        choices=['all', 'digit', 'intersection', 'solver', 'pipeline'],
        default='all',
        help='Which component to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of test samples to use (default: all)'
    )
    
    return parser.parse_args()


def evaluate_components(args):
    """Evaluate all system components."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    logger.info(f"Loading models from {args.model_dir}")
    
    # Initialize pipeline with all components
    pipeline = SudokuRecognizerPipeline()
    pipeline.load_models(args.model_dir)
    
    # Get individual components
    intersection_detector = pipeline.intersection_detector
    digit_recognizer = pipeline.digit_recognizer
    solver = pipeline.solver
    
    # Results dictionary
    results = {}
    
    # Evaluate intersection detector
    if args.component in ['all', 'intersection']:
        logger.info("Evaluating intersection detector")
        
        # Load test data
        test_images, test_points = load_dataset(args.processed_dir, "test_intersections")
        
        # Limit samples if requested
        if args.num_samples is not None:
            test_images = test_images[:args.num_samples]
            test_points = test_points[:args.num_samples]
            
        # Evaluate
        intersection_results = evaluate_intersection_detector(
            intersection_detector,
            test_images,
            test_points,
            save_dir=os.path.join(args.output_dir, "intersection_detector")
        )
        
        results["intersection_detector"] = intersection_results
        
        logger.info(f"Intersection detector precision: {intersection_results['precision']:.4f}")
        logger.info(f"Intersection detector recall: {intersection_results['recall']:.4f}")
        logger.info(f"Intersection detector F1 score: {intersection_results['f1_score']:.4f}")
    
    # Evaluate digit recognizer
    if args.component in ['all', 'digit']:
        logger.info("Evaluating digit recognizer")
        
        # Load test data
        test_cells, test_labels = load_dataset(args.processed_dir, "test_digits")
        
        # Limit samples if requested
        if args.num_samples is not None:
            test_cells = test_cells[:args.num_samples]
            test_labels = test_labels[:args.num_samples]
            
        # Evaluate
        digit_results = evaluate_digit_recognizer(
            digit_recognizer,
            test_cells,
            test_labels,
            save_dir=os.path.join(args.output_dir, "digit_recognizer")
        )
        
        results["digit_recognizer"] = digit_results
        
        logger.info(f"Digit recognizer accuracy: {digit_results['accuracy']:.4f}")
        logger.info(f"Digit recognizer precision: {digit_results['precision']:.4f}")
        logger.info(f"Digit recognizer recall: {digit_results['recall']:.4f}")
        logger.info(f"Digit recognizer F1 score: {digit_results['f1_score']:.4f}")
    
    # Evaluate solver
    if args.component in ['all', 'solver']:
        logger.info("Evaluating Sudoku solver")
        
        # Load test data - we need to construct test puzzles
        # Use original data for this
        original_images, original_grids = load_dataset(args.processed_dir, "original")
        
        # Create puzzles with some cells empty
        test_puzzles = []
        for grid in original_grids:
            # Create a copy
            puzzle = [row[:] for row in grid]
            
            # Randomly remove some cells (about 50%)
            for i in range(9):
                for j in range(9):
                    if np.random.rand() < 0.5:
                        puzzle[i][j] = 0
                        
            test_puzzles.append(puzzle)
            
        # Limit samples if requested
        if args.num_samples is not None:
            test_puzzles = test_puzzles[:args.num_samples]
            
        # Evaluate
        solver_results = evaluate_sudoku_solver(
            solver,
            test_puzzles,
            save_dir=os.path.join(args.output_dir, "solver")
        )
        
        results["solver"] = solver_results
        
        logger.info(f"Solver success rate: {solver_results['success_rate']:.4f}")
        logger.info(f"Solver average time: {solver_results['average_solving_time']:.4f}s")
    
    # Evaluate full pipeline
    if args.component in ['all', 'pipeline']:
        logger.info("Evaluating full pipeline")
        
        # Load test data
        original_images, original_grids = load_dataset(args.processed_dir, "original")
        
        # Limit samples if requested
        if args.num_samples is not None:
            original_images = original_images[:args.num_samples]
            original_grids = original_grids[:args.num_samples]
            
        # Evaluate
        pipeline_results = evaluate_full_pipeline(
            pipeline,
            original_images,
            original_grids,
            save_dir=os.path.join(args.output_dir, "pipeline")
        )
        
        results["pipeline"] = pipeline_results
        
        logger.info(f"Pipeline grid detection rate: {pipeline_results['grid_detection_rate']:.4f}")
        logger.info(f"Pipeline cell extraction rate: {pipeline_results['cell_extraction_rate']:.4f}")
        logger.info(f"Pipeline digit recognition rate: {pipeline_results['digit_recognition_rate']:.4f}")
        logger.info(f"Pipeline solving rate: {pipeline_results['solving_rate']:.4f}")
        logger.info(f"Pipeline end-to-end rate: {pipeline_results['end_to_end_rate']:.4f}")
    
    # Generate comprehensive report
    report = generate_evaluation_report(
        results,
        save_path=os.path.join(args.output_dir, "evaluation_report.txt")
    )
    
    # Save results as JSON
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        # Convert numpy values to Python types for JSON serialization
        serializable_results = {}
        for component, metrics in results.items():
            serializable_results[component] = {}
            for metric, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_results[component][metric] = value.tolist()
                elif isinstance(value, np.generic):
                    serializable_results[component][metric] = value.item()
                else:
                    serializable_results[component][metric] = value
                    
        json.dump(serializable_results, f, indent=2)
        
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")
    
    return results


def main():
    """Main entry point for evaluation script."""
    # Parse arguments
    args = parse_args()
    
    # Set up exception handling
    setup_exception_handling()
    
    # Initialize settings
    settings = initialize_settings(args.config)
    
    # Evaluate components
    results = evaluate_components(args)
    
    # Print summary to console
    if "pipeline" in results:
        pipeline_results = results["pipeline"]
        print("\nPipeline Performance Summary:")
        print(f"Grid Detection Rate: {pipeline_results['grid_detection_rate']:.4f}")
        print(f"Cell Extraction Rate: {pipeline_results['cell_extraction_rate']:.4f}")
        print(f"Digit Recognition Rate: {pipeline_results['digit_recognition_rate']:.4f}")
        print(f"Solving Rate: {pipeline_results['solving_rate']:.4f}")
        print(f"End-to-End Success Rate: {pipeline_results['end_to_end_rate']:.4f}")
        print(f"Average Processing Time: {pipeline_results['average_processing_time']:.4f}s")
        
    logger.info("Evaluation script completed successfully")


if __name__ == "__main__":
    main()
