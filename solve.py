#!/usr/bin/env python3
"""
Sudoku Solver Command-line Tool.

This script processes Sudoku puzzles from images or text files and solves them.
"""

import os
import sys
import argparse
import logging
import time
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

from config.settings import initialize_settings
from utils.error_handling import setup_exception_handling
from utils.visualization import (
    visualize_digit_grid, visualize_solution, create_visualization_report,
    overlay_solution_on_image
)
from pipeline import SudokuRecognizerPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('solve.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Solve Sudoku puzzles from images or text files')
    
    parser.add_argument(
        'input',
        type=str,
        help='Input image path or text file with Sudoku grid'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='data/models',
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.json',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize results'
    )
    
    parser.add_argument(
        '--text-output',
        action='store_true',
        help='Output solution as text file'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    return parser.parse_args()


def is_image_file(file_path: str) -> bool:
    """Check if file is an image."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    ext = os.path.splitext(file_path)[1].lower()
    return ext in image_extensions


def parse_text_file(file_path: str) -> List[List[int]]:
    """Parse a text file containing a Sudoku grid."""
    grid = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # Process each line
    for line in lines:
        # Remove comments and whitespace
        line = line.split('#')[0].strip()
        if not line:
            continue
            
        # Parse row
        row = []
        for char in line:
            if char.isdigit():
                row.append(int(char))
            elif char in '.0_':  # Common empty cell indicators
                row.append(0)
                
        # Skip incomplete rows
        if len(row) != 9:
            continue
            
        grid.append(row)
        
        # Stop after 9 rows
        if len(grid) == 9:
            break
            
    # Validate grid
    if len(grid) != 9:
        raise ValueError(f"Invalid grid: {len(grid)} rows, expected 9")
        
    return grid


def solve_from_image(args, pipeline: SudokuRecognizerPipeline):
    """Solve Sudoku from image file."""
    logger.info(f"Processing image: {args.input}")
    
    try:
        # Process image
        results = pipeline.process_image(args.input)
        
        # Extract results
        grid_detected = results.get("grid_detected", False)
        cells_extracted = results.get("cells_extracted", False)
        digits_recognized = results.get("digits_recognized", False)
        puzzle_solved = results.get("puzzle_solved", False)
        digit_grid = results.get("digit_grid")
        confidence_grid = results.get("confidence_grid")
        solved_grid = results.get("solved_grid")
        
        # Log results
        logger.info(f"Grid detection: {'Success' if grid_detected else 'Failure'}")
        logger.info(f"Cell extraction: {'Success' if cells_extracted else 'Failure'}")
        logger.info(f"Digit recognition: {'Success' if digits_recognized else 'Failure'}")
        logger.info(f"Puzzle solving: {'Success' if puzzle_solved else 'Failure'}")
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Save results
        if digits_recognized and digit_grid is not None:
            # Save recognized grid visualization
            visualize_digit_grid(
                digit_grid,
                confidence_grid,
                save_path=os.path.join(args.output, "recognized_digits.png")
            )
            
            # Save recognized grid as text
            if args.text_output:
                with open(os.path.join(args.output, "recognized_grid.txt"), "w") as f:
                    for row in digit_grid:
                        f.write("".join(str(d) for d in row) + "\n")
                        
        if puzzle_solved and solved_grid is not None:
            # Save solved grid visualization
            if digit_grid is not None:
                visualize_solution(
                    digit_grid,
                    solved_grid,
                    save_path=os.path.join(args.output, "solution.png")
                )
                
                # Load original image
                original_image = cv2.imread(args.input)
                
                # Overlay solution on original image
                if original_image is not None and pipeline.current_state["grid_points"] is not None:
                    overlay_solution_on_image(
                        original_image,
                        digit_grid,
                        solved_grid,
                        pipeline.current_state["grid_points"],
                        save_path=os.path.join(args.output, "overlay.png")
                    )
                    
            # Save solved grid as text
            if args.text_output:
                with open(os.path.join(args.output, "solution.txt"), "w") as f:
                    for row in solved_grid:
                        f.write("".join(str(d) for d in row) + "\n")
                        
        # Create comprehensive report
        if original_image is not None:
            create_visualization_report(
                results,
                original_image,
                save_path=os.path.join(args.output, "report.png")
            )
            
        # Show visualizations if requested
        if args.visualize:
            if digits_recognized and digit_grid is not None:
                visualize_digit_grid(
                    digit_grid,
                    confidence_grid,
                    show=True
                )
                
            if puzzle_solved and solved_grid is not None and digit_grid is not None:
                visualize_solution(
                    digit_grid,
                    solved_grid,
                    show=True
                )
                
        # Print summary
        print("\nSudoku Processing Summary:")
        print(f"Grid Detection: {'Success' if grid_detected else 'Failure'}")
        print(f"Cell Extraction: {'Success' if cells_extracted else 'Failure'}")
        print(f"Digit Recognition: {'Success' if digits_recognized else 'Failure'}")
        print(f"Puzzle Solving: {'Success' if puzzle_solved else 'Failure'}")
        print(f"Results saved to: {args.output}")
        
        if puzzle_solved and solved_grid is not None:
            print("\nSolution:")
            for row in solved_grid:
                print(" ".join(str(d) for d in row))
                
        return results
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


def solve_from_text(args, pipeline: SudokuRecognizerPipeline):
    """Solve Sudoku from text file."""
    logger.info(f"Processing text file: {args.input}")
    
    try:
        # Parse text file
        grid = parse_text_file(args.input)
        
        # Print input grid
        print("\nInput Grid:")
        for row in grid:
            print(" ".join(str(d) for d in row))
            
        # Solve puzzle
        start_time = time.time()
        solved_grid = pipeline.solver.solve(grid)
        solving_time = time.time() - start_time
        
        logger.info(f"Puzzle solved in {solving_time:.4f} seconds")
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Save results
        visualize_solution(
            grid,
            solved_grid,
            save_path=os.path.join(args.output, "solution.png")
        )
        
        if args.text_output:
            with open(os.path.join(args.output, "solution.txt"), "w") as f:
                for row in solved_grid:
                    f.write("".join(str(d) for d in row) + "\n")
                    
        # Show visualization if requested
        if args.visualize:
            visualize_solution(
                grid,
                solved_grid,
                show=True
            )
            
        # Print solution
        print("\nSolution:")
        for row in solved_grid:
            print(" ".join(str(d) for d in row))
            
        print(f"\nResults saved to: {args.output}")
        
        return {
            "initial_grid": grid,
            "solved_grid": solved_grid,
            "solving_time": solving_time
        }
        
    except Exception as e:
        logger.error(f"Error processing text file: {str(e)}")
        raise


def main():
    """Main entry point for solve script."""
    # Parse arguments
    args = parse_args()
    
    # Set up exception handling
    setup_exception_handling()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Initialize settings
    settings = initialize_settings(args.config)
    
    # Create pipeline
    pipeline = SudokuRecognizerPipeline()
    
    # Load models
    logger.info(f"Loading models from {args.model_dir}")
    if not pipeline.load_models(args.model_dir):
        logger.error(f"Failed to load models from {args.model_dir}")
        return 1
        
    # Process input
    try:
        if is_image_file(args.input):
            solve_from_image(args, pipeline)
        else:
            solve_from_text(args, pipeline)
            
        logger.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
