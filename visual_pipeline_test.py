#!/usr/bin/env python3
"""
Visual Pipeline Test Script.

This script processes a Sudoku image and saves visualizations at every stage
of the pipeline for visual inspection and debugging.
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
    visualize_intersections, visualize_grid, visualize_cells,
    visualize_digit_grid, visualize_solution, overlay_solution_on_image
)
from pipeline import SudokuRecognizerPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('visual_pipeline_test.log')
    ]
)
logger = logging.getLogger(__name__)


def save_stage_visualization(image, stage_name, output_dir, data=None):
    """Save visualization for a specific pipeline stage."""
    try:
        stage_dir = os.path.join(output_dir, f"{stage_name:02d}_{stage_name}")
        os.makedirs(stage_dir, exist_ok=True)
        
        # Save original stage image
        stage_path = os.path.join(stage_dir, f"{stage_name}_original.jpg")
        cv2.imwrite(stage_path, image)
        
        return stage_path
    except Exception as e:
        logger.error(f"Error saving stage {stage_name}: {str(e)}")
        return None


def visualize_intersections_custom(image, intersections, save_path=None):
    """Visualize detected intersections on the image"""
    if not intersections:
        return image
        
    viz_image = image.copy()
    for i, (x, y) in enumerate(intersections):
        cv2.circle(viz_image, (int(x), int(y)), 4, (0, 0, 255), -1)  # Red points
        
    if save_path:
        cv2.imwrite(save_path, viz_image)
        
    return viz_image


def visualize_grid_custom(image, grid_points, save_path=None):
    """Visualize reconstructed grid on the image"""
    if not grid_points:
        return image
        
    viz_image = image.copy()
    
    # Draw horizontal lines
    for i in range(10):  # 10 horizontal lines for a 9x9 grid
        if i < len(grid_points):
            points = [(int(p[0]), int(p[1])) for p in grid_points[i]]
            for j in range(len(points) - 1):
                cv2.line(viz_image, points[j], points[j+1], (0, 255, 0), 2)
    
    # Draw vertical lines  
    for j in range(10):  # 10 vertical lines for a 9x9 grid
        points = []
        for i in range(min(10, len(grid_points))):
            if j < len(grid_points[i]):
                points.append((int(grid_points[i][j][0]), int(grid_points[i][j][1])))
        
        for k in range(len(points) - 1):
            cv2.line(viz_image, points[k], points[k+1], (0, 255, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, viz_image)
        
    return viz_image


def visualize_cells_custom(cells, save_path=None):
    """Visualize extracted cells in a grid"""
    if not cells or not cells[0]:
        return None
        
    # Calculate cell size for visualization
    cell_size = 64
    border = 2
    
    # Create visualization grid
    grid_width = 9 * (cell_size + border) - border
    grid_height = 9 * (cell_size + border) - border
    viz_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    for i in range(9):
        for j in range(9):
            if i < len(cells) and j < len(cells[i]) and cells[i][j] is not None:
                cell = cells[i][j]
                
                # Resize cell to standard size
                resized_cell = cv2.resize(cell, (cell_size, cell_size))
                
                # Convert to BGR if grayscale
                if len(resized_cell.shape) == 2:
                    resized_cell = cv2.cvtColor(resized_cell, cv2.COLOR_GRAY2BGR)
                
                # Calculate position
                y_start = i * (cell_size + border)
                y_end = y_start + cell_size
                x_start = j * (cell_size + border)
                x_end = x_start + cell_size
                
                # Place cell in grid
                viz_image[y_start:y_end, x_start:x_end] = resized_cell
    
    if save_path:
        cv2.imwrite(save_path, viz_image)
        
    return viz_image


def visualize_digits_custom(digit_grid, confidence_grid=None, save_path=None):
    """Visualize recognized digits in a grid"""
    if not digit_grid:
        return None
        
    # Create visualization
    cell_size = 80
    border = 2
    grid_width = 9 * (cell_size + border) - border
    grid_height = 9 * (cell_size + border) - border
    viz_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    for i in range(9):
        for j in range(9):
            if i < len(digit_grid) and j < len(digit_grid[i]):
                digit = digit_grid[i][j]
                confidence = confidence_grid[i][j] if confidence_grid else 1.0
                
                # Calculate position
                y_start = i * (cell_size + border)
                x_start = j * (cell_size + border)
                
                # Draw cell border
                cv2.rectangle(viz_image, 
                            (x_start, y_start), 
                            (x_start + cell_size, y_start + cell_size), 
                            (200, 200, 200), 1)
                
                # Draw digit if not empty
                if digit != 0:
                    # Color based on confidence
                    color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                    
                    # Draw digit
                    cv2.putText(viz_image, str(digit),
                              (x_start + cell_size//3, y_start + cell_size//2 + 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
                    
                    # Draw confidence
                    if confidence_grid:
                        cv2.putText(viz_image, f"{confidence:.2f}",
                                  (x_start + 5, y_start + cell_size - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    if save_path:
        cv2.imwrite(save_path, viz_image)
        
    return viz_image


def run_visual_pipeline_test(image_path: str, output_dir: str, pipeline: SudokuRecognizerPipeline):
    """Run the complete pipeline with visual output at each stage."""
    print(f"\n🔍 Starting Visual Pipeline Test")
    print(f"📁 Input: {image_path}")
    print(f"📁 Output: {output_dir}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Save original image
    original_path = os.path.join(output_dir, "00_original_image.jpg")
    cv2.imwrite(original_path, original_image)
    print(f"✅ Stage 0: Original image saved to {original_path}")
    
    # Initialize pipeline state tracking
    stage_results = {}
    
    try:
        # Stage 1: Load and preprocess image
        print("\n🎯 Stage 1: Image Loading and Preprocessing")
        pipeline.current_state["image"] = original_image
        
        # Save preprocessed image (if any preprocessing occurs)
        preprocessed_path = os.path.join(output_dir, "01_preprocessed_image.jpg")
        cv2.imwrite(preprocessed_path, original_image)
        print(f"✅ Stage 1: Preprocessed image saved to {preprocessed_path}")
        
        # Stage 2: Board Detection (if enabled)
        print("\n🎯 Stage 2: Board Detection")
        try:
            if pipeline.board_detector and pipeline.use_board_detection:
                # FIXED: Try board detection with proper tuple handling
                board_result = pipeline.board_detector.detect(original_image)
                if board_result is not None:
                    # board_result is a tuple (x1, y1, x2, y2, confidence)
                    # Visualize board detection
                    board_viz = original_image.copy()
                    x1, y1, x2, y2, confidence = board_result
                    cv2.rectangle(board_viz, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(board_viz, f"Board Detected (conf: {confidence:.2f})",
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    board_path = os.path.join(output_dir, "02_board_detection.jpg")
                    cv2.imwrite(board_path, board_viz)
                    print(f"✅ Stage 2: Board detection saved to {board_path}")
                    stage_results["board_detection"] = {"success": True, "bbox": (x1, y1, x2, y2), "confidence": confidence}
                else:
                    print("⚠️  Stage 2: Board detection failed, will use intersection method")
                    stage_results["board_detection"] = {"success": False, "error": "Low confidence or detection failed"}
            else:
                print("ℹ️  Stage 2: Board detection disabled, skipping")
                stage_results["board_detection"] = {"success": False, "error": "Disabled"}
        except Exception as e:
            print(f"❌ Stage 2: Board detection error: {str(e)}")
            stage_results["board_detection"] = {"success": False, "error": str(e)}
        
        # Stage 3: Grid Detection (Intersection Detection)
        print("\n🎯 Stage 3: Grid Detection (Intersection Detection)")
        try:
            grid_result = pipeline._detect_grid()
            
            if pipeline.current_state.get("intersections"):
                # Visualize intersections
                intersections_viz = visualize_intersections_custom(
                    original_image,
                    pipeline.current_state["intersections"],
                    save_path=os.path.join(output_dir, "03_intersections.jpg")
                )
                print(f"✅ Stage 3: Intersections saved ({len(pipeline.current_state['intersections'])} points)")
                stage_results["intersection_detection"] = {"success": True, "count": len(pipeline.current_state["intersections"])}
            else:
                print("❌ Stage 3: No intersections detected")
                stage_results["intersection_detection"] = {"success": False, "error": "No intersections found"}
            
            if pipeline.current_state.get("grid_points"):
                # Visualize grid
                grid_viz = visualize_grid_custom(
                    original_image,
                    pipeline.current_state["grid_points"],
                    save_path=os.path.join(output_dir, "04_grid_reconstruction.jpg")
                )
                print(f"✅ Stage 3: Grid reconstruction saved")
                stage_results["grid_reconstruction"] = {"success": True}
            else:
                print("❌ Stage 3: Grid reconstruction failed")
                stage_results["grid_reconstruction"] = {"success": False, "error": "No grid points generated"}
                
        except Exception as e:
            print(f"❌ Stage 3: Grid detection error: {str(e)}")
            stage_results["intersection_detection"] = {"success": False, "error": str(e)}
            stage_results["grid_reconstruction"] = {"success": False, "error": str(e)}
        
        # Stage 4: Cell Extraction
        print("\n🎯 Stage 4: Cell Extraction")
        try:
            if pipeline.current_state.get("grid_points"):
                extraction_result = pipeline._extract_cells()
                
                if pipeline.current_state.get("cell_images"):
                    # Visualize cells
                    cells_viz = visualize_cells_custom(
                        pipeline.current_state["cell_images"],
                        save_path=os.path.join(output_dir, "05_cell_extraction.jpg")
                    )
                    print(f"✅ Stage 4: Cell extraction saved")
                    stage_results["cell_extraction"] = {"success": True}
                else:
                    print("❌ Stage 4: Cell extraction failed")
                    stage_results["cell_extraction"] = {"success": False, "error": "No cells extracted"}
            else:
                print("⚠️  Stage 4: Skipping cell extraction (no grid points)")
                stage_results["cell_extraction"] = {"success": False, "error": "No grid points available"}
                
        except Exception as e:
            print(f"❌ Stage 4: Cell extraction error: {str(e)}")
            stage_results["cell_extraction"] = {"success": False, "error": str(e)}
        
        # Stage 5: Digit Recognition
        print("\n🎯 Stage 5: Digit Recognition")
        try:
            if pipeline.current_state.get("cell_images"):
                recognition_result = pipeline._recognize_digits()
                
                if pipeline.current_state.get("digit_grid"):
                    # Visualize digits
                    digits_viz = visualize_digits_custom(
                        pipeline.current_state["digit_grid"],
                        pipeline.current_state.get("confidence_grid"),
                        save_path=os.path.join(output_dir, "06_digit_recognition.jpg")
                    )
                    print(f"✅ Stage 5: Digit recognition saved")
                    stage_results["digit_recognition"] = {"success": True}
                    
                    # Print recognized grid
                    print("📋 Recognized digits:")
                    for row in pipeline.current_state["digit_grid"]:
                        print("  " + " ".join([str(d) if d != 0 else "." for d in row]))
                else:
                    print("❌ Stage 5: Digit recognition failed")
                    stage_results["digit_recognition"] = {"success": False, "error": "No digits recognized"}
            else:
                print("⚠️  Stage 5: Skipping digit recognition (no cells)")
                stage_results["digit_recognition"] = {"success": False, "error": "No cells available"}
                
        except Exception as e:
            print(f"❌ Stage 5: Digit recognition error: {str(e)}")
            stage_results["digit_recognition"] = {"success": False, "error": str(e)}
        
        # Stage 6: Puzzle Solving
        print("\n🎯 Stage 6: Puzzle Solving")
        try:
            if pipeline.current_state.get("digit_grid"):
                solving_result = pipeline._solve_puzzle()
                
                if pipeline.current_state.get("solution"):
                    # Visualize solution
                    solution_viz = visualize_digits_custom(
                        pipeline.current_state["solution"],
                        save_path=os.path.join(output_dir, "07_solution.jpg")
                    )
                    print(f"✅ Stage 6: Solution saved")
                    stage_results["puzzle_solving"] = {"success": True}
                    
                    # Print solution
                    print("🎯 Solution:")
                    for row in pipeline.current_state["solution"]:
                        print("  " + " ".join([str(d) for d in row]))
                else:
                    print("❌ Stage 6: Puzzle solving failed")
                    stage_results["puzzle_solving"] = {"success": False, "error": "No solution found"}
            else:
                print("⚠️  Stage 6: Skipping puzzle solving (no digit grid)")
                stage_results["puzzle_solving"] = {"success": False, "error": "No digit grid available"}
                
        except Exception as e:
            print(f"❌ Stage 6: Puzzle solving error: {str(e)}")
            stage_results["puzzle_solving"] = {"success": False, "error": str(e)}
        
        # Stage 7: Final Visualization
        print("\n🎯 Stage 7: Creating Pipeline Summary")
        try:
            # Create summary visualization
            summary_images = []
            summary_names = []
            
            # Collect successful stage images
            stage_files = [
                ("00_original_image.jpg", "Original"),
                ("02_board_detection.jpg", "Board"),
                ("03_intersections.jpg", "Intersections"),
                ("04_grid_reconstruction.jpg", "Grid"),
                ("05_cell_extraction.jpg", "Cells"),
                ("06_digit_recognition.jpg", "Digits"),
                ("07_solution.jpg", "Solution")
            ]
            
            target_height = 200
            
            for filename, name in stage_files:
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    img = cv2.imread(filepath)
                    if img is not None:
                        # Resize to target height while maintaining aspect ratio
                        height, width = img.shape[:2]
                        if height > 0:
                            target_width = int(width * target_height / height)
                            resized = cv2.resize(img, (target_width, target_height))
                            
                            # Add label
                            cv2.putText(resized, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            cv2.putText(resized, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                            
                            summary_images.append(resized)
            
            # Combine horizontally
            if len(summary_images) > 0:
                summary_image = np.hstack(summary_images)
                summary_path = os.path.join(output_dir, "08_pipeline_summary.jpg")
                cv2.imwrite(summary_path, summary_image)
                print(f"✅ Stage 7: Pipeline summary saved to {summary_path}")
            
        except Exception as e:
            print(f"❌ Stage 7: Summary creation error: {str(e)}")
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🏁 PIPELINE COMPLETION SUMMARY")
    print("=" * 60)
    
    for stage, result in stage_results.items():
        success = result.get("success", False) if isinstance(result, dict) else True
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{stage.upper().replace('_', ' '):20} {status}")
        if not success and "error" in result:
            print(f"{'':20} Error: {result['error']}")
    
    print(f"\n📁 All outputs saved to: {output_dir}")
    print("🔍 Check the numbered files to see each pipeline stage!")
    
    return stage_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Visual Pipeline Test - Save images at every stage')
    
    parser.add_argument(
        'input',
        type=str,
        help='Input image path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='visual_pipeline_output',
        help='Output directory for stage visualizations'
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
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    args = parser.parse_args()
    
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
    
    # Run visual pipeline test
    try:
        start_time = time.time()
        
        results = run_visual_pipeline_test(args.input, args.output, pipeline)
        
        processing_time = time.time() - start_time
        print(f"\n⏱️  Total processing time: {processing_time:.2f} seconds")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
