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
                # Try board detection
                board_result = pipeline.board_detector.detect(original_image)
                if board_result.get("success", False):
                    # Visualize board detection
                    board_viz = original_image.copy()
                    bbox = board_result.get("bounding_box")
                    if bbox:
                        x, y, w, h = bbox
                        cv2.rectangle(board_viz, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(board_viz, f"Board Detected (conf: {board_result.get('confidence', 0):.2f})", 
                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    board_path = os.path.join(output_dir, "02_board_detection.jpg")
                    cv2.imwrite(board_path, board_viz)
                    print(f"✅ Stage 2: Board detection saved to {board_path}")
                else:
                    print("⚠️  Stage 2: Board detection failed, will use intersection method")
            else:
                print("ℹ️  Stage 2: Board detection disabled, skipping")
        except Exception as e:
            print(f"❌ Stage 2: Board detection error: {str(e)}")
        
        # Stage 3: Grid Detection (Intersection Detection)
        print("\n🎯 Stage 3: Grid Detection (Intersection Detection)")
        try:
            grid_result = pipeline._detect_grid()
            
            if pipeline.current_state["intersections"]:
                # Visualize intersections
                intersections_viz = visualize_intersections(
                    original_image,
                    pipeline.current_state["intersections"],
                    save_path=os.path.join(output_dir, "03_intersections.jpg")
                )
                print(f"✅ Stage 3: Intersections saved ({len(pipeline.current_state['intersections'])} points)")
            
            if pipeline.current_state["grid_points"]:
                # Visualize grid
                grid_viz = visualize_grid(
                    original_image,
                    pipeline.current_state["grid_points"],
                    save_path=os.path.join(output_dir, "04_grid_reconstruction.jpg")
                )
                print(f"✅ Stage 3: Grid reconstruction saved")
            
            stage_results["grid_detection"] = grid_result
            
        except Exception as e:
            print(f"❌ Stage 3: Grid detection error: {str(e)}")
            stage_results["grid_detection"] = {"success": False, "error": str(e)}
        
        # Stage 4: Cell Extraction
        print("\n🎯 Stage 4: Cell Extraction")
        try:
            if pipeline.current_state["grid_points"]:
                cell_result = pipeline._extract_cells()
                
                if pipeline.current_state["cell_images"]:
                    # Visualize cells
                    cells_viz = visualize_cells(
                        pipeline.current_state["cell_images"],
                        save_path=os.path.join(output_dir, "05_extracted_cells.jpg")
                    )
                    print(f"✅ Stage 4: Cell extraction saved ({len(pipeline.current_state['cell_images'])}x{len(pipeline.current_state['cell_images'][0])} cells)")
                    
                    # Save individual cells for detailed inspection
                    cells_dir = os.path.join(output_dir, "05_individual_cells")
                    os.makedirs(cells_dir, exist_ok=True)
                    
                    for i in range(9):
                        for j in range(9):
                            if i < len(pipeline.current_state["cell_images"]) and j < len(pipeline.current_state["cell_images"][i]):
                                cell_path = os.path.join(cells_dir, f"cell_{i}_{j}.jpg")
                                cv2.imwrite(cell_path, pipeline.current_state["cell_images"][i][j])
                    
                    print(f"✅ Stage 4: Individual cells saved to {cells_dir}")
                
                stage_results["cell_extraction"] = cell_result
            else:
                print("❌ Stage 4: No grid points available for cell extraction")
                
        except Exception as e:
            print(f"❌ Stage 4: Cell extraction error: {str(e)}")
            stage_results["cell_extraction"] = {"success": False, "error": str(e)}
        
        # Stage 5: Digit Recognition
        print("\n🎯 Stage 5: Digit Recognition")
        try:
            if pipeline.current_state["cell_images"]:
                digit_result = pipeline._recognize_digits()
                
                if pipeline.current_state["digit_grid"]:
                    # Visualize recognized digits
                    digits_viz = visualize_digit_grid(
                        pipeline.current_state["digit_grid"],
                        pipeline.current_state.get("confidence_grid"),
                        save_path=os.path.join(output_dir, "06_recognized_digits.jpg")
                    )
                    print(f"✅ Stage 5: Digit recognition saved")
                    
                    # Print recognized grid
                    print("📋 Recognized Sudoku Grid:")
                    for row in pipeline.current_state["digit_grid"]:
                        print("  " + " ".join(str(d) if d > 0 else "." for d in row))
                
                stage_results["digit_recognition"] = digit_result
            else:
                print("❌ Stage 5: No cell images available for digit recognition")
                
        except Exception as e:
            print(f"❌ Stage 5: Digit recognition error: {str(e)}")
            stage_results["digit_recognition"] = {"success": False, "error": str(e)}
        
        # Stage 6: Puzzle Solving
        print("\n🎯 Stage 6: Puzzle Solving")
        try:
            if pipeline.current_state["digit_grid"]:
                solve_result = pipeline._solve_puzzle()
                
                if pipeline.current_state["solved_grid"]:
                    # Visualize solution
                    solution_viz = visualize_solution(
                        pipeline.current_state["digit_grid"],
                        pipeline.current_state["solved_grid"],
                        save_path=os.path.join(output_dir, "07_solution.jpg")
                    )
                    print(f"✅ Stage 6: Solution saved")
                    
                    # Create overlay on original image
                    if pipeline.current_state["grid_points"]:
                        overlay_viz = overlay_solution_on_image(
                            original_image,
                            pipeline.current_state["digit_grid"],
                            pipeline.current_state["solved_grid"],
                            pipeline.current_state["grid_points"],
                            save_path=os.path.join(output_dir, "08_solution_overlay.jpg")
                        )
                        print(f"✅ Stage 6: Solution overlay saved")
                    
                    # Print solution
                    print("🎉 Solved Sudoku Grid:")
                    for row in pipeline.current_state["solved_grid"]:
                        print("  " + " ".join(str(d) for d in row))
                
                stage_results["solving"] = solve_result
            else:
                print("❌ Stage 6: No digit grid available for solving")
                
        except Exception as e:
            print(f"❌ Stage 6: Puzzle solving error: {str(e)}")
            stage_results["solving"] = {"success": False, "error": str(e)}
        
        # Stage 7: Summary Report
        print("\n🎯 Stage 7: Creating Summary Report")
        try:
            # Create a comprehensive summary image
            summary_parts = []
            
            # Collect available images
            summary_images = [
                ("Original", original_image),
            ]
            
            if pipeline.current_state["intersections"]:
                intersections_img = visualize_intersections(original_image, pipeline.current_state["intersections"])
                summary_images.append(("Intersections", intersections_img))
            
            if pipeline.current_state["grid_points"]:
                grid_img = visualize_grid(original_image, pipeline.current_state["grid_points"])
                summary_images.append(("Grid", grid_img))
            
            if pipeline.current_state["cell_images"]:
                cells_img = visualize_cells(pipeline.current_state["cell_images"])
                summary_images.append(("Cells", cells_img))
            
            if pipeline.current_state["digit_grid"]:
                digits_img = visualize_digit_grid(
                    pipeline.current_state["digit_grid"],
                    pipeline.current_state.get("confidence_grid")
                )
                summary_images.append(("Digits", digits_img))
            
            if pipeline.current_state["solved_grid"]:
                solution_img = visualize_solution(
                    pipeline.current_state["digit_grid"],
                    pipeline.current_state["solved_grid"]
                )
                summary_images.append(("Solution", solution_img))
            
            # Create summary grid
            if len(summary_images) > 0:
                # Resize all images to same height
                target_height = 300
                resized_images = []
                
                for name, img in summary_images:
                    if len(img.shape) == 3:
                        h, w = img.shape[:2]
                    else:
                        h, w = img.shape
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                    aspect_ratio = w / h
                    new_width = int(target_height * aspect_ratio)
                    resized = cv2.resize(img, (new_width, target_height))
                    
                    # Add label
                    cv2.putText(resized, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(resized, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                    
                    resized_images.append(resized)
                
                # Combine horizontally
                if len(resized_images) > 0:
                    summary_image = np.hstack(resized_images)
                    summary_path = os.path.join(output_dir, "09_pipeline_summary.jpg")
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
        print(f"{stage.upper():20} {status}")
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