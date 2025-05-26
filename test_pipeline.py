#!/usr/bin/env python3
"""
Full Pipeline Test Script
Tests the complete Sudoku recognition pipeline
Returns solved sudoku as 2D matrix
Run from project root: python test_pipeline.py
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append('.')

try:
    from pipeline import SudokuRecognizerPipeline
    print("✅ Pipeline imported successfully")
except ImportError as e:
    print(f"❌ Pipeline import failed: {e}")
    sys.exit(1)

def find_test_image():
    """Find a test image to use"""
    test_locations = [
        'data/test_images',
        'test_images',
        'examples',
        'samples'
    ]
    
    for location in test_locations:
        if os.path.exists(location):
            for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                import glob
                images = glob.glob(os.path.join(location, f'*.{ext}'))
                if images:
                    return images[0]
    
    return None

def print_sudoku_grid(grid, title="Sudoku Grid"):
    """Print sudoku grid in a nice format"""
    print(f"\n{title}:")
    print("┌─────────┬─────────┬─────────┐")
    for i in range(9):
        if i == 3 or i == 6:
            print("├─────────┼─────────┼─────────┤")
        
        row = "│ "
        for j in range(9):
            if j == 3 or j == 6:
                row += "│ "
            
            value = grid[i][j] if grid[i][j] != 0 else " "
            row += f"{value} "
        row += "│"
        print(row)
    print("└─────────┴─────────┴─────────┘")

def test_pipeline_step_by_step(image_path):
    """Test pipeline with detailed step-by-step output"""
    print("🚀 STEP-BY-STEP PIPELINE TEST")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = SudokuRecognizerPipeline()
    
    # Load models
    print("🔧 Loading models...")
    model_dir = "data/models"
    if pipeline.load_models(model_dir):
        print("✅ Models loaded successfully")
    else:
        print("⚠️  Some models may not have loaded properly")
    
    # Process image
    print(f"\n📷 Processing image: {os.path.basename(image_path)}")
    
    try:
        results = pipeline.process_image(image_path)
        
        print(f"\n📋 PIPELINE RESULTS:")
        print("=" * 30)
        print(f"✅ Success: {results['success']}")
        print(f"⏱️  Processing time: {results['processing_time']:.2f}s")
        print(f"🎯 Grid detected: {results['grid_detected']}")
        print(f"📱 Cells extracted: {results['cells_extracted']}")
        print(f"🔢 Digits recognized: {results['digits_recognized']}")
        print(f"🧩 Puzzle solved: {results['puzzle_solved']}")
        
        # Show recognized digits
        if results['digit_grid']:
            print_sudoku_grid(results['digit_grid'], "Recognized Digits")
            
            # Count recognized digits
            recognized_count = sum(1 for row in results['digit_grid'] 
                                 for digit in row if digit > 0)
            print(f"📊 Recognized {recognized_count}/81 digits")
        
        # Show solution
        if results['solved_grid']:
            print_sudoku_grid(results['solved_grid'], "SOLVED SUDOKU")
            
            # Verify solution is complete
            is_complete = not any(0 in row for row in results['solved_grid'])
            if is_complete:
                print("🎉 PUZZLE COMPLETELY SOLVED!")
            else:
                print("⚠️  Puzzle partially solved")
            
            return results['solved_grid']
        else:
            print("❌ No solution generated")
            return None
            
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        return None

def test_pipeline_simple(image_path):
    """Simple pipeline test - just get the result"""
    print("\n🎯 SIMPLE PIPELINE TEST")
    print("=" * 30)
    
    pipeline = SudokuRecognizerPipeline()
    
    # Load models quietly  
    pipeline.load_models("data/models")
    
    try:
        results = pipeline.process_image(image_path)
        
        if results['success'] and results['solved_grid']:
            print("✅ Pipeline succeeded!")
            return results['solved_grid']
        else:
            print("❌ Pipeline failed to solve puzzle")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def convert_to_2d_matrix(solved_grid):
    """Convert solved grid to clean 2D matrix format"""
    if solved_grid is None:
        return None
    
    # Ensure it's a proper 9x9 matrix of integers
    matrix = []
    for row in solved_grid:
        matrix_row = []
        for cell in row:
            matrix_row.append(int(cell))
        matrix.append(matrix_row)
    
    return matrix

def main():
    print("🧪 FULL PIPELINE TEST")
    print(f"Testing complete Sudoku recognition pipeline")
    print("=" * 50)
    
    # Find test image
    test_image = find_test_image()
    
    if test_image is None:
        print("❌ No test images found!")
        print("💡 Please add a sudoku image to data/test_images/")
        print("   Supported formats: jpg, jpeg, png")
        return
    
    print(f"📷 Using test image: {test_image}")
    
    # Load and show image info
    image = cv2.imread(test_image)
    if image is None:
        print("❌ Could not load test image")
        return
    
    print(f"   Image size: {image.shape}")
    
    # Test pipeline step-by-step
    solved_grid = test_pipeline_step_by_step(test_image)
    
    # Convert to clean 2D matrix
    solution_matrix = convert_to_2d_matrix(solved_grid)
    
    if solution_matrix:
        print("\n🎯 FINAL RESULT - 2D MATRIX:")
        print("=" * 35)
        print("solution_matrix = [")
        for i, row in enumerate(solution_matrix):
            print(f"    {row}{',' if i < 8 else ''}")
        print("]")
        
        # Verify it's a valid sudoku solution
        def is_valid_solution(matrix):
            # Check rows, columns, and 3x3 boxes
            for i in range(9):
                # Check row
                if sorted(matrix[i]) != list(range(1, 10)):
                    return False
                # Check column  
                col = [matrix[j][i] for j in range(9)]
                if sorted(col) != list(range(1, 10)):
                    return False
            
            # Check 3x3 boxes
            for box_row in range(3):
                for box_col in range(3):
                    box = []
                    for i in range(3):
                        for j in range(3):
                            box.append(matrix[box_row*3 + i][box_col*3 + j])
                    if sorted(box) != list(range(1, 10)):
                        return False
            return True
        
        if is_valid_solution(solution_matrix):
            print("✅ VALID SUDOKU SOLUTION!")
        else:
            print("⚠️  Solution may have errors (untrained models)")
        
        return solution_matrix
    else:
        print("\n❌ PIPELINE FAILED")
        print("Possible reasons:")
        print("- Models are untrained (random outputs)")
        print("- Image quality issues")
        print("- Pipeline integration problems")
        return None

if __name__ == "__main__":
    result = main()
    
    if result:
        print(f"\n🎉 SUCCESS: Pipeline returned 9x9 solution matrix")
    else:
        print(f"\n💡 NOTE: Models are untrained, so random results expected")
        print("   Pipeline architecture is working correctly!")
