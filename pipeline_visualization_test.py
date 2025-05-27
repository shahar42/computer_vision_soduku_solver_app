#!/usr/bin/env python3
"""
Sudoku Pipeline Visualization Test

This script runs the complete Sudoku recognition pipeline and
visualizes each step of the process with detailed graphics.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

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

def create_output_dir():
    """Create output directory for visualizations"""
    output_dir = 'pipeline_visualization'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def visualize_intersections(image, intersections, output_dir):
    """Visualize detected intersections on the image"""
    print("📊 Visualizing intersections...")
    
    # Convert BGR to RGB for matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(rgb_image)
    
    # Plot intersections
    if intersections:
        x_points = [p[0] for p in intersections]
        y_points = [p[1] for p in intersections]
        plt.scatter(x_points, y_points, c='red', marker='x', s=50)
    
    plt.title(f'Detected Intersections: {len(intersections)} points')
    plt.axis('off')
    
    # Save figure
    plt.savefig(f'{output_dir}/1_intersections.png', bbox_inches='tight')
    plt.close()

def visualize_grid(image, grid_points, output_dir):
    """Visualize reconstructed grid on the image"""
    print("📊 Visualizing grid reconstruction...")
    
    # Convert BGR to RGB for matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(rgb_image)
    
    if grid_points:
        # Draw horizontal lines
        for i in range(10):  # 10 horizontal lines for a 9x9 grid
            x_points = [p[0] for p in grid_points[i]]
            y_points = [p[1] for p in grid_points[i]]
            plt.plot(x_points, y_points, 'g-', linewidth=2)
        
        # Draw vertical lines
        for j in range(10):  # 10 vertical lines for a 9x9 grid
            x_points = [grid_points[i][j][0] for i in range(10)]
            y_points = [grid_points[i][j][1] for i in range(10)]
            plt.plot(x_points, y_points, 'g-', linewidth=2)
        
        # Highlight grid intersections
        for i in range(10):
            for j in range(10):
                plt.plot(grid_points[i][j][0], grid_points[i][j][1], 'b.', markersize=8)
    
    plt.title('Reconstructed Grid')
    plt.axis('off')
    
    # Save figure
    plt.savefig(f'{output_dir}/2_grid.png', bbox_inches='tight')
    plt.close()

def visualize_cells(cell_images, output_dir):
    """Visualize extracted cells"""
    print("📊 Visualizing extracted cells...")
    
    plt.figure(figsize=(15, 15))
    
    # Create a 9x9 grid of subplots
    for i in range(9):
        for j in range(9):
            plt.subplot(9, 9, i*9 + j + 1)
            plt.imshow(cell_images[i][j], cmap='gray')
            plt.axis('off')
    
    plt.suptitle('Extracted Cells', fontsize=20)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'{output_dir}/3_cells.png', bbox_inches='tight')
    plt.close()
    
    # Create a composite image of all cells
    cell_size = cell_images[0][0].shape[0]
    grid_image = np.ones((9 * cell_size, 9 * cell_size), dtype=np.uint8) * 255
    
    for i in range(9):
        for j in range(9):
            cell = cell_images[i][j].copy()
            
            # Resize if necessary
            if cell.shape[0] != cell_size or cell.shape[1] != cell_size:
                cell = cv2.resize(cell, (cell_size, cell_size))
            
            grid_image[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = cell
    
    cv2.imwrite(f'{output_dir}/3_cells_grid.png', grid_image)

def visualize_digits(digit_grid, confidence_grid, output_dir):
    """Visualize recognized digits with confidence scores"""
    print("📊 Visualizing recognized digits...")
    
    plt.figure(figsize=(12, 12))
    
    # Create a visual grid
    for i in range(9):
        for j in range(9):
            digit = digit_grid[i][j]
            confidence = confidence_grid[i][j]
            
            color = 'green' if confidence > 0.8 else 'orange' if confidence > 0.5 else 'red'
            
            # Create cell
            rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', linewidth=1)
            plt.gca().add_patch(rect)
            
            # Add digit and confidence
            if digit > 0:
                plt.text(j + 0.5, i + 0.5, str(digit), 
                         ha='center', va='center', fontsize=20, color=color)
                plt.text(j + 0.5, i + 0.7, f"{confidence:.2f}", 
                         ha='center', va='center', fontsize=8, color=color)
    
    # Add thicker lines for 3x3 boxes
    for i in range(0, 10, 3):
        plt.axhline(y=i, color='black', linewidth=2)
        plt.axvline(x=i, color='black', linewidth=2)
    
    plt.xlim(0, 9)
    plt.ylim(9, 0)  # Invert y-axis
    plt.title('Recognized Digits with Confidence Scores')
    plt.axis('off')
    
    # Save figure
    plt.savefig(f'{output_dir}/4_digits.png', bbox_inches='tight')
    plt.close()

def visualize_solution(digit_grid, solved_grid, output_dir):
    """Visualize the original and solved Sudoku grids side by side"""
    print("📊 Visualizing solution...")
    
    plt.figure(figsize=(20, 10))
    
    # Original grid
    plt.subplot(1, 2, 1)
    for i in range(9):
        for j in range(9):
            # Create cell
            rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', linewidth=1)
            plt.gca().add_patch(rect)
            
            # Add digit
            if digit_grid[i][j] > 0:
                plt.text(j + 0.5, i + 0.5, str(digit_grid[i][j]), 
                         ha='center', va='center', fontsize=20, color='blue')
    
    # Add thicker lines for 3x3 boxes
    for i in range(0, 10, 3):
        plt.axhline(y=i, color='black', linewidth=2)
        plt.axvline(x=i, color='black', linewidth=2)
    
    plt.xlim(0, 9)
    plt.ylim(9, 0)  # Invert y-axis
    plt.title('Recognized Digits')
    plt.axis('off')
    
    # Solved grid
    plt.subplot(1, 2, 2)
    for i in range(9):
        for j in range(9):
            # Create cell
            rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', linewidth=1)
            plt.gca().add_patch(rect)
            
            # Add digit
            if solved_grid[i][j] > 0:
                # Highlight original digits
                color = 'blue' if digit_grid[i][j] > 0 else 'green'
                plt.text(j + 0.5, i + 0.5, str(solved_grid[i][j]), 
                         ha='center', va='center', fontsize=20, color=color)
    
    # Add thicker lines for 3x3 boxes
    for i in range(0, 10, 3):
        plt.axhline(y=i, color='black', linewidth=2)
        plt.axvline(x=i, color='black', linewidth=2)
    
    plt.xlim(0, 9)
    plt.ylim(9, 0)  # Invert y-axis
    plt.title('Solved Puzzle')
    plt.axis('off')
    
    # Save figure
    plt.savefig(f'{output_dir}/5_solution.png', bbox_inches='tight')
    plt.close()

def create_pipeline_summary(image_path, results, output_dir):
    """Create a summary visualization of the entire pipeline"""
    print("📊 Creating pipeline summary...")
    
    # Create a figure with multiple images
    plt.figure(figsize=(20, 20))
    gs = GridSpec(3, 2, figure=plt.gcf())
    
    # 1. Original image
    ax1 = plt.subplot(gs[0, 0])
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(rgb_image)
    ax1.set_title('1. Original Image')
    ax1.axis('off')
    
    # 2. Grid detection
    ax2 = plt.subplot(gs[0, 1])
    grid_img = cv2.imread(f'{output_dir}/2_grid.png')
    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
    ax2.imshow(grid_img)
    ax2.set_title('2. Grid Detection')
    ax2.axis('off')
    
    # 3. Cell extraction
    ax3 = plt.subplot(gs[1, 0])
    cells_img = cv2.imread(f'{output_dir}/3_cells_grid.png')
    cells_img = cv2.cvtColor(cells_img, cv2.COLOR_BGR2RGB) if len(cells_img.shape) == 3 else cells_img
    ax3.imshow(cells_img, cmap='gray' if len(cells_img.shape) == 2 else None)
    ax3.set_title('3. Cell Extraction')
    ax3.axis('off')
    
    # 4. Digit recognition
    ax4 = plt.subplot(gs[1, 1])
    digits_img = cv2.imread(f'{output_dir}/4_digits.png')
    digits_img = cv2.cvtColor(digits_img, cv2.COLOR_BGR2RGB)
    ax4.imshow(digits_img)
    ax4.set_title('4. Digit Recognition')
    ax4.axis('off')
    
    # 5. Solved puzzle
    ax5 = plt.subplot(gs[2, :])
    solution_img = cv2.imread(f'{output_dir}/5_solution.png')
    solution_img = cv2.cvtColor(solution_img, cv2.COLOR_BGR2RGB)
    ax5.imshow(solution_img)
    ax5.set_title('5. Puzzle Solution')
    ax5.axis('off')
    
    # Add statistics
    plt.figtext(0.5, 0.01, 
                f"Processing time: {results['processing_time']:.2f}s | "
                f"Recognized digits: {sum(1 for row in results['digit_grid'] for digit in row if digit > 0)}/81 | "
                f"Success: {'Yes' if results['success'] else 'No'}", 
                ha='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pipeline_summary.png', bbox_inches='tight')
    plt.close()

def run_visualization_test(image_path):
    """Run the pipeline and visualize each step"""
    print("🚀 PIPELINE VISUALIZATION TEST")
    print("=" * 50)
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"📂 Saving visualizations to: {output_dir}")
    
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
        
        # Visualize each step
        visualize_intersections(pipeline.current_state["image"], 
                               pipeline.current_state["intersections"], 
                               output_dir)
        
        visualize_grid(pipeline.current_state["image"], 
                      pipeline.current_state["grid_points"], 
                      output_dir)
        
        visualize_cells(pipeline.current_state["cell_images"], 
                       output_dir)
        
        visualize_digits(pipeline.current_state["digit_grid"], 
                        pipeline.current_state["confidence_grid"], 
                        output_dir)
        
        if results['solved_grid']:
            visualize_solution(pipeline.current_state["digit_grid"], 
                              pipeline.current_state["solved_grid"], 
                              output_dir)
        
        # Create pipeline summary
        create_pipeline_summary(image_path, results, output_dir)
        
        print(f"\n✅ Visualizations saved to {output_dir}/")
        print("📊 Key visualizations:")
        print(f"  1. Intersections: {output_dir}/1_intersections.png")
        print(f"  2. Grid: {output_dir}/2_grid.png")
        print(f"  3. Cells: {output_dir}/3_cells.png")
        print(f"  4. Digits: {output_dir}/4_digits.png")
        print(f"  5. Solution: {output_dir}/5_solution.png")
        print(f"  6. Summary: {output_dir}/pipeline_summary.png")
        
        return results
            
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🧪 SUDOKU PIPELINE VISUALIZATION TEST")
    print(f"Visualizing each step of the Sudoku recognition pipeline")
    print("=" * 50)
    
    # Find test image
    test_image = find_test_image()
    
    if test_image is None:
        print("❌ No test images found!")
        print("💡 Please add a sudoku image to data/test_images/")
        print("   Supported formats: jpg, jpeg, png")
        return
    
    print(f"📷 Using test image: {test_image}")
    
    # Run visualization test
    results = run_visualization_test(test_image)
    
    if results and results['success']:
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY")
    else:
        print("\n⚠️  PIPELINE ENCOUNTERED ISSUES")
        
    print("\n💡 Check the output directory for visualizations")

if __name__ == "__main__":
    main()
