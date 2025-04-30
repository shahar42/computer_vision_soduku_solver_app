"""
Visualization Utilities.

This module provides functions for visualizing Sudoku grids, cells, and results.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from config import get_settings

# Define types
ImageType = np.ndarray
PointType = Tuple[int, int]
GridPointsType = List[List[PointType]]
GridType = List[List[int]]


def visualize_intersections(
    image: ImageType,
    intersections: List[PointType],
    save_path: Optional[str] = None,
    show: bool = False
) -> ImageType:
    """
    Visualize detected intersections on an image.
    
    Args:
        image: Input image
        intersections: List of intersection points
        save_path: Path to save visualization (optional)
        show: Whether to show visualization
        
    Returns:
        Visualization image
    """
    # Create a copy of the image
    viz = image.copy()
    
    # Draw intersection points
    for x, y in intersections:
        cv2.circle(viz, (x, y), 5, (0, 0, 255), -1)
        
    # Save visualization if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, viz)
        
    # Show visualization if requested
    if show:
        plt.figure(figsize=(10, 8))
        if len(viz.shape) == 3:
            plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(viz, cmap='gray')
        plt.title(f"Detected Intersections ({len(intersections)} points)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    return viz


def visualize_grid(
    image: ImageType,
    grid_points: GridPointsType,
    save_path: Optional[str] = None,
    show: bool = False
) -> ImageType:
    """
    Visualize detected grid on an image.
    
    Args:
        image: Input image
        grid_points: 2D list of grid points
        save_path: Path to save visualization (optional)
        show: Whether to show visualization
        
    Returns:
        Visualization image
    """
    # Create a copy of the image
    viz = image.copy()
    
    # Draw grid lines
    for i in range(len(grid_points)):
        # Draw horizontal line
        points = np.array(grid_points[i], dtype=np.int32)
        cv2.polylines(viz, [points], False, (0, 255, 0), 2)
        
        # Draw vertical line (if grid is square)
        if len(grid_points) == len(grid_points[i]):
            points = np.array([grid_points[j][i] for j in range(len(grid_points))], dtype=np.int32)
            cv2.polylines(viz, [points], False, (0, 255, 0), 2)
            
    # Draw grid points
    for row in grid_points:
        for x, y in row:
            cv2.circle(viz, (x, y), 3, (255, 0, 0), -1)
            
    # Save visualization if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, viz)
        
    # Show visualization if requested
    if show:
        plt.figure(figsize=(10, 8))
        if len(viz.shape) == 3:
            plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(viz, cmap='gray')
        plt.title("Detected Grid")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    return viz


def visualize_cells(
    cell_images: List[List[ImageType]],
    save_path: Optional[str] = None,
    show: bool = False,
    grid_size: int = 9
) -> ImageType:
    """
    Visualize extracted cell images.
    
    Args:
        cell_images: 2D list of cell images
        save_path: Path to save visualization (optional)
        show: Whether to show visualization
        grid_size: Size of the grid
        
    Returns:
        Visualization image
    """
    # Determine cell size (assume all cells have the same size)
    cell_size = cell_images[0][0].shape[0]
    
    # Create a grid image
    grid_image = np.ones((grid_size * cell_size, grid_size * cell_size), dtype=np.uint8) * 255
    
    # Place cell images in grid
    for i in range(grid_size):
        for j in range(grid_size):
            cell = cell_images[i][j]
            
            # Convert to grayscale if color
            if len(cell.shape) == 3:
                cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                
            # Resize if necessary
            if cell.shape[0] != cell_size or cell.shape[1] != cell_size:
                cell = cv2.resize(cell, (cell_size, cell_size))
                
            # Place in grid
            grid_image[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size] = cell
            
    # Draw grid lines
    for i in range(1, grid_size):
        # Horizontal lines
        cv2.line(grid_image, (0, i * cell_size), (grid_size * cell_size, i * cell_size), 0, 1)
        
        # Vertical lines
        cv2.line(grid_image, (i * cell_size, 0), (i * cell_size, grid_size * cell_size), 0, 1)
        
    # Draw thicker lines for 3x3 boxes
    for i in range(1, 3):
        # Horizontal lines
        cv2.line(grid_image, (0, i * 3 * cell_size), (grid_size * cell_size, i * 3 * cell_size), 0, 2)
        
        # Vertical lines
        cv2.line(grid_image, (i * 3 * cell_size, 0), (i * 3 * cell_size, grid_size * cell_size), 0, 2)
        
    # Save visualization if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, grid_image)
        
    # Show visualization if requested
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_image, cmap='gray')
        plt.title("Extracted Cells")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    return grid_image


def visualize_digit_grid(
    digit_grid: GridType,
    confidence_grid: Optional[List[List[float]]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    cell_size: int = 50,
    grid_size: int = 9
) -> ImageType:
    """
    Visualize recognized digit grid.
    
    Args:
        digit_grid: Grid of recognized digits
        confidence_grid: Grid of confidence scores (optional)
        save_path: Path to save visualization (optional)
        show: Whether to show visualization
        cell_size: Size of each cell in pixels
        grid_size: Size of the grid
        
    Returns:
        Visualization image
    """
    # Create a grid image
    grid_image = np.ones((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8) * 255
    
    # Draw digits
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = cell_size / 50.0
    font_thickness = max(1, int(cell_size / 25))
    
    for i in range(grid_size):
        for j in range(grid_size):
            digit = digit_grid[i][j]
            
            if digit > 0:
                # Calculate text size and position
                text = str(digit)
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = j * cell_size + (cell_size - text_size[0]) // 2
                text_y = i * cell_size + (cell_size + text_size[1]) // 2
                
                # Set color based on confidence
                color = (0, 0, 0)  # Black by default
                
                if confidence_grid is not None:
                    confidence = confidence_grid[i][j]
                    
                    if confidence < 0.5:
                        color = (0, 0, 255)  # Red for low confidence
                    elif confidence < 0.8:
                        color = (0, 128, 255)  # Orange for medium confidence
                    else:
                        color = (0, 0, 0)  # Black for high confidence
                        
                # Draw digit
                cv2.putText(grid_image, text, (text_x, text_y), font, font_scale, color, font_thickness)
                
    # Draw grid lines
    for i in range(1, grid_size):
        # Horizontal lines
        cv2.line(grid_image, (0, i * cell_size), (grid_size * cell_size, i * cell_size), (0, 0, 0), 1)
        
        # Vertical lines
        cv2.line(grid_image, (i * cell_size, 0), (i * cell_size, grid_size * cell_size), (0, 0, 0), 1)
        
    # Draw thicker lines for 3x3 boxes
    for i in range(1, 3):
        # Horizontal lines
        cv2.line(grid_image, (0, i * 3 * cell_size), (grid_size * cell_size, i * 3 * cell_size), (0, 0, 0), 2)
        
        # Vertical lines
        cv2.line(grid_image, (i * 3 * cell_size, 0), (i * 3 * cell_size, grid_size * cell_size), (0, 0, 0), 2)
        
    # Draw outer border
    cv2.rectangle(grid_image, (0, 0), (grid_size * cell_size, grid_size * cell_size), (0, 0, 0), 2)
    
    # Save visualization if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, grid_image)
        
    # Show visualization if requested
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
        plt.title("Recognized Digits")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    return grid_image


def visualize_solution(
    initial_grid: GridType,
    solved_grid: GridType,
    original_image: Optional[ImageType] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    cell_size: int = 50,
    grid_size: int = 9
) -> ImageType:
    """
    Visualize Sudoku solution.
    
    Args:
        initial_grid: Initial digit grid
        solved_grid: Solved digit grid
        original_image: Original image (optional)
        save_path: Path to save visualization (optional)
        show: Whether to show visualization
        cell_size: Size of each cell in pixels
        grid_size: Size of the grid
        
    Returns:
        Visualization image
    """
    # Create a grid image
    grid_image = np.ones((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8) * 255
    
    # Draw digits
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = cell_size / 50.0
    font_thickness = max(1, int(cell_size / 25))
    
    for i in range(grid_size):
        for j in range(grid_size):
            initial_digit = initial_grid[i][j]
            solved_digit = solved_grid[i][j]
            
            if solved_digit > 0:
                # Calculate text size and position
                text = str(solved_digit)
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = j * cell_size + (cell_size - text_size[0]) // 2
                text_y = i * cell_size + (cell_size + text_size[1]) // 2
                
                # Set color based on whether digit was initial or solved
                color = (0, 0, 0)  # Black for initial digits
                
                if initial_digit == 0:
                    color = (0, 128, 0)  # Green for solved digits
                    
                # Draw digit
                cv2.putText(grid_image, text, (text_x, text_y), font, font_scale, color, font_thickness)
                
    # Draw grid lines
    for i in range(1, grid_size):
        # Horizontal lines
        cv2.line(grid_image, (0, i * cell_size), (grid_size * cell_size, i * cell_size), (0, 0, 0), 1)
        
        # Vertical lines
        cv2.line(grid_image, (i * cell_size, 0), (i * cell_size, grid_size * cell_size), (0, 0, 0), 1)
        
    # Draw thicker lines for 3x3 boxes
    for i in range(1, 3):
        # Horizontal lines
        cv2.line(grid_image, (0, i * 3 * cell_size), (grid_size * cell_size, i * 3 * cell_size), (0, 0, 0), 2)
        
        # Vertical lines
        cv2.line(grid_image, (i * 3 * cell_size, 0), (i * 3 * cell_size, grid_size * cell_size), (0, 0, 0), 2)
        
    # Draw outer border
    cv2.rectangle(grid_image, (0, 0), (grid_size * cell_size, grid_size * cell_size), (0, 0, 0), 2)
    
    # If original image is provided, create a side-by-side visualization
    if original_image is not None:
        # Resize original image to match solution grid
        original_resized = cv2.resize(
            original_image,
            (grid_size * cell_size, grid_size * cell_size)
        )
        
        # Create a combined image
        combined_image = np.zeros((grid_size * cell_size, 2 * grid_size * cell_size, 3), dtype=np.uint8)
        
        # Place original image on the left
        if len(original_resized.shape) == 2:
            original_colored = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)
            combined_image[:, :grid_size * cell_size] = original_colored
        else:
            combined_image[:, :grid_size * cell_size] = original_resized
            
        # Place solution on the right
        combined_image[:, grid_size * cell_size:] = grid_image
        
        grid_image = combined_image
        
    # Save visualization if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, grid_image)
        
    # Show visualization if requested
    if show:
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
        if original_image is not None:
            plt.title("Original Image and Solution")
        else:
            plt.title("Sudoku Solution")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    return grid_image


def visualize_processing_stages(
    image: ImageType,
    intersections: List[PointType],
    grid_points: GridPointsType,
    cell_images: List[List[ImageType]],
    digit_grid: GridType,
    confidence_grid: List[List[float]],
    solved_grid: GridType,
    save_dir: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Visualize all processing stages.
    
    Args:
        image: Original image
        intersections: Detected intersection points
        grid_points: Reconstructed grid points
        cell_images: Extracted cell images
        digit_grid: Recognized digit grid
        confidence_grid: Confidence scores
        solved_grid: Solved Sudoku grid
        save_dir: Directory to save visualizations (optional)
        show: Whether to show visualizations
    """
    # Create output directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    # Visualize intersections
    intersection_viz = visualize_intersections(
        image,
        intersections,
        save_path=os.path.join(save_dir, "intersections.jpg") if save_dir else None,
        show=show
    )
    
    # Visualize grid
    grid_viz = visualize_grid(
        image,
        grid_points,
        save_path=os.path.join(save_dir, "grid.jpg") if save_dir else None,
        show=show
    )
    
    # Visualize cells
    cells_viz = visualize_cells(
        cell_images,
        save_path=os.path.join(save_dir, "cells.jpg") if save_dir else None,
        show=show
    )
    
    # Visualize digit grid
    digit_viz = visualize_digit_grid(
        digit_grid,
        confidence_grid,
        save_path=os.path.join(save_dir, "digits.jpg") if save_dir else None,
        show=show
    )
    
    # Visualize solution
    solution_viz = visualize_solution(
        digit_grid,
        solved_grid,
        original_image=image,
        save_path=os.path.join(save_dir, "solution.jpg") if save_dir else None,
        show=show
    )
    
    # If show is True, display a summary
    if show:
        print("Processing stages visualized successfully")
        if save_dir:
            print(f"Visualizations saved to: {save_dir}")


def create_visualization_report(
    results: Dict[str, Any],
    image: ImageType,
    save_path: Optional[str] = None,
    show: bool = False
) -> ImageType:
    """
    Create a comprehensive visualization report.
    
    Args:
        results: Pipeline results
        image: Original image
        save_path: Path to save report (optional)
        show: Whether to show report
        
    Returns:
        Report image
    """
    # Extract results
    digit_grid = results.get("digit_grid")
    confidence_grid = results.get("confidence_grid")
    solved_grid = results.get("solved_grid")
    
    # Calculate cell size based on image dimensions
    height, width = image.shape[:2]
    cell_size = min(width, height) // 12
    
    # Create report image
    report_width = width + cell_size * 18  # Original image + 2 grids
    report_height = max(height, cell_size * 10)  # Max of original image height or grid height
    report = np.ones((report_height, report_width, 3), dtype=np.uint8) * 255
    
    # Place original image
    if len(image.shape) == 2:
        image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_colored = image.copy()
        
    # Resize image to fit report
    aspect_ratio = width / height
    if height > report_height:
        new_height = report_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = width
        new_height = height
        
    image_resized = cv2.resize(image_colored, (new_width, new_height))
    report[:new_height, :new_width] = image_resized
    
    # Place recognized digit grid
    if digit_grid is not None and confidence_grid is not None:
        digit_viz = visualize_digit_grid(
            digit_grid,
            confidence_grid,
            cell_size=cell_size,
            show=False
        )
        
        report_offset_x = new_width + cell_size
        report_offset_y = (report_height - cell_size * 9) // 2
        
        report[
            report_offset_y:report_offset_y + cell_size * 9,
            report_offset_x:report_offset_x + cell_size * 9
        ] = digit_viz[:cell_size * 9, :cell_size * 9]
        
        # Add label
        cv2.putText(
            report,
            "Recognized Digits",
            (report_offset_x, report_offset_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
    # Place solved grid
    if digit_grid is not None and solved_grid is not None:
        solution_viz = visualize_solution(
            digit_grid,
            solved_grid,
            cell_size=cell_size,
            show=False
        )
        
        report_offset_x = new_width + cell_size * 10
        report_offset_y = (report_height - cell_size * 9) // 2
        
        report[
            report_offset_y:report_offset_y + cell_size * 9,
            report_offset_x:report_offset_x + cell_size * 9
        ] = solution_viz[:cell_size * 9, :cell_size * 9]
        
        # Add label
        cv2.putText(
            report,
            "Solution",
            (report_offset_x, report_offset_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
    # Add title
    cv2.putText(
        report,
        "Sudoku Recognizer Results",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2
    )
    
    # Add processing time
    processing_time = results.get("processing_time", 0)
    cv2.putText(
        report,
        f"Processing Time: {processing_time:.2f} seconds",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    
    # Save report if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, report)
        
    # Show report if requested
    if show:
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(report, cv2.COLOR_BGR2RGB))
        plt.title("Sudoku Recognizer Report")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    return report


def overlay_solution_on_image(
    image: ImageType,
    digit_grid: GridType,
    solved_grid: GridType,
    grid_points: GridPointsType,
    save_path: Optional[str] = None,
    show: bool = False
) -> ImageType:
    """
    Overlay solution on the original image.
    
    Args:
        image: Original image
        digit_grid: Recognized digit grid
        solved_grid: Solved digit grid
        grid_points: Grid points for mapping
        save_path: Path to save overlay (optional)
        show: Whether to show overlay
        
    Returns:
        Overlay image
    """
    # Create a copy of the image
    overlay = image.copy()
    
    # Convert to color if grayscale
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        
    # Draw semi-transparent grid
    grid_overlay = np.zeros_like(overlay)
    
    # Draw cells
    for i in range(9):
        for j in range(9):
            # Get cell corners
            tl = grid_points[i][j]
            tr = grid_points[i][j+1]
            bl = grid_points[i+1][j]
            br = grid_points[i+1][j+1]
            
            # Create polygon
            pts = np.array([tl, tr, br, bl], dtype=np.int32).reshape((-1, 1, 2))
            
            # Fill cell with semi-transparent white
            cv2.fillPoly(grid_overlay, [pts], (255, 255, 255))
            
    # Blend with original image
    alpha = 0.4
    overlay = cv2.addWeighted(overlay, 1 - alpha, grid_overlay, alpha, 0)
    
    # Draw digits
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    
    for i in range(9):
        for j in range(9):
            initial_digit = digit_grid[i][j]
            solved_digit = solved_grid[i][j]
            
            if initial_digit == 0 and solved_digit > 0:
                # Calculate cell center
                tl = grid_points[i][j]
                tr = grid_points[i][j+1]
                bl = grid_points[i+1][j]
                br = grid_points[i+1][j+1]
                
                center_x = int((tl[0] + tr[0] + bl[0] + br[0]) / 4)
                center_y = int((tl[1] + tr[1] + bl[1] + br[1]) / 4)
                
                # Calculate text size
                text = str(solved_digit)
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                
                # Calculate text position
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                
                # Draw digit with green color for solved digits
                cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
                
    # Save overlay if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, overlay)
        
    # Show overlay if requested
    if show:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Solution Overlay")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    return overlay
