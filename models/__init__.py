import logging
"""
Sudoku Recognizer Core Models.

This module provides abstract base classes for all system components.
"""
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

import abc
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

# Define common types
ImageType = np.ndarray  # OpenCV image (numpy array)
PointType = Tuple[int, int]  # (x, y) coordinates
GridType = List[List[int]]  # 9x9 grid of integers (0 = empty)


class ModelBase(abc.ABC):
    """Abstract base class for all models."""
    
    @abc.abstractmethod
    def load(self, model_path: str) -> bool:
        """
        Load model from file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if successful
            
        Raises:
            SudokuRecognizerError: If loading fails
        """
        pass
    
    @abc.abstractmethod
    def save(self, model_path: str) -> bool:
        """
        Save model to file.
        
        Args:
            model_path: Path to save model
            
        Returns:
            True if successful
            
        Raises:
            SudokuRecognizerError: If saving fails
        """
        pass


class IntersectionDetectorBase(ModelBase):
    """Abstract base class for intersection detectors."""
    
    @abc.abstractmethod
    def detect(self, image: ImageType) -> List[PointType]:
        """
        Detect grid line intersections in an image.
        
        Args:
            image: Input image
            
        Returns:
            List of intersection points (x, y)
            
        Raises:
            SudokuRecognizerError: If detection fails
        """
        pass
    
    @abc.abstractmethod
    def train(self, images: List[ImageType], annotations: List[List[PointType]]) -> None:
        """
        Train the intersection detector.
        
        Args:
            images: List of training images
            annotations: List of intersection point annotations for each image
            
        Raises:
            SudokuRecognizerError: If training fails
        """
        pass


class GridReconstructorBase(ModelBase):
    """Abstract base class for grid reconstructors."""
    
    @abc.abstractmethod
    def reconstruct(self, points: List[PointType], image_shape: Tuple[int, int]) -> List[List[PointType]]:
        """
        Reconstruct grid from intersection points.
        
        Args:
            points: List of detected intersection points
            image_shape: Shape of the original image
            
        Returns:
            2D list of ordered grid points (outer index is row, inner is column)
            
        Raises:
            SudokuRecognizerError: If reconstruction fails
        """
        pass


class CellExtractorBase(ModelBase):
    """Abstract base class for cell extractors."""
    
    @abc.abstractmethod
    def extract(self, image: ImageType, grid_points: List[List[PointType]]) -> List[List[ImageType]]:
        """
        Extract cell images from grid.
        
        Args:
            image: Original image
            grid_points: 2D list of ordered grid points
            
        Returns:
            2D list of cell images (9x9 for standard Sudoku)
            
        Raises:
            SudokuRecognizerError: If extraction fails
        """
        pass


class DigitRecognizerBase(ModelBase):
    """Abstract base class for digit recognizers."""
    
    @abc.abstractmethod
    def recognize(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """
        Recognize digits in cell images.
        
        Args:
            cell_images: 2D list of cell images
            
        Returns:
            Tuple of (grid of recognized digits, grid of confidence scores)
            
        Raises:
            SudokuRecognizerError: If recognition fails
        """
        pass
    
    @abc.abstractmethod
    def train(self, cell_images: List[ImageType], labels: List[int]) -> None:
        """
        Train the digit recognizer.
        
        Args:
            cell_images: List of cell images
            labels: List of digit labels (0-9, where 0 is empty)
            
        Raises:
            SudokuRecognizerError: If training fails
        """
        pass


class SolverBase(ModelBase):
    """Abstract base class for Sudoku solvers."""
    
    @abc.abstractmethod
    def solve(self, grid: GridType) -> GridType:
        """
        Solve a Sudoku puzzle.
        
        Args:
            grid: 9x9 grid with initial values (0 for empty)
            
        Returns:
            Solved 9x9 grid
            
        Raises:
            SudokuRecognizerError: If solving fails or puzzle is unsolvable
        """
        pass
