"""
Sudoku Recognizer Core Models.
"""

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
        """Load model from file."""
        pass
    
    @abc.abstractmethod
    def save(self, model_path: str) -> bool:
        """Save model to file."""
        pass

class DigitRecognizerBase(ModelBase):
    """Abstract base class for digit recognizers."""
    
    @abc.abstractmethod
    def recognize(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """Recognize digits in cell images."""
        pass
    
    @abc.abstractmethod
    def train(self, cell_images: List[ImageType], labels: List[int]) -> None:
        """Train the digit recognizer."""
        pass
