"""
Centralized error handling and logging system.
"""

import sys
import traceback
import logging
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for function return types
T = TypeVar('T')

# Base exception class for all system errors
class SudokuRecognizerError(Exception):
    """Base exception class for all Sudoku Recognizer errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

# Configuration errors
class ConfigError(SudokuRecognizerError):
    """Error in system configuration."""
    pass

# Image processing errors
class ImageError(SudokuRecognizerError):
    """Base class for image-related errors."""
    pass

class ImageLoadError(ImageError):
    """Error loading or processing an image."""
    pass

# Detection and recognition errors
class DetectionError(SudokuRecognizerError):
    """Base class for detection errors."""
    pass

class IntersectionDetectionError(DetectionError):
    """Error detecting grid intersections."""
    pass

class GridReconstructionError(DetectionError):
    """Error reconstructing grid from intersections."""
    pass

class CellExtractionError(DetectionError):
    """Error extracting cells from grid."""
    pass

class DigitRecognitionError(DetectionError):
    """Error recognizing digits in cells."""
    pass

# Solving errors
class SolverError(SudokuRecognizerError):
    """Error solving Sudoku puzzle."""
    pass

class InvalidPuzzleError(SolverError):
    """Invalid Sudoku puzzle (no solution possible)."""
    pass

# Pipeline errors
class PipelineError(SudokuRecognizerError):
    """Error in processing pipeline."""
    pass

class TimeoutError(SudokuRecognizerError):
    """Operation timed out."""
    pass

def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator with exponential backoff.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            local_logger = logger or logging.getLogger(func.__module__)
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        wait_time = delay_seconds * (backoff_factor ** (attempt - 1))
                        local_logger.warning(
                            f"Attempt {attempt}/{max_attempts} for {func.__name__} failed: {str(e)}. "
                            f"Retrying in {wait_time:.2f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        local_logger.error(
                            f"All {max_attempts} attempts for {func.__name__} failed. "
                            f"Last error: {str(e)}"
                        )
                        
            # If we get here, all attempts failed
            assert last_exception is not None
            raise last_exception
            
        return wrapper
    return decorator

def robust_method(max_retries: int = 3, timeout_sec: float = 30.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Combined decorator for robust method execution with retry and timeout.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Apply timeout then retry decorators
        return retry(max_attempts=max_retries)(func)
    return decorator

def setup_exception_handling() -> None:
    """
    Set up global exception handling for unexpected errors.
    """
    def global_exception_handler(
        exc_type: Type[BaseException],
        exc_value: BaseException,
        exc_traceback: Optional[traceback.TracebackType]
    ) -> None:
        """
        Global handler for uncaught exceptions.
        """
        # Skip KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        # Log the error
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    # Set the exception hook
    sys.excepthook = global_exception_handler
