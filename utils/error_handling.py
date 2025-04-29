"""
Centralized error handling and logging system.

This module provides a comprehensive set of custom exceptions, error handlers,
and recovery mechanisms for the Sudoku Recognizer system.
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
        """
        Initialize exception with message and optional details.
        
        Args:
            message: Error message
            details: Additional error details and context
        """
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


class ImageFormatError(ImageError):
    """Unsupported or invalid image format."""
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


# Recovery results
class RecoveryResult:
    """Result of a recovery attempt."""
    
    def __init__(
        self, 
        success: bool, 
        data: Any = None, 
        error: Optional[Exception] = None,
        recovery_method: Optional[str] = None
    ):
        """
        Initialize recovery result.
        
        Args:
            success: Whether recovery was successful
            data: Recovered data (if successful)
            error: Original error (if unsuccessful)
            recovery_method: Name of recovery method used
        """
        self.success = success
        self.data = data
        self.error = error
        self.recovery_method = recovery_method
        
    def __bool__(self) -> bool:
        """Allow boolean evaluation based on success."""
        return self.success


def log_error(
    error: Exception, 
    level: int = logging.ERROR,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an error with context and traceback.
    
    Args:
        error: Exception to log
        level: Logging level
        context: Additional context information
    """
    ctx_str = f" [Context: {context}]" if context else ""
    
    if isinstance(error, SudokuRecognizerError) and error.details:
        ctx_str += f" [Details: {error.details}]"
    
    error_type = type(error).__name__
    error_tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    
    logger.log(level, f"{error_type}: {str(error)}{ctx_str}\n{error_tb}")


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay_seconds: Initial delay between retries in seconds
        backoff_factor: Multiplicative factor for backoff
        exceptions: Tuple of exceptions to catch and retry
        logger: Logger instance to use (uses module logger if None)
        
    Returns:
        Decorated function with retry logic
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


def timeout(seconds: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Timeout decorator to limit execution time.
    
    Note: This implementation uses signal, which only works on Unix systems.
    For Windows compatibility, consider using threading or multiprocessing.
    
    Args:
        seconds: Maximum execution time in seconds
        
    Returns:
        Decorated function with timeout logic
        
    Raises:
        TimeoutError: If function execution exceeds the time limit
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            import signal
            
            def handler(signum: int, frame: Any) -> None:
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set timeout handler
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore previous handler and cancel alarm
                signal.signal(signal.SIGALRM, old_handler)
                signal.alarm(0)
                
            return result
            
        return wrapper
    return decorator


def fallback(
    fallback_function: Callable[..., T],
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Fallback decorator to use alternative function on failure.
    
    Args:
        fallback_function: Function to call if primary function fails
        exceptions: Exceptions to catch and trigger fallback
        logger: Logger instance to use
        
    Returns:
        Decorated function with fallback logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            local_logger = logger or logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                local_logger.warning(
                    f"Function {func.__name__} failed with {type(e).__name__}: {str(e)}. "
                    f"Using fallback function {fallback_function.__name__}."
                )
                return fallback_function(*args, **kwargs)
                
        return wrapper
    return decorator


def robust_method(max_retries: int = 3, timeout_sec: float = 30.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Combined decorator for robust method execution with retry and timeout.
    
    Args:
        max_retries: Maximum number of retry attempts
        timeout_sec: Maximum execution time in seconds
        
    Returns:
        Decorated function with retry and timeout logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Apply timeout then retry decorators
        timed_func = timeout(timeout_sec)(func)
        return retry(max_attempts=max_retries)(timed_func)
    return decorator


class ErrorHandler:
    """
    Error handler with recovery strategies.
    """
    
    def __init__(self):
        """Initialize error handler with default recovery strategies."""
        # Set up default recovery strategies
        self._recovery_strategies: Dict[Type[Exception], List[Callable]] = {}
        
    def register_recovery(
        self,
        exception_type: Type[Exception],
        recovery_func: Callable[..., Any]
    ) -> None:
        """
        Register a recovery function for an exception type.
        
        Args:
            exception_type: Exception type to handle
            recovery_func: Function to call for recovery
        """
        if exception_type not in self._recovery_strategies:
            self._recovery_strategies[exception_type] = []
            
        self._recovery_strategies[exception_type].append(recovery_func)
        
    def try_recover(
        self,
        error: Exception,
        *args: Any,
        **kwargs: Any
    ) -> RecoveryResult:
        """
        Try to recover from an error using registered strategies.
        
        Args:
            error: Exception to recover from
            *args: Arguments to pass to recovery function
            **kwargs: Keyword arguments to pass to recovery function
            
        Returns:
            RecoveryResult indicating success or failure
        """
        # Find the most specific matching exception type
        matching_types = [
            exc_type for exc_type in self._recovery_strategies.keys()
            if isinstance(error, exc_type)
        ]
        
        if not matching_types:
            return RecoveryResult(False, error=error)
            
        # Sort by specificity (most specific first)
        matching_types.sort(key=lambda t: len(t.__mro__), reverse=True)
        most_specific_type = matching_types[0]
        
        # Try each recovery strategy in order
        for i, recovery_func in enumerate(self._recovery_strategies[most_specific_type]):
            try:
                logger.info(f"Attempting recovery strategy {i+1} for {type(error).__name__}: {recovery_func.__name__}")
                result = recovery_func(*args, **kwargs)
                return RecoveryResult(True, data=result, recovery_method=recovery_func.__name__)
            except Exception as e:
                logger.warning(f"Recovery strategy {recovery_func.__name__} failed: {str(e)}")
                
        # If all strategies failed
        return RecoveryResult(False, error=error)
        
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> RecoveryResult:
        """
        Handle error with logging and recovery attempt.
        
        Args:
            error: Exception to handle
            context: Error context for logging
            *args: Arguments for recovery functions
            **kwargs: Keyword arguments for recovery functions
            
        Returns:
            RecoveryResult indicating success or failure
        """
        # Log the error
        log_error(error, context=context)
        
        # Try to recover
        return self.try_recover(error, *args, **kwargs)


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """
    Get global error handler instance, initializing if necessary.
    
    Returns:
        ErrorHandler instance
    """
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def register_recovery_strategy(
    exception_type: Type[Exception],
    recovery_func: Callable[..., Any]
) -> None:
    """
    Register a recovery strategy with the global error handler.
    
    Args:
        exception_type: Exception type to handle
        recovery_func: Function to call for recovery
    """
    handler = get_error_handler()
    handler.register_recovery(exception_type, recovery_func)


def safe_execute(
    func: Callable[..., T],
    *args: Any,
    error_type: Type[SudokuRecognizerError] = SudokuRecognizerError,
    error_msg: str = "Function execution failed",
    **kwargs: Any
) -> T:
    """
    Execute a function safely, converting all exceptions to a specific error type.
    
    Args:
        func: Function to execute
        *args: Arguments to pass to function
        error_type: Type of error to raise on failure
        error_msg: Error message to use
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Function result
        
    Raises:
        The specified error_type with original exception details
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if isinstance(e, error_type):
            # Preserve the original error if it's already the right type
            raise
            
        # Create a new error of the specified type with details
        details = {
            "original_error": str(e),
            "original_type": type(e).__name__,
            "function": func.__name__
        }
        
        raise error_type(f"{error_msg}: {str(e)}", details) from e


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
        
        Args:
            exc_type: Exception type
            exc_value: Exception instance
            exc_traceback: Exception traceback
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
