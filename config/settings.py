"""
Sudoku Recognizer Settings Module

This module handles loading, validating, and providing access to system settings.
It enforces critical thresholds and validates configuration parameters.
"""

import os
import logging
from typing import Any, Dict, List, Optional

from .default_fallbacks import DEFAULT_CONFIG, CRITICAL_THRESHOLDS
from utils.error_handling import ConfigError

# Configure logging
logger = logging.getLogger(__name__)

class Settings:
    """
    Centralized settings management with validation and critical thresholds.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings with config from file or defaults.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Import here to avoid circular imports
        from . import load_config
        
        self._config = load_config(config_path)
        self._validate_config()
        
    def _validate_config(self) -> None:
        """
        Validate configuration and enforce critical thresholds.
        
        Raises:
            ConfigError: If configuration contains invalid values
        """
        # Check and enforce critical thresholds
        for key, min_value in CRITICAL_THRESHOLDS.items():
            current_value = self.get(key)
            
            if current_value is not None and current_value < min_value:
                logger.warning(
                    f"Config value {key}={current_value} is below critical threshold {min_value}. "
                    f"Setting to minimum allowed value."
                )
                self.set(key, min_value)
                
        # Validate image size settings
        min_size = self.get("system.min_image_size")
        max_size = self.get("system.max_image_size")
        
        if min_size > max_size:
            logger.error(f"Invalid image size range: min={min_size}, max={max_size}")
            raise ConfigError(f"Invalid configuration: min_image_size ({min_size}) > max_image_size ({max_size})")
            
        # Validate model paths
        for key in ["intersection_detector.model_path", "digit_recognizer.model_path"]:
            model_path = self.get(key)
            if model_path and not os.path.isabs(model_path):
                # Convert relative path to absolute based on project root
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                abs_path = os.path.join(project_root, model_path)
                self.set(key, abs_path)
                
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with dot notation support.
        
        Args:
            key: Configuration key (e.g., "system.log_level")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        # Import here to avoid circular imports
        from . import get_setting
        return get_setting(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value with dot notation support.
        
        Args:
            key: Configuration key (e.g., "system.log_level")
            value: New value to set
        """
        # Import here to avoid circular imports
        from . import set_setting
        set_setting(key, value)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.
        
        Returns:
            Dict with all configuration settings
        """
        return self._config
        
    def get_nested(self, section: str) -> Dict[str, Any]:
        """
        Get all settings within a section.
        
        Args:
            section: Section name (e.g., "system", "intersection_detector")
            
        Returns:
            Dict with section settings or empty dict if section not found
        """
        return self._config.get(section, {})
        
    def is_debug_mode(self) -> bool:
        """
        Check if system is in debug mode.
        
        Returns:
            True if in debug mode, False otherwise
        """
        return self.get("system.debug_mode", False)
        
    def get_fallback_methods(self, component: str) -> List[str]:
        """
        Get list of fallback methods for a component.
        
        Args:
            component: Component name
            
        Returns:
            List of fallback method names
        """
        if component == "intersection_detector":
            return self.get("intersection_detector.detection_methods", ["cnn", "hough"])
        elif component == "grid_reconstructor":
            return self.get("grid_reconstructor.grid_detection_methods", ["ransac", "hough"])
        elif component == "digit_recognizer":
            return self.get("digit_recognizer.fallback_models", ["cnn", "svm"])
        else:
            return []
            
    def get_recovery_strategy(self, failure_type: str) -> List[str]:
        """
        Get recovery strategy for a specific failure type.
        
        Args:
            failure_type: Type of failure
            
        Returns:
            List of recovery steps
        """
        strategies = self.get("pipeline.recovery_strategies", {})
        return strategies.get(failure_type, [])


# Global settings instance
_settings: Optional[Settings] = None


def initialize_settings(config_path: Optional[str] = None) -> Settings:
    """
    Initialize global settings instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Settings instance
    """
    global _settings
    _settings = Settings(config_path)
    return _settings


def get_settings() -> Settings:
    """
    Get global settings instance, initializing if necessary.
    
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = initialize_settings()
    return _settings
