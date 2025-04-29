"""
Sudoku Recognizer Configuration Module

This module provides centralized configuration management with error handling
and fallback mechanisms.
"""

import os
import json
import logging
from typing import Any, Dict, Optional

from .default_fallbacks import DEFAULT_CONFIG
from ..utils.error_handling import ConfigError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global configuration dictionary
_config: Dict[str, Any] = {}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file with fallback to default values.
    
    Args:
        config_path: Path to configuration file (JSON format)
        
    Returns:
        Dict containing configuration settings
        
    Raises:
        ConfigError: If configuration file exists but cannot be parsed
    """
    global _config
    
    # Start with default configuration
    _config = DEFAULT_CONFIG.copy()
    
    # If config path is provided, try to load it
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                
            # Update default config with user settings
            _config.update(user_config)
            logger.info(f"Configuration loaded from {config_path}")
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            logger.warning("Using default configuration")
            raise ConfigError(f"Failed to load configuration: {str(e)}")
    else:
        logger.info("No configuration file provided or found, using defaults")
    
    return _config


def get_config() -> Dict[str, Any]:
    """
    Get the current configuration dictionary.
    
    Returns:
        Dict containing configuration settings
    """
    if not _config:
        return load_config()
    return _config


def get_setting(key: str, default: Any = None) -> Any:
    """
    Get a specific configuration setting with fallback.
    
    Args:
        key: Configuration key to retrieve
        default: Default value if key is not found
        
    Returns:
        Configuration value or default
    """
    config = get_config()
    
    # Support nested keys with dot notation (e.g., "model.threshold")
    if '.' in key:
        parts = key.split('.')
        value = config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                logger.warning(f"Configuration key '{key}' not found, using default: {default}")
                return default
        return value
    
    return config.get(key, default)


def set_setting(key: str, value: Any) -> None:
    """
    Update a specific configuration setting.
    
    Args:
        key: Configuration key to update
        value: New value for the key
    """
    global _config
    
    # Support nested keys with dot notation
    if '.' in key:
        parts = key.split('.')
        config = _config
        for i, part in enumerate(parts[:-1]):
            if part not in config:
                config[part] = {}
            config = config[part]
        config[parts[-1]] = value
    else:
        _config[key] = value
    
    logger.debug(f"Configuration updated: {key} = {value}")


def save_config(config_path: str) -> None:
    """
    Save current configuration to file.
    
    Args:
        config_path: Path to save configuration file
        
    Raises:
        ConfigError: If configuration cannot be saved
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(_config, f, indent=4)
            
        logger.info(f"Configuration saved to {config_path}")
        
    except (IOError, OSError) as e:
        logger.error(f"Failed to save configuration to {config_path}: {str(e)}")
        raise ConfigError(f"Failed to save configuration: {str(e)}")
