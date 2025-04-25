#!/usr/bin/env python3
"""
Flask Web Application for Sudoku Recognizer.
"""

import os
import sys
import logging
import time
import uuid
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from flask import (
    Flask, render_template, request, redirect, url_for, jsonify, flash, session
)
from werkzeug.utils import secure_filename

from config import initialize_settings
from utils.error_handling import setup_exception_handling
from models.digit_recognizer import RobustDigitRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global model instance
digit_recognizer = None

def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_exceptions(func):
    """Decorator for handling exceptions in routes."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in route {func.__name__}: {str(e)}")
            flash(f"An error occurred: {str(e)}", "danger")
            return redirect(url_for('index'))
    wrapper.__name__ = func.__name__
    return wrapper

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@handle_exceptions
def upload_file():
    """Handle file upload."""
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
        
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(request.url)
        
    # Check if file is allowed
    if not allowed_file(file.filename):
        flash(f'File type not allowed. Please upload {", ".join(ALLOWED_EXTENSIONS)}', 'danger')
        return redirect(request.url)
        
    # Generate session ID if not present
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        
    # Create session upload directory
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], session['session_id'])
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save file
    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)
    
    # Process file (simplified, just show the image for now)
    try:
        # Store the file path in session
        session['file_path'] = os.path.relpath(file_path)
        
        # Redirect to results page
        return redirect(url_for('show_results'))
        
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/results')
@handle_exceptions
def show_results():
    """Show processing results."""
    # Check if file path exists in session
    if 'file_path' not in session:
        flash('No image uploaded', 'warning')
        return redirect(url_for('index'))
        
    file_path = session['file_path']
    
    # For now, just show the uploaded image
    results = {
        'file_path': file_path,
        'processing_time': 0.0,
        'grid_detected': False,
        'cells_extracted': False,
        'digits_recognized': False,
        'puzzle_solved': False
    }
    
    return render_template('results.html', results=results)

@app.route('/manual_input', methods=['GET', 'POST'])
@handle_exceptions
def manual_input():
    """Handle manual Sudoku grid input."""
    if request.method == 'POST':
        try:
            # Parse grid from form data
            grid = [[0 for _ in range(9)] for _ in range(9)]
            
            for i in range(9):
                for j in range(9):
                    cell_value = request.form.get(f'cell_{i}_{j}', '')
                    if cell_value.isdigit() and 1 <= int(cell_value) <= 9:
                        grid[i][j] = int(cell_value)
                        
            # Store the grid in session
            session['manual_grid'] = grid
            
            # Redirect to results page
            return redirect(url_for('show_manual_results'))
            
        except Exception as e:
            flash(f'Error processing input: {str(e)}', 'danger')
            
    return render_template('manual_input.html')

@app.route('/manual_results')
@handle_exceptions
def show_manual_results():
    """Show manual input results."""
    # Check if grid exists in session
    if 'manual_grid' not in session:
        flash('No grid entered', 'warning')
        return redirect(url_for('manual_input'))
        
    grid = session['manual_grid']
    
    # For now, just show the grid
    results = {
        'digit_grid': grid,
        'is_manual': True,
        'processing_time': 0.0,
        'digits_recognized': True,
        'puzzle_solved': False
    }
    
    return render_template('results.html', results=results)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash(f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)} MB', 'danger')
    return redirect(url_for('index'))

@app.errorhandler(500)
def server_error(e):
    """Handle server error."""
    logger.error(f"Server error: {str(e)}")
    flash('An unexpected error occurred. Please try again.', 'danger')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Handle not found error."""
    flash('Page not found', 'warning')
    return redirect(url_for('index'))

def initialize_app(model_dir: str = 'data/models'):
    """Initialize the application."""
    global digit_recognizer
    
    # Set up exception handling
    setup_exception_handling()
    
    # Initialize settings
    initialize_settings()
    
    # Load models if available
    digit_recognizer = RobustDigitRecognizer()
    model_path = os.path.join(model_dir, "digit_recognizer.h5")
    
    if os.path.exists(model_path):
        logger.info(f"Loading digit recognizer from {model_path}")
        digit_recognizer.load(model_path)
    else:
        logger.warning(f"Digit recognizer model not found at {model_path}")

if __name__ == '__main__':
    # Initialize application
    initialize_app()
    
    # Run application
    app.run(host='0.0.0.0', port=5000, debug=True)
