#!/usr/bin/env python3
"""
Flask Web Application for Sudoku Recognizer.

This application provides a web interface for the Sudoku recognizer system.
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
from utils.validation import validate_image_file
from pipeline import SudokuRecognizerPipeline
from utils.visualization import (
    visualize_intersection, visualize_grid, visualize_cells,
    visualize_digit_grid, visualize_solution, overlay_solution_on_image
)

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

# Global pipeline instance
pipeline = None


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


def process_image(image_path: str) -> Dict[str, Any]:
    """Process image with Sudoku recognizer pipeline."""
    global pipeline
    
    if pipeline is None:
        flash("Pipeline not initialized", "danger")
        return {}
        
    try:
        # Process image
        results = pipeline.process_image(image_path)
        
        # Generate result paths
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        session_id = session.get('session_id', str(uuid.uuid4()))
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Save visualizations if successful
        visualization_paths = {}
        
        if results.get("grid_detected"):
            # Save grid visualization
            grid_viz_path = os.path.join(result_dir, f"{base_filename}_grid.jpg")
            grid_img = visualize_grid(
                pipeline.current_state["image"],
                pipeline.current_state["grid_points"],
                save_path=grid_viz_path
            )
            visualization_paths["grid"] = os.path.relpath(grid_viz_path)
            
        if results.get("cells_extracted"):
            # Save cells visualization
            cells_viz_path = os.path.join(result_dir, f"{base_filename}_cells.jpg")
            cells_img = visualize_cells(
                pipeline.current_state["cell_images"],
                save_path=cells_viz_path
            )
            visualization_paths["cells"] = os.path.relpath(cells_viz_path)
            
        if results.get("digits_recognized"):
            # Save digit grid visualization
            digits_viz_path = os.path.join(result_dir, f"{base_filename}_digits.jpg")
            digits_img = visualize_digit_grid(
                results["digit_grid"],
                results["confidence_grid"],
                save_path=digits_viz_path
            )
            visualization_paths["digits"] = os.path.relpath(digits_viz_path)
            
        if results.get("puzzle_solved"):
            # Save solution visualization
            solution_viz_path = os.path.join(result_dir, f"{base_filename}_solution.jpg")
            solution_img = visualize_solution(
                results["digit_grid"],
                results["solved_grid"],
                save_path=solution_viz_path
            )
            visualization_paths["solution"] = os.path.relpath(solution_viz_path)
            
            # Save overlay visualization
            overlay_viz_path = os.path.join(result_dir, f"{base_filename}_overlay.jpg")
            overlay_img = overlay_solution_on_image(
                pipeline.current_state["image"],
                results["digit_grid"],
                results["solved_grid"],
                pipeline.current_state["grid_points"],
                save_path=overlay_viz_path
            )
            visualization_paths["overlay"] = os.path.relpath(overlay_viz_path)
            
        # Add visualization paths to results
        results["visualization_paths"] = visualization_paths
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


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
    
    # Process file
    try:
        start_time = time.time()
        results = process_image(file_path)
        processing_time = time.time() - start_time
        
        # Add file path and timing to results
        results['file_path'] = os.path.relpath(file_path)
        results['processing_time'] = processing_time
        
        # Store results in session
        session['results'] = results
        
        # Redirect to results page
        return redirect(url_for('show_results'))
        
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'danger')
        return redirect(url_for('index'))


@app.route('/results')
@handle_exceptions
def show_results():
    """Show processing results."""
    # Check if results exist in session
    if 'results' not in session:
        flash('No results found', 'warning')
        return redirect(url_for('index'))
        
    results = session['results']
    
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
                        
            # Solve grid
            global pipeline
            if pipeline is None:
                flash("Pipeline not initialized", "danger")
                return redirect(url_for('index'))
                
            # Solve the puzzle
            start_time = time.time()
            solved_grid = pipeline.solver.solve(grid)
            solving_time = time.time() - start_time
            
            # Generate session ID if not present
            if 'session_id' not in session:
                session['session_id'] = str(uuid.uuid4())
                
            # Save visualizations
            result_dir = os.path.join(app.config['RESULTS_FOLDER'], session['session_id'])
            os.makedirs(result_dir, exist_ok=True)
            
            # Create visualization
            timestamp = int(time.time())
            solution_viz_path = os.path.join(result_dir, f"manual_solution_{timestamp}.jpg")
            solution_img = visualize_solution(grid, solved_grid, save_path=solution_viz_path)
            
            # Create results
            results = {
                'digit_grid': grid,
                'solved_grid': solved_grid,
                'solving_time': solving_time,
                'processing_time': solving_time,
                'digits_recognized': True,
                'puzzle_solved': True,
                'is_manual': True,
                'visualization_paths': {
                    'solution': os.path.relpath(solution_viz_path)
                }
            }
            
            # Store results in session
            session['results'] = results
            
            # Redirect to results page
            return redirect(url_for('show_results'))
            
        except Exception as e:
            flash(f'Error solving puzzle: {str(e)}', 'danger')
            
    return render_template('manual_input.html')


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


def initialize_app(model_dir: str = 'data/models', config_path: str = 'config/settings.json'):
    """Initialize the application."""
    global pipeline
    
    # Initialize settings
    settings = initialize_settings(config_path)
    
    # Set up exception handling
    setup_exception_handling()
    
    # Initialize pipeline
    pipeline = SudokuRecognizerPipeline()
    
    # Load models
    logger.info(f"Loading models from {model_dir}")
    if not pipeline.load_models(model_dir):
        logger.error(f"Failed to load models from {model_dir}")


if __name__ == '__main__':
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Start Sudoku Recognizer web application')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--model-dir', type=str, default='data/models', help='Directory containing trained models')
    parser.add_argument('--config', type=str, default='config/settings.json', help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize application
    initialize_app(args.model_dir, args.config)
    
    # Run application
    app.run(host=args.host, port=args.port, debug=args.debug)
