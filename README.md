# Sudoku Recognizer

A robust computer vision system for recognizing and solving Sudoku puzzles from images with comprehensive defensive programming against all failure modes.

## Features

- **Grid Detection**: Uses the "Drop of Water" approach to detect grid intersections and reconstruct the Sudoku grid
- **Cell Extraction**: Extracts individual cells using perspective transformation
- **Digit Recognition**: Recognizes digits in cells using multiple models with fallback mechanisms
- **Puzzle Solving**: Solves the Sudoku puzzle using constraint propagation and backtracking algorithms
- **Web Interface**: User-friendly interface for uploading and processing Sudoku images
- **Defensive Programming**: Comprehensive error handling, validation, and recovery strategies

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV
- TensorFlow 2.x
- Flask (for web interface)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sudoku-recognizer.git
   cd sudoku-recognizer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Models

Train models with your own dataset:

```bash
python train.py --data-dir data/raw --processed-dir data/processed --model-dir data/models
```

### Web Application

Start the Flask application:

```bash
python app.py
```

Open a web browser and navigate to `http://localhost:5000`

## Training in Google Colab

The project includes a Colab notebook for training with GPU acceleration:

1. Upload `training_notebook.ipynb` to Google Colab
2. Upload your dataset to Google Drive
3. Follow the steps in the notebook

## Deployment

The project includes a `render.yaml` file for easy deployment on Render.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
