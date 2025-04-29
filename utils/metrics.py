"""
Performance Evaluation Metrics.

This module provides functions for evaluating model performance and generating
reports for the Sudoku recognition system.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from ..config import get_settings

# Define types
ImageType = np.ndarray
PointType = Tuple[int, int]
GridType = List[List[int]]

# Configure logging
logger = logging.getLogger(__name__)


def evaluate_intersection_detector(
    detector,
    test_images: List[ImageType],
    test_points: List[List[PointType]],
    distance_threshold: int = 15,
    save_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate intersection detector performance.
    
    Args:
        detector: Intersection detector model
        test_images: List of test images
        test_points: List of ground truth points for each image
        distance_threshold: Maximum distance for a point to be considered correct
        save_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    if len(test_images) != len(test_points):
        raise ValueError("Number of test images and point sets must match")
        
    if len(test_images) == 0:
        raise ValueError("Empty test set")
        
    # Metrics
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_time = 0.0
    
    # Process each image
    for i, (image, ground_truth) in enumerate(zip(test_images, test_points)):
        try:
            # Measure detection time
            start_time = time.time()
            detected_points = detector.detect(image)
            detection_time = time.time() - start_time
            
            # Calculate metrics
            precision, recall, f1 = calculate_point_detection_metrics(
                detected_points, ground_truth, distance_threshold
            )
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_time += detection_time
            
            # Log results
            logger.info(
                f"Image {i+1}/{len(test_images)}: "
                f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, "
                f"Time={detection_time:.4f}s"
            )
            
        except Exception as e:
            logger.error(f"Error evaluating image {i}: {str(e)}")
            
    # Calculate averages
    avg_precision = total_precision / len(test_images)
    avg_recall = total_recall / len(test_images)
    avg_f1 = total_f1 / len(test_images)
    avg_time = total_time / len(test_images)
    
    # Create results dictionary
    results = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1,
        "average_time": avg_time
    }
    
    # Save results if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics as text file
        with open(os.path.join(save_dir, "intersection_detector_metrics.txt"), "w") as f:
            f.write("Intersection Detector Evaluation\n")
            f.write("===============================\n\n")
            f.write(f"Number of test images: {len(test_images)}\n")
            f.write(f"Average precision: {avg_precision:.4f}\n")
            f.write(f"Average recall: {avg_recall:.4f}\n")
            f.write(f"Average F1 score: {avg_f1:.4f}\n")
            f.write(f"Average detection time: {avg_time:.4f}s\n")
            
        # Plot precision-recall bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(['Precision', 'Recall', 'F1 Score'], [avg_precision, avg_recall, avg_f1])
        plt.ylim(0, 1.0)
        plt.title('Intersection Detector Performance')
        plt.ylabel('Score')
        plt.savefig(os.path.join(save_dir, "intersection_detector_metrics.png"))
        
    return results


def calculate_point_detection_metrics(
    detected_points: List[PointType],
    ground_truth_points: List[PointType],
    distance_threshold: int = 15
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score for point detection.
    
    Args:
        detected_points: List of detected points
        ground_truth_points: List of ground truth points
        distance_threshold: Maximum distance for a point to be considered correct
        
    Returns:
        Tuple of (precision, recall, F1 score)
    """
    if not ground_truth_points:
        return 0.0, 0.0, 0.0
        
    if not detected_points:
        return 0.0, 0.0, 0.0
        
    # Count true positives, false positives, and false negatives
    tp = 0
    matched_gt = set()
    
    # For each detected point, find closest ground truth point
    for d_point in detected_points:
        min_distance = float('inf')
        closest_gt_idx = None
        
        for gt_idx, gt_point in enumerate(ground_truth_points):
            # Skip already matched ground truth points
            if gt_idx in matched_gt:
                continue
                
            # Calculate distance
            distance = np.sqrt(
                (d_point[0] - gt_point[0]) ** 2 +
                (d_point[1] - gt_point[1]) ** 2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_gt_idx = gt_idx
                
        # If closest point is within threshold, count as true positive
        if closest_gt_idx is not None and min_distance <= distance_threshold:
            tp += 1
            matched_gt.add(closest_gt_idx)
            
    # Calculate metrics
    fp = len(detected_points) - tp
    fn = len(ground_truth_points) - tp
    
    # Calculate precision, recall, and F1 score
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    return precision, recall, f1


def evaluate_digit_recognizer(
    recognizer,
    test_cells: List[ImageType],
    test_labels: List[int],
    save_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate digit recognizer performance.
    
    Args:
        recognizer: Digit recognizer model
        test_cells: List of test cell images
        test_labels: List of ground truth digit labels
        save_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    if len(test_cells) != len(test_labels):
        raise ValueError("Number of test cells and labels must match")
        
    if len(test_cells) == 0:
        raise ValueError("Empty test set")
        
    # Prepare test cells in 9x9 grids
    num_test_grids = (len(test_cells) + 80) // 81  # Ceiling division
    test_grids = []
    
    for i in range(num_test_grids):
        start_idx = i * 81
        end_idx = min(start_idx + 81, len(test_cells))
        
        # Create a 9x9 grid with available cells
        grid = [[None for _ in range(9)] for _ in range(9)]
        for j in range(start_idx, end_idx):
            row = (j - start_idx) // 9
            col = (j - start_idx) % 9
            grid[row][col] = test_cells[j]
            
        # Fill remaining cells with blank images
        for row in range(9):
            for col in range(9):
                if grid[row][col] is None:
                    grid[row][col] = np.zeros_like(test_cells[0])
                    
        test_grids.append(grid)
        
    # Similar for labels
    label_grids = []
    
    for i in range(num_test_grids):
        start_idx = i * 81
        end_idx = min(start_idx + 81, len(test_labels))
        
        # Create a 9x9 grid with available labels
        grid = [[0 for _ in range(9)] for _ in range(9)]
        for j in range(start_idx, end_idx):
            row = (j - start_idx) // 9
            col = (j - start_idx) % 9
            grid[row][col] = test_labels[j]
            
        label_grids.append(grid)
        
    # Metrics
    all_predictions = []
    all_confidence = []
    total_time = 0.0
    
    # Process each grid
    for i, (grid, label_grid) in enumerate(zip(test_grids, label_grids)):
        try:
            # Measure recognition time
            start_time = time.time()
            digit_grid, confidence_grid = recognizer.recognize(grid)
            recognition_time = time.time() - start_time
            
            # Flatten grids
            pred_digits = [digit_grid[r][c] for r in range(9) for c in range(9)]
            pred_conf = [confidence_grid[r][c] for r in range(9) for c in range(9)]
            true_digits = [label_grid[r][c] for r in range(9) for c in range(9)]
            
            # Append to overall results
            all_predictions.extend(list(zip(true_digits, pred_digits, pred_conf)))
            total_time += recognition_time
            
        except Exception as e:
            logger.error(f"Error evaluating grid {i}: {str(e)}")
            
    # Extract prediction results
    true_digits = [p[0] for p in all_predictions]
    pred_digits = [p[1] for p in all_predictions]
    confidences = [p[2] for p in all_predictions]
    
    # Calculate metrics
    accuracy = accuracy_score(true_digits, pred_digits)
    precision = precision_score(true_digits, pred_digits, average='macro', zero_division=0)
    recall = recall_score(true_digits, pred_digits, average='macro', zero_division=0)
    f1 = f1_score(true_digits, pred_digits, average='macro', zero_division=0)
    
    # Calculate per-digit metrics
    per_digit_metrics = {}
    for digit in range(10):  # 0-9
        digit_indices = [i for i, d in enumerate(true_digits) if d == digit]
        if digit_indices:
            digit_true = [true_digits[i] for i in digit_indices]
            digit_pred = [pred_digits[i] for i in digit_indices]
            digit_accuracy = accuracy_score(digit_true, digit_pred)
            per_digit_metrics[digit] = digit_accuracy
            
    # Calculate average confidence
    avg_confidence = np.mean(confidences)
    avg_time_per_grid = total_time / len(test_grids)
    
    # Create confusion matrix
    cm = confusion_matrix(true_digits, pred_digits, labels=range(10))
    
    # Create results dictionary
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "average_confidence": avg_confidence,
        "average_time_per_grid": avg_time_per_grid
    }
    
    # Add per-digit accuracies
    for digit, acc in per_digit_metrics.items():
        results[f"digit_{digit}_accuracy"] = acc
        
    # Save results if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics as text file
        with open(os.path.join(save_dir, "digit_recognizer_metrics.txt"), "w") as f:
            f.write("Digit Recognizer Evaluation\n")
            f.write("=========================\n\n")
            f.write(f"Number of test cells: {len(test_cells)}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 score: {f1:.4f}\n")
            f.write(f"Average confidence: {avg_confidence:.4f}\n")
            f.write(f"Average recognition time per grid: {avg_time_per_grid:.4f}s\n\n")
            f.write("Per-digit accuracies:\n")
            for digit, acc in sorted(per_digit_metrics.items()):
                f.write(f"  Digit {digit}: {acc:.4f}\n")
                
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(cm, classes=range(10), title='Digit Recognition Confusion Matrix')
        plt.savefig(os.path.join(save_dir, "digit_recognizer_confusion_matrix.png"))
        
        # Plot per-digit accuracies
        plt.figure(figsize=(10, 6))
        digits = sorted(per_digit_metrics.keys())
        accuracies = [per_digit_metrics[d] for d in digits]
        plt.bar([str(d) for d in digits], accuracies)
        plt.ylim(0, 1.0)
        plt.title('Digit Recognition Accuracy by Digit')
        plt.xlabel('Digit')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(save_dir, "digit_recognizer_per_digit_accuracy.png"))
        
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[int],
    normalize: bool = False,
    title: str = 'Confusion matrix',
    cmap: Any = plt.cm.Blues
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        classes: List of class labels
        normalize: Whether to normalize confusion matrix
        title: Plot title
        cmap: Color map
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.tight_layout()
    plt.ylabel('True digit')
    plt.xlabel('Predicted digit')


def evaluate_sudoku_solver(
    solver,
    test_grids: List[GridType],
    save_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate Sudoku solver performance.
    
    Args:
        solver: Sudoku solver model
        test_grids: List of test Sudoku grids
        save_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    if len(test_grids) == 0:
        raise ValueError("Empty test set")
        
    # Metrics
    success_count = 0
    total_time = 0.0
    error_types = {
        "timeout": 0,
        "invalid_puzzle": 0,
        "other": 0
    }
    
    # Process each grid
    for i, grid in enumerate(test_grids):
        try:
            # Measure solving time
            start_time = time.time()
            solved_grid = solver.solve(grid)
            solving_time = time.time() - start_time
            
            # Check if solution is valid
            success = is_valid_solution(grid, solved_grid)
            
            if success:
                success_count += 1
                total_time += solving_time
                
            # Log results
            logger.info(
                f"Grid {i+1}/{len(test_grids)}: "
                f"{'Success' if success else 'Failure'}, "
                f"Time={solving_time:.4f}s"
            )
            
        except Exception as e:
            logger.error(f"Error solving grid {i}: {str(e)}")
            
            # Categorize error
            if "timeout" in str(e).lower():
                error_types["timeout"] += 1
            elif "invalid" in str(e).lower():
                error_types["invalid_puzzle"] += 1
            else:
                error_types["other"] += 1
                
    # Calculate metrics
    success_rate = success_count / len(test_grids)
    avg_time = total_time / max(success_count, 1)
    
    # Create results dictionary
    results = {
        "success_rate": success_rate,
        "average_solving_time": avg_time,
        "timeout_errors": error_types["timeout"],
        "invalid_puzzle_errors": error_types["invalid_puzzle"],
        "other_errors": error_types["other"]
    }
    
    # Save results if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics as text file
        with open(os.path.join(save_dir, "solver_metrics.txt"), "w") as f:
            f.write("Sudoku Solver Evaluation\n")
            f.write("======================\n\n")
            f.write(f"Number of test grids: {len(test_grids)}\n")
            f.write(f"Success rate: {success_rate:.4f}\n")
            f.write(f"Average solving time: {avg_time:.4f}s\n\n")
            f.write("Error types:\n")
            f.write(f"  Timeout errors: {error_types['timeout']}\n")
            f.write(f"  Invalid puzzle errors: {error_types['invalid_puzzle']}\n")
            f.write(f"  Other errors: {error_types['other']}\n")
            
        # Plot error types
        plt.figure(figsize=(10, 6))
        labels = ['Success', 'Timeout', 'Invalid Puzzle', 'Other']
        values = [success_count, error_types['timeout'], error_types['invalid_puzzle'], error_types['other']]
        plt.bar(labels, values)
        plt.title('Solver Results')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_dir, "solver_results.png"))
        
    return results


def is_valid_solution(initial_grid: GridType, solved_grid: GridType) -> bool:
    """
    Check if a Sudoku solution is valid.
    
    Args:
        initial_grid: Initial Sudoku grid
        solved_grid: Solved Sudoku grid
        
    Returns:
        True if solution is valid
    """
    # Check if solved grid is complete
    if any(0 in row for row in solved_grid):
        return False
        
    # Check if solved grid respects initial values
    for i in range(9):
        for j in range(9):
            if initial_grid[i][j] != 0 and initial_grid[i][j] != solved_grid[i][j]:
                return False
                
    # Check rows
    for row in solved_grid:
        if len(set(row)) != 9:
            return False
            
    # Check columns
    for j in range(9):
        col = [solved_grid[i][j] for i in range(9)]
        if len(set(col)) != 9:
            return False
            
    # Check 3x3 boxes
    for box_i in range(3):
        for box_j in range(3):
            box = [
                solved_grid[box_i * 3 + i][box_j * 3 + j]
                for i in range(3)
                for j in range(3)
            ]
            if len(set(box)) != 9:
                return False
                
    return True


def evaluate_full_pipeline(
    pipeline,
    test_images: List[ImageType],
    test_grids: List[GridType],
    save_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate full Sudoku recognition pipeline.
    
    Args:
        pipeline: SudokuRecognizerPipeline instance
        test_images: List of test images
        test_grids: List of ground truth Sudoku grids
        save_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    if len(test_images) != len(test_grids):
        raise ValueError("Number of test images and grids must match")
        
    if len(test_images) == 0:
        raise ValueError("Empty test set")
        
    # Metrics
    stage_success = {
        "grid_detection": 0,
        "cell_extraction": 0,
        "digit_recognition": 0,
        "solving": 0,
        "end_to_end": 0
    }
    
    digit_accuracy_sum = 0.0
    total_time = 0.0
    
    # Process each image
    for i, (image, true_grid) in enumerate(zip(test_images, test_grids)):
        try:
            # Process image
            start_time = time.time()
            results = pipeline.process_image_data(image)
            processing_time = time.time() - start_time
            
            # Extract results
            grid_detected = results.get("grid_detected", False)
            cells_extracted = results.get("cells_extracted", False)
            digits_recognized = results.get("digits_recognized", False)
            puzzle_solved = results.get("puzzle_solved", False)
            digit_grid = results.get("digit_grid")
            solved_grid = results.get("solved_grid")
            
            # Update stage success
            if grid_detected:
                stage_success["grid_detection"] += 1
                
            if cells_extracted:
                stage_success["cell_extraction"] += 1
                
            if digits_recognized and digit_grid is not None:
                stage_success["digit_recognition"] += 1
                
                # Calculate digit recognition accuracy
                true_digits = []
                pred_digits = []
                
                for r in range(9):
                    for c in range(9):
                        if true_grid[r][c] != 0:  # Only consider non-empty cells
                            true_digits.append(true_grid[r][c])
                            pred_digits.append(digit_grid[r][c])
                            
                if true_digits:
                    accuracy = accuracy_score(true_digits, pred_digits)
                    digit_accuracy_sum += accuracy
                
            if puzzle_solved and solved_grid is not None:
                stage_success["solving"] += 1
                
                # Check if solution is correct
                if is_valid_solution(true_grid, solved_grid):
                    stage_success["end_to_end"] += 1
                    
            total_time += processing_time
            
            # Log results
            logger.info(
                f"Image {i+1}/{len(test_images)}: "
                f"Grid Detection={'Success' if grid_detected else 'Failure'}, "
                f"Cell Extraction={'Success' if cells_extracted else 'Failure'}, "
                f"Digit Recognition={'Success' if digits_recognized else 'Failure'}, "
                f"Solving={'Success' if puzzle_solved else 'Failure'}, "
                f"Time={processing_time:.4f}s"
            )
            
        except Exception as e:
            logger.error(f"Error processing image {i}: {str(e)}")
            
    # Calculate metrics
    num_images = len(test_images)
    grid_detection_rate = stage_success["grid_detection"] / num_images
    cell_extraction_rate = stage_success["cell_extraction"] / num_images
    digit_recognition_rate = stage_success["digit_recognition"] / num_images
    solving_rate = stage_success["solving"] / num_images
    end_to_end_rate = stage_success["end_to_end"] / num_images
    
    avg_digit_accuracy = digit_accuracy_sum / max(stage_success["digit_recognition"], 1)
    avg_time = total_time / num_images
    
    # Create results dictionary
    results = {
        "grid_detection_rate": grid_detection_rate,
        "cell_extraction_rate": cell_extraction_rate,
        "digit_recognition_rate": digit_recognition_rate,
        "solving_rate": solving_rate,
        "end_to_end_rate": end_to_end_rate,
        "average_digit_accuracy": avg_digit_accuracy,
        "average_processing_time": avg_time
    }
    
    # Save results if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics as text file
        with open(os.path.join(save_dir, "pipeline_metrics.txt"), "w") as f:
            f.write("Full Pipeline Evaluation\n")
            f.write("======================\n\n")
            f.write(f"Number of test images: {num_images}\n")
            f.write(f"Grid detection rate: {grid_detection_rate:.4f}\n")
            f.write(f"Cell extraction rate: {cell_extraction_rate:.4f}\n")
            f.write(f"Digit recognition rate: {digit_recognition_rate:.4f}\n")
            f.write(f"Solving rate: {solving_rate:.4f}\n")
            f.write(f"End-to-end success rate: {end_to_end_rate:.4f}\n")
            f.write(f"Average digit recognition accuracy: {avg_digit_accuracy:.4f}\n")
            f.write(f"Average processing time: {avg_time:.4f}s\n")
            
        # Plot stage success rates
        plt.figure(figsize=(12, 6))
        stages = ['Grid Detection', 'Cell Extraction', 'Digit Recognition', 'Solving', 'End-to-End']
        rates = [
            grid_detection_rate,
            cell_extraction_rate,
            digit_recognition_rate,
            solving_rate,
            end_to_end_rate
        ]
        plt.bar(stages, rates)
        plt.ylim(0, 1.0)
        plt.title('Pipeline Stage Success Rates')
        plt.ylabel('Success Rate')
        plt.savefig(os.path.join(save_dir, "pipeline_success_rates.png"))
        
    return results


def generate_evaluation_report(
    results_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        results_dict: Dictionary with evaluation results for each component
        save_path: Path to save report (optional)
        
    Returns:
        Report text
    """
    # Create report text
    report = "Sudoku Recognizer System Evaluation Report\n"
    report += "=======================================\n\n"
    
    # Add full pipeline results
    if "pipeline" in results_dict:
        pipeline_results = results_dict["pipeline"]
        report += "Full Pipeline Evaluation\n"
        report += "-----------------------\n"
        report += f"Grid detection rate: {pipeline_results.get('grid_detection_rate', 0.0):.4f}\n"
        report += f"Cell extraction rate: {pipeline_results.get('cell_extraction_rate', 0.0):.4f}\n"
        report += f"Digit recognition rate: {pipeline_results.get('digit_recognition_rate', 0.0):.4f}\n"
        report += f"Solving rate: {pipeline_results.get('solving_rate', 0.0):.4f}\n"
        report += f"End-to-end success rate: {pipeline_results.get('end_to_end_rate', 0.0):.4f}\n"
        report += f"Average digit recognition accuracy: {pipeline_results.get('average_digit_accuracy', 0.0):.4f}\n"
        report += f"Average processing time: {pipeline_results.get('average_processing_time', 0.0):.4f}s\n\n"
        
    # Add intersection detector results
    if "intersection_detector" in results_dict:
        detector_results = results_dict["intersection_detector"]
        report += "Intersection Detector Evaluation\n"
        report += "--------------------------------\n"
        report += f"Precision: {detector_results.get('precision', 0.0):.4f}\n"
        report += f"Recall: {detector_results.get('recall', 0.0):.4f}\n"
        report += f"F1 score: {detector_results.get('f1_score', 0.0):.4f}\n"
        report += f"Average detection time: {detector_results.get('average_time', 0.0):.4f}s\n\n"
        
    # Add digit recognizer results
    if "digit_recognizer" in results_dict:
        recognizer_results = results_dict["digit_recognizer"]
        report += "Digit Recognizer Evaluation\n"
        report += "---------------------------\n"
        report += f"Accuracy: {recognizer_results.get('accuracy', 0.0):.4f}\n"
        report += f"Precision: {recognizer_results.get('precision', 0.0):.4f}\n"
        report += f"Recall: {recognizer_results.get('recall', 0.0):.4f}\n"
        report += f"F1 score: {recognizer_results.get('f1_score', 0.0):.4f}\n"
        report += f"Average confidence: {recognizer_results.get('average_confidence', 0.0):.4f}\n"
        report += f"Average recognition time per grid: {recognizer_results.get('average_time_per_grid', 0.0):.4f}s\n\n"
        
        # Add per-digit accuracies
        report += "Per-digit accuracies:\n"
        for digit in range(10):
            key = f"digit_{digit}_accuracy"
            if key in recognizer_results:
                report += f"  Digit {digit}: {recognizer_results[key]:.4f}\n"
        report += "\n"
        
    # Add solver results
    if "solver" in results_dict:
        solver_results = results_dict["solver"]
        report += "Sudoku Solver Evaluation\n"
        report += "------------------------\n"
        report += f"Success rate: {solver_results.get('success_rate', 0.0):.4f}\n"
        report += f"Average solving time: {solver_results.get('average_solving_time', 0.0):.4f}s\n"
        report += f"Timeout errors: {solver_results.get('timeout_errors', 0)}\n"
        report += f"Invalid puzzle errors: {solver_results.get('invalid_puzzle_errors', 0)}\n"
        report += f"Other errors: {solver_results.get('other_errors', 0)}\n\n"
        
    # Add summary
    report += "Summary\n"
    report += "-------\n"
    
    if "pipeline" in results_dict:
        pipeline_results = results_dict["pipeline"]
        end_to_end_rate = pipeline_results.get('end_to_end_rate', 0.0)
        report += f"Overall system success rate: {end_to_end_rate:.4f}\n"
        
        # Add analysis of bottlenecks
        bottlenecks = []
        
        grid_detection_rate = pipeline_results.get('grid_detection_rate', 1.0)
        if grid_detection_rate < 0.95:
            bottlenecks.append(f"Grid detection ({grid_detection_rate:.4f})")
            
        cell_extraction_rate = pipeline_results.get('cell_extraction_rate', 1.0)
        if cell_extraction_rate < 0.95:
            bottlenecks.append(f"Cell extraction ({cell_extraction_rate:.4f})")
            
        digit_recognition_rate = pipeline_results.get('digit_recognition_rate', 1.0)
        if digit_recognition_rate < 0.95:
            bottlenecks.append(f"Digit recognition ({digit_recognition_rate:.4f})")
            
        solving_rate = pipeline_results.get('solving_rate', 1.0)
        if solving_rate < 0.95:
            bottlenecks.append(f"Solving ({solving_rate:.4f})")
            
        if bottlenecks:
            report += f"System bottlenecks: {', '.join(bottlenecks)}\n"
            
    # Save report if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report)
            
    return report
