"""
Sudoku Solver Module.

This module implements Sudoku puzzle solving using constraint propagation, 
backtracking, and other advanced techniques with robust error handling.
"""

import os
import numpy as np
import copy
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from collections import defaultdict, deque

from . import SolverBase
from config.settings import get_settings
from utils.error_handling import (
    SolverError, InvalidPuzzleError, TimeoutError, retry, fallback, robust_method
)
from utils.validation import validate_grid_values, validate_sudoku_rules

# Define types
GridType = List[List[int]]

# Configure logging
logger = logging.getLogger(__name__)


class BacktrackingSolver(SolverBase):
    """
    Backtracking-based Sudoku solver.
    
    This class implements Sudoku puzzle solving using depth-first search
    with backtracking.
    """
    
    def __init__(self):
        """Initialize backtracking solver with default parameters."""
        self.settings = get_settings().get_nested("solver")
        
        # Solver settings
        self.max_solving_time = self.settings.get("max_solving_time", 5)
        
    def load(self, model_path: str) -> bool:
        """
        Load model parameters (dummy method, no model to load).
        
        Args:
            model_path: Path to model file
            
        Returns:
            Always True
        """
        return True
        
    def save(self, model_path: str) -> bool:
        """
        Save model parameters (dummy method, no model to save).
        
        Args:
            model_path: Path to model file
            
        Returns:
            Always True
        """
        return True
    
    @robust_method(max_retries=1, timeout_sec=10.0)
    def solve(self, grid: GridType) -> GridType:
        """
        Solve a Sudoku puzzle using backtracking.
        
        Args:
            grid: 9x9 grid with initial values (0 for empty)
            
        Returns:
            Solved 9x9 grid
            
        Raises:
            SolverError: If solving fails
            InvalidPuzzleError: If puzzle is unsolvable
        """
        try:
            # Validate input grid
            validate_grid_values(grid)
            
            # Check if grid follows Sudoku rules
            try:
                validate_sudoku_rules(grid)
            except InvalidPuzzleError as e:
                logger.error(f"Invalid Sudoku puzzle: {str(e)}")
                raise
                
            # Start solving timer
            start_time = time.time()
            
            # Create a copy of the grid to avoid modifying the original
            working_grid = copy.deepcopy(grid)
            
            # Try to solve the puzzle
            if self._solve_backtracking(working_grid, start_time):
                return working_grid
            else:
                raise InvalidPuzzleError("Sudoku puzzle has no solution")
                
        except TimeoutError:
            raise SolverError("Solver timed out")
        except InvalidPuzzleError:
            raise
        except Exception as e:
            raise SolverError(f"Error solving Sudoku puzzle: {str(e)}")
    
    def _solve_backtracking(self, grid: GridType, start_time: float) -> bool:
        """
        Recursive backtracking solver.
        
        Args:
            grid: Current state of the grid
            start_time: Starting time for timeout calculation
            
        Returns:
            True if puzzle was solved, False otherwise
        """
        # Check if we've exceeded the time limit
        if time.time() - start_time > self.max_solving_time:
            raise TimeoutError(f"Solver exceeded time limit of {self.max_solving_time} seconds")
            
        # Find an empty cell
        empty_cell = self._find_empty_cell(grid)
        
        # If no empty cell, puzzle is solved
        if empty_cell is None:
            return True
            
        row, col = empty_cell
        
        # Try digits 1-9
        for digit in range(1, 10):
            # Check if digit is valid in this cell
            if self._is_valid_move(grid, row, col, digit):
                # Place digit
                grid[row][col] = digit
                
                # Recursively solve rest of the puzzle
                if self._solve_backtracking(grid, start_time):
                    return True
                    
                # If we get here, this digit didn't work
                # Backtrack and try the next digit
                grid[row][col] = 0
                
        # If we tried all digits and none worked, the puzzle is unsolvable
        return False
    
    def _find_empty_cell(self, grid: GridType) -> Optional[Tuple[int, int]]:
        """
        Find an empty cell in the grid.
        
        Args:
            grid: Current state of the grid
            
        Returns:
            Tuple of (row, col) or None if no empty cell
        """
        for row in range(9):
            for col in range(9):
                if grid[row][col] == 0:
                    return (row, col)
                    
        return None
    
    def _is_valid_move(self, grid: GridType, row: int, col: int, digit: int) -> bool:
        """
        Check if a digit can be placed in a cell.
        
        Args:
            grid: Current state of the grid
            row: Row index
            col: Column index
            digit: Digit to place
            
        Returns:
            True if move is valid
        """
        # Check row
        for i in range(9):
            if grid[row][i] == digit:
                return False
                
        # Check column
        for i in range(9):
            if grid[i][col] == digit:
                return False
                
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        
        for i in range(3):
            for j in range(3):
                if grid[box_row + i][box_col + j] == digit:
                    return False
                    
        return True


class ConstraintPropagationSolver(SolverBase):
    """
    Constraint propagation-based Sudoku solver.
    
    This class implements Sudoku puzzle solving using constraint propagation
    with the AC-3 (Arc Consistency Algorithm #3) algorithm.
    """
    
    def __init__(self):
        """Initialize constraint propagation solver with default parameters."""
        self.settings = get_settings().get_nested("solver")
        
        # Solver settings
        self.max_solving_time = self.settings.get("max_solving_time", 5)
        
    def load(self, model_path: str) -> bool:
        """
        Load model parameters (dummy method, no model to load).
        
        Args:
            model_path: Path to model file
            
        Returns:
            Always True
        """
        return True
        
    def save(self, model_path: str) -> bool:
        """
        Save model parameters (dummy method, no model to save).
        
        Args:
            model_path: Path to model file
            
        Returns:
            Always True
        """
        return True
    
    @robust_method(max_retries=1, timeout_sec=10.0)
    def solve(self, grid: GridType) -> GridType:
        """
        Solve a Sudoku puzzle using constraint propagation and search.
        
        Args:
            grid: 9x9 grid with initial values (0 for empty)
            
        Returns:
            Solved 9x9 grid
            
        Raises:
            SolverError: If solving fails
            InvalidPuzzleError: If puzzle is unsolvable
        """
        try:
            # Validate input grid
            validate_grid_values(grid)
            
            # Check if grid follows Sudoku rules
            try:
                validate_sudoku_rules(grid)
            except InvalidPuzzleError as e:
                logger.error(f"Invalid Sudoku puzzle: {str(e)}")
                raise
                
            # Start solving timer
            start_time = time.time()
            
            # Convert grid to values dictionary
            values = self._grid_to_values(grid)
            
            # Apply initial constraint propagation
            values = self._constraint_propagation(values)
            
            if values is False:
                raise InvalidPuzzleError("Sudoku puzzle has no solution (constraint propagation)")
                
            # Try to solve using search
            solution = self._search(values, start_time)
            
            if solution is False:
                raise InvalidPuzzleError("Sudoku puzzle has no solution (search)")
                
            # Convert solution back to grid
            solved_grid = self._values_to_grid(solution)
            
            return solved_grid
            
        except TimeoutError:
            raise SolverError("Solver timed out")
        except InvalidPuzzleError:
            raise
        except Exception as e:
            raise SolverError(f"Error solving Sudoku puzzle: {str(e)}")
    
    def _grid_to_values(self, grid: GridType) -> Dict[str, str]:
        """
        Convert grid to values dictionary.
        
        Args:
            grid: 9x9 grid
            
        Returns:
            Dictionary mapping cell names to possible values
        """
        values = {}
        rows = 'ABCDEFGHI'
        cols = '123456789'
        
        for r, row in enumerate(rows):
            for c, col in enumerate(cols):
                cell = row + col
                digit = grid[r][c]
                if digit == 0:
                    values[cell] = '123456789'  # All digits possible
                else:
                    values[cell] = str(digit)  # Only this digit
                    
        return values
    
    def _values_to_grid(self, values: Dict[str, str]) -> GridType:
        """
        Convert values dictionary to grid.
        
        Args:
            values: Dictionary mapping cell names to values
            
        Returns:
            9x9 grid
        """
        grid = [[0 for _ in range(9)] for _ in range(9)]
        rows = 'ABCDEFGHI'
        cols = '123456789'
        
        for r, row in enumerate(rows):
            for c, col in enumerate(cols):
                cell = row + col
                value = values[cell]
                
                if len(value) == 1:
                    grid[r][c] = int(value)
                else:
                    grid[r][c] = 0  # Not solved yet
                    
        return grid
    
    def _constraint_propagation(self, values: Dict[str, str]) -> Union[Dict[str, str], bool]:
        """
        Apply constraint propagation using AC-3 algorithm.
        
        Args:
            values: Dictionary mapping cell names to possible values
            
        Returns:
            Updated values dictionary or False if no solution
        """
        # Define all cell names
        rows = 'ABCDEFGHI'
        cols = '123456789'
        
        # Define units and peers
        unitlist = (
            # Rows
            [r + c for c in cols] for r in rows
        ) + (
            # Columns
            [r + c for r in rows] for c in cols
        ) + (
            # Boxes
            [r + c for r in box_r for c in box_c]
            for box_r in ('ABC', 'DEF', 'GHI')
            for box_c in ('123', '456', '789')
        )
        
        units = {
            cell: [unit for unit in unitlist if cell in unit]
            for cell in [r + c for r in rows for c in cols]
        }
        
        peers = {
            cell: set(sum(units[cell], [])) - {cell}
            for cell in [r + c for r in rows for c in cols]
        }
        
        # Apply AC-3 algorithm
        queue = deque([(cell, peer) for cell in values for peer in peers[cell]])
        
        while queue:
            cell1, cell2 = queue.popleft()
            
            if self._revise(values, cell1, cell2):
                if len(values[cell1]) == 0:
                    return False  # No solution
                    
                # Add new constraints to queue
                for peer in peers[cell1] - {cell2}:
                    queue.append((peer, cell1))
                    
        return values
    
    def _revise(self, values: Dict[str, str], cell1: str, cell2: str) -> bool:
        """
        Revise the domain of cell1 with respect to cell2.
        
        Args:
            values: Dictionary mapping cell names to possible values
            cell1: First cell
            cell2: Second cell
            
        Returns:
            True if values changed
        """
        revised = False
        
        if len(values[cell2]) == 1:
            # Remove value of cell2 from domain of cell1
            digit = values[cell2]
            if digit in values[cell1] and len(values[cell1]) > 1:
                values[cell1] = values[cell1].replace(digit, '')
                revised = True
                
        return revised
    
    def _search(self, values: Dict[str, str], start_time: float) -> Union[Dict[str, str], bool]:
        """
        Use depth-first search and constraint propagation to solve the puzzle.
        
        Args:
            values: Dictionary mapping cell names to possible values
            start_time: Starting time for timeout calculation
            
        Returns:
            Solution dictionary or False if no solution
        """
        # Check if we've exceeded the time limit
        if time.time() - start_time > self.max_solving_time:
            raise TimeoutError(f"Solver exceeded time limit of {self.max_solving_time} seconds")
            
        # Check if the puzzle is solved
        if all(len(values[cell]) == 1 for cell in values):
            return values
            
        # Choose the cell with the fewest possibilities
        cell = min(
            (cell for cell in values if len(values[cell]) > 1),
            key=lambda cell: len(values[cell])
        )
        
        # Try each possible value
        for digit in values[cell]:
            new_values = values.copy()
            new_values[cell] = digit
            
            # Apply constraint propagation
            result = self._constraint_propagation(new_values)
            
            if result is not False:
                # Recursively search
                solution = self._search(result, start_time)
                
                if solution is not False:
                    return solution
                    
        return False  # No solution found


class RobustSolver(SolverBase):
    """
    Robust Sudoku solver with multiple methods and fallback mechanisms.
    
    This class combines constraint propagation and backtracking solvers for
    robustness, with intelligent method selection and fallback strategies.
    """
    
    def __init__(self):
        """Initialize robust solver with multiple methods."""
        self.settings = get_settings().get_nested("solver")
        
        # Initialize solvers
        self.constraint_solver = ConstraintPropagationSolver()
        self.backtracking_solver = BacktrackingSolver()
        
        # Settings
        self.use_constraint_propagation = self.settings.get("use_constraint_propagation", True)
        self.use_backtracking = self.settings.get("use_backtracking", True)
        self.max_solving_time = self.settings.get("max_solving_time", 5)
        self.use_multiple_solvers = self.settings.get("use_multiple_solvers", True)
        self.validate_solution = self.settings.get("validate_solution", True)
        self.fallback_to_simpler_solver = self.settings.get("fallback_to_simpler_solver", True)
        
    def load(self, model_path: str) -> bool:
        """
        Load model parameters (dummy method, no model to load).
        
        Args:
            model_path: Path to model file
            
        Returns:
            Always True
        """
        return True
        
    def save(self, model_path: str) -> bool:
        """
        Save model parameters (dummy method, no model to save).
        
        Args:
            model_path: Path to model file
            
        Returns:
            Always True
        """
        return True
    
    @robust_method(max_retries=2, timeout_sec=15.0)
    def solve(self, grid: GridType) -> GridType:
        """
        Solve a Sudoku puzzle using multiple methods with fallback.
        
        Args:
            grid: 9x9 grid with initial values (0 for empty)
            
        Returns:
            Solved 9x9 grid
            
        Raises:
            SolverError: If all solving methods fail
            InvalidPuzzleError: If puzzle is unsolvable
        """
        try:
            # Validate input grid
            validate_grid_values(grid)
            
            # Check if grid follows Sudoku rules
            try:
                validate_sudoku_rules(grid)
            except InvalidPuzzleError as e:
                logger.error(f"Invalid Sudoku puzzle: {str(e)}")
                raise
                
            # Start solving timer
            start_time = time.time()
            
            # If the puzzle is already solved, return it
            if self._is_solved(grid):
                return grid
                
            # If puzzle is empty or nearly empty, use backtracking directly
            if self._is_very_sparse(grid):
                logger.info("Puzzle is very sparse, using backtracking solver directly")
                return self.backtracking_solver.solve(grid)
                
            # Try constraint propagation first (faster for most puzzles)
            if self.use_constraint_propagation:
                try:
                    logger.info("Trying constraint propagation solver")
                    solution = self.constraint_solver.solve(grid)
                    
                    # Validate solution if required
                    if self.validate_solution and not self._is_valid_solution(grid, solution):
                        logger.warning("Invalid solution from constraint propagation solver")
                        raise SolverError("Invalid solution")
                        
                    logger.info("Puzzle solved with constraint propagation")
                    return solution
                except (SolverError, InvalidPuzzleError) as e:
                    logger.warning(f"Constraint propagation solver failed: {str(e)}")
                    
                    # Check if time limit exceeded
                    if time.time() - start_time > self.max_solving_time:
                        raise SolverError("Solving time limit exceeded")
                    
            # Fallback to backtracking
            if self.use_backtracking:
                try:
                    logger.info("Trying backtracking solver")
                    solution = self.backtracking_solver.solve(grid)
                    
                    # Validate solution if required
                    if self.validate_solution and not self._is_valid_solution(grid, solution):
                        logger.warning("Invalid solution from backtracking solver")
                        raise SolverError("Invalid solution")
                        
                    logger.info("Puzzle solved with backtracking")
                    return solution
                except (SolverError, InvalidPuzzleError) as e:
                    logger.warning(f"Backtracking solver failed: {str(e)}")
                    
            # If all methods failed, try simple guess-based fallback
            if self.fallback_to_simpler_solver:
                logger.warning("All solvers failed, trying simple fallback solver")
                
                try:
                    solution = self._fallback_solve(grid)
                    
                    if solution is not None:
                        logger.info("Puzzle solved with fallback solver")
                        return solution
                except Exception as e:
                    logger.error(f"Fallback solver failed: {str(e)}")
                    
            # If we get here, all methods failed
            raise InvalidPuzzleError("Failed to solve Sudoku puzzle with all methods")
            
        except TimeoutError:
            raise SolverError("Solver timed out")
        except InvalidPuzzleError:
            raise
        except Exception as e:
            raise SolverError(f"Error solving Sudoku puzzle: {str(e)}")
    
    def _is_solved(self, grid: GridType) -> bool:
        """
        Check if a grid is already solved.
        
        Args:
            grid: 9x9 grid
            
        Returns:
            True if grid is solved
        """
        # Check if all cells are filled
        for row in grid:
            if 0 in row:
                return False
                
        # Check if grid is valid
        try:
            validate_sudoku_rules(grid)
            return True
        except InvalidPuzzleError:
            return False
    
    def _is_very_sparse(self, grid: GridType) -> bool:
        """
        Check if a grid is very sparse (few filled cells).
        
        Args:
            grid: 9x9 grid
            
        Returns:
            True if grid is very sparse
        """
        filled_cells = sum(1 for row in grid for cell in row if cell != 0)
        return filled_cells < 17  # Empirical threshold
    
    def _is_valid_solution(self, original_grid: GridType, solution_grid: GridType) -> bool:
        """
        Validate that a solution is correct.
        
        Args:
            original_grid: Original unsolved grid
            solution_grid: Proposed solution grid
            
        Returns:
            True if solution is valid
        """
        # Check if solution is complete
        for row in solution_grid:
            if 0 in row:
                return False
                
        # Check if solution is valid according to Sudoku rules
        try:
            validate_sudoku_rules(solution_grid)
        except InvalidPuzzleError:
            return False
            
        # Check if solution respects original filled cells
        for i in range(9):
            for j in range(9):
                if original_grid[i][j] != 0 and original_grid[i][j] != solution_grid[i][j]:
                    return False
                    
        return True
    
    def _fallback_solve(self, grid: GridType) -> Optional[GridType]:
        """
        Simple fallback solver for desperate cases.
        
        Args:
            grid: 9x9 grid with initial values
            
        Returns:
            Solved grid or None if unsolvable
        """
        # Create a working copy
        working_grid = copy.deepcopy(grid)
        
        # Find all empty cells
        empty_cells = []
        for i in range(9):
            for j in range(9):
                if working_grid[i][j] == 0:
                    empty_cells.append((i, j))
                    
        # Get possible values for each empty cell
        possible_values = {}
        for row, col in empty_cells:
            values = self._get_possible_values(working_grid, row, col)
            possible_values[(row, col)] = values
            
            # If any cell has no possible values, puzzle is unsolvable
            if not values:
                return None
                
        # Sort empty cells by number of possible values
        empty_cells.sort(key=lambda cell: len(possible_values[cell]))
        
        # Try to fill cells with only one possible value
        progress = True
        while progress:
            progress = False
            
            for i, (row, col) in enumerate(empty_cells):
                if working_grid[row][col] == 0:
                    values = possible_values[(row, col)]
                    
                    if len(values) == 1:
                        # Fill cell with only possible value
                        working_grid[row][col] = list(values)[0]
                        progress = True
                        
                        # Update possible values for affected cells
                        for r, c in empty_cells:
                            if working_grid[r][c] == 0 and ((r == row) or (c == col) or 
                                (r // 3 == row // 3 and c // 3 == col // 3)):
                                possible_values[(r, c)] -= {working_grid[row][col]}
                                
        # If grid is still not solved, give up
        if any(working_grid[row][col] == 0 for row, col in empty_cells):
            return None
            
        return working_grid
    
    def _get_possible_values(self, grid: GridType, row: int, col: int) -> Set[int]:
        """
        Get possible values for a cell.
        
        Args:
            grid: Current state of the grid
            row: Row index
            col: Column index
            
        Returns:
            Set of possible values
        """
        # Start with all digits
        values = set(range(1, 10))
        
        # Remove digits in the same row
        values -= {grid[row][c] for c in range(9) if grid[row][c] != 0}
        
        # Remove digits in the same column
        values -= {grid[r][col] for r in range(9) if grid[r][col] != 0}
        
        # Remove digits in the same 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        values -= {
            grid[box_row + r][box_col + c]
            for r in range(3) for c in range(3)
            if grid[box_row + r][box_col + c] != 0
        }
        
        return values
