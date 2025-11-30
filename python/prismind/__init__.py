"""
Prismind - High-performance Othello AI Engine

This package provides Python bindings for the Prismind Othello AI,
featuring pattern-based evaluation and reinforcement learning through self-play.

Example usage:
    >>> from prismind import PyEvaluator
    >>> evaluator = PyEvaluator()
    >>> board = [0] * 64  # Empty board
    >>> board[27] = 2  # White at D4
    >>> board[28] = 1  # Black at E4
    >>> board[35] = 1  # Black at D5
    >>> board[36] = 2  # White at E5
    >>> score = evaluator.evaluate(board, 1)  # Evaluate for black
"""

from prismind._prismind import (
    PyCheckpointManager,
    PyDebugModule,
    PyEvaluator,
    PyStatisticsManager,
    PyTrainingManager,
    __version__,
)

__all__ = [
    "PyEvaluator",
    "PyTrainingManager",
    "PyCheckpointManager",
    "PyStatisticsManager",
    "PyDebugModule",
    "__version__",
]
