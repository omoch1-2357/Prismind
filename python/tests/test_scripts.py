#!/usr/bin/env python3
"""
Test suite for Prismind Python training scripts.

This module tests the Python scripts (train.py, monitor.py, evaluate.py)
to verify they meet requirements.

Requirements Coverage:
- Req 10.1-10.3, 10.8: train.py tests
- Req 10.4-10.5: monitor.py tests
- Req 10.6-10.7: evaluate.py tests
"""

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import script modules
try:
    import train
    import monitor
    import evaluate
except ImportError:
    # Create mock modules if prismind is not available
    pass


class TestTrainScript(unittest.TestCase):
    """Tests for train.py script."""

    def test_argument_parser_creation(self):
        """Test that argument parser is created with all required arguments."""
        parser = train.create_argument_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_argument_parser_target_games(self):
        """Test --target-games argument (Req 10.1)."""
        parser = train.create_argument_parser()
        args = parser.parse_args(["--target-games", "100000"])
        self.assertEqual(args.target_games, 100000)

    def test_argument_parser_checkpoint_interval(self):
        """Test --checkpoint-interval argument (Req 10.1)."""
        parser = train.create_argument_parser()
        args = parser.parse_args(["--checkpoint-interval", "5000"])
        self.assertEqual(args.checkpoint_interval, 5000)

    def test_argument_parser_search_time(self):
        """Test --search-time argument (Req 10.1)."""
        parser = train.create_argument_parser()
        args = parser.parse_args(["--search-time", "20"])
        self.assertEqual(args.search_time, 20)

    def test_argument_parser_epsilon(self):
        """Test --epsilon argument (Req 10.1)."""
        parser = train.create_argument_parser()
        args = parser.parse_args(["--epsilon", "0.15"])
        self.assertEqual(args.epsilon, "0.15")

    def test_argument_parser_resume_flag(self):
        """Test --resume flag (Req 10.2)."""
        parser = train.create_argument_parser()
        args = parser.parse_args(["--resume"])
        self.assertTrue(args.resume)

    def test_argument_parser_default_resume_false(self):
        """Test --resume flag defaults to False (Req 10.2)."""
        parser = train.create_argument_parser()
        args = parser.parse_args([])
        self.assertFalse(args.resume)

    def test_argument_parser_log_dir(self):
        """Test --log-dir argument (Req 10.3)."""
        parser = train.create_argument_parser()
        args = parser.parse_args(["--log-dir", "custom_logs"])
        self.assertEqual(args.log_dir, "custom_logs")

    def test_epsilon_schedule_constant(self):
        """Test constant epsilon schedule parsing."""
        schedule = train.parse_epsilon_schedule("0.1")
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0][1], 0.1)

    def test_epsilon_schedule_decay(self):
        """Test decay epsilon schedule parsing."""
        schedule = train.parse_epsilon_schedule("0.3:0.05:500000")
        self.assertEqual(len(schedule), 1)
        # Decay format: (games, start, end)
        self.assertEqual(schedule[0][0], 500000)
        self.assertEqual(schedule[0][1], 0.3)
        self.assertEqual(schedule[0][2], 0.05)

    def test_format_duration_seconds(self):
        """Test duration formatting for seconds."""
        result = train.format_duration(45)
        self.assertIn("s", result)

    def test_format_duration_minutes(self):
        """Test duration formatting for minutes."""
        result = train.format_duration(125)
        self.assertIn("m", result)

    def test_format_duration_hours(self):
        """Test duration formatting for hours."""
        result = train.format_duration(3700)
        self.assertIn("h", result)

    def test_format_duration_days(self):
        """Test duration formatting for days."""
        result = train.format_duration(90000)
        self.assertIn("d", result)


class TestMonitorScript(unittest.TestCase):
    """Tests for monitor.py script."""

    def test_argument_parser_creation(self):
        """Test that argument parser is created."""
        parser = monitor.create_argument_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_argument_parser_refresh(self):
        """Test --refresh argument."""
        parser = monitor.create_argument_parser()
        args = parser.parse_args(["--refresh", "10"])
        self.assertEqual(args.refresh, 10)

    def test_argument_parser_plot_flag(self):
        """Test --plot flag (Req 10.5)."""
        parser = monitor.create_argument_parser()
        args = parser.parse_args(["--plot"])
        self.assertTrue(args.plot)

    def test_argument_parser_once_flag(self):
        """Test --once flag for single display."""
        parser = monitor.create_argument_parser()
        args = parser.parse_args(["--once"])
        self.assertTrue(args.once)

    def test_format_number(self):
        """Test number formatting with commas."""
        result = monitor.format_number(1234567)
        self.assertEqual(result, "1,234,567")

    def test_format_number_decimals(self):
        """Test number formatting with decimals."""
        result = monitor.format_number(1234.567, decimals=2)
        self.assertEqual(result, "1,234.57")

    def test_format_duration(self):
        """Test duration formatting."""
        result = monitor.format_duration(3661)
        self.assertIn("h", result)
        self.assertIn("m", result)

    def test_format_percentage(self):
        """Test percentage formatting."""
        result = monitor.format_percentage(0.752)
        self.assertEqual(result, "75.2%")

    def test_format_mb(self):
        """Test megabyte formatting."""
        result = monitor.format_mb(256.5)
        self.assertEqual(result, "256.5 MB")

    def test_progress_bar_empty(self):
        """Test progress bar at 0%."""
        result = monitor.progress_bar(0.0, width=10)
        self.assertIn("[", result)
        self.assertIn("]", result)
        self.assertIn("0.0%", result)

    def test_progress_bar_full(self):
        """Test progress bar at 100%."""
        result = monitor.progress_bar(1.0, width=10)
        self.assertIn("100.0%", result)

    def test_progress_bar_half(self):
        """Test progress bar at 50%."""
        result = monitor.progress_bar(0.5, width=10)
        self.assertIn("50.0%", result)

    def test_progress_bar_clamped(self):
        """Test progress bar value clamping."""
        result = monitor.progress_bar(1.5, width=10)
        self.assertIn("100.0%", result)

        result = monitor.progress_bar(-0.5, width=10)
        self.assertIn("0.0%", result)

    def test_ascii_chart_empty_data(self):
        """Test ASCII chart with no data."""
        result = monitor.ascii_chart([])
        self.assertIn("No data", result)

    def test_ascii_chart_single_point(self):
        """Test ASCII chart with single data point."""
        result = monitor.ascii_chart([(0, 1.0)])
        self.assertIsInstance(result, str)

    def test_ascii_chart_multiple_points(self):
        """Test ASCII chart with multiple data points."""
        data = [(i, i * 0.1) for i in range(10)]
        result = monitor.ascii_chart(data, width=40, height=10)
        self.assertIsInstance(result, str)
        # Should contain at least one data point marker
        self.assertIn("*", result)


class TestEvaluateScript(unittest.TestCase):
    """Tests for evaluate.py script."""

    def test_argument_parser_creation(self):
        """Test that argument parser is created."""
        parser = evaluate.create_argument_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_argument_parser_checkpoint_required(self):
        """Test --checkpoint is required (Req 10.6, 10.7)."""
        parser = evaluate.create_argument_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_argument_parser_checkpoint(self):
        """Test --checkpoint argument (Req 10.6, 10.7)."""
        parser = evaluate.create_argument_parser()
        args = parser.parse_args(["--checkpoint", "test.bin"])
        self.assertEqual(args.checkpoint, "test.bin")

    def test_argument_parser_games(self):
        """Test --games argument."""
        parser = evaluate.create_argument_parser()
        args = parser.parse_args(["--checkpoint", "test.bin", "--games", "1000"])
        self.assertEqual(args.games, 1000)

    def test_argument_parser_random_only(self):
        """Test --random-only flag (Req 10.6)."""
        parser = evaluate.create_argument_parser()
        args = parser.parse_args(["--checkpoint", "test.bin", "--random-only"])
        self.assertTrue(args.random_only)

    def test_argument_parser_heuristic_only(self):
        """Test --heuristic-only flag (Req 10.7)."""
        parser = evaluate.create_argument_parser()
        args = parser.parse_args(["--checkpoint", "test.bin", "--heuristic-only"])
        self.assertTrue(args.heuristic_only)

    def test_initial_board(self):
        """Test initial board creation."""
        board = evaluate.create_initial_board()
        self.assertEqual(len(board), 64)
        # Check initial position
        self.assertEqual(board[27], evaluate.WHITE)  # D4
        self.assertEqual(board[28], evaluate.BLACK)  # E4
        self.assertEqual(board[35], evaluate.BLACK)  # D5
        self.assertEqual(board[36], evaluate.WHITE)  # E5
        # Count empty squares
        empty = sum(1 for cell in board if cell == evaluate.EMPTY)
        self.assertEqual(empty, 60)

    def test_pos_to_coords(self):
        """Test position to coordinates conversion."""
        # A1 = position 0
        self.assertEqual(evaluate.pos_to_coords(0), (0, 0))
        # H1 = position 7
        self.assertEqual(evaluate.pos_to_coords(7), (0, 7))
        # A8 = position 56
        self.assertEqual(evaluate.pos_to_coords(56), (7, 0))
        # H8 = position 63
        self.assertEqual(evaluate.pos_to_coords(63), (7, 7))

    def test_coords_to_pos(self):
        """Test coordinates to position conversion."""
        self.assertEqual(evaluate.coords_to_pos(0, 0), 0)
        self.assertEqual(evaluate.coords_to_pos(0, 7), 7)
        self.assertEqual(evaluate.coords_to_pos(7, 0), 56)
        self.assertEqual(evaluate.coords_to_pos(7, 7), 63)

    def test_get_opponent(self):
        """Test getting opponent color."""
        self.assertEqual(evaluate.get_opponent(evaluate.BLACK), evaluate.WHITE)
        self.assertEqual(evaluate.get_opponent(evaluate.WHITE), evaluate.BLACK)

    def test_get_legal_moves_initial(self):
        """Test legal moves from initial position."""
        board = evaluate.create_initial_board()
        # Black moves first
        moves = evaluate.get_legal_moves(board, evaluate.BLACK)
        # Initial position should have 4 legal moves for black
        self.assertEqual(len(moves), 4)
        # D3, C4, F5, E6 are the legal moves
        expected = {19, 26, 37, 44}
        self.assertEqual(set(moves), expected)

    def test_get_legal_moves_no_moves(self):
        """Test when no legal moves are available."""
        # Create a board where player has no moves
        board = [evaluate.EMPTY] * 64
        board[0] = evaluate.BLACK
        # No opponent pieces to flip
        moves = evaluate.get_legal_moves(board, evaluate.BLACK)
        self.assertEqual(len(moves), 0)

    def test_make_move(self):
        """Test making a move on the board."""
        board = evaluate.create_initial_board()
        # Black plays D3 (position 19)
        new_board = evaluate.make_move(board, 19, evaluate.BLACK)
        # Original board should be unchanged
        self.assertEqual(board[19], evaluate.EMPTY)
        # New board should have the piece
        self.assertEqual(new_board[19], evaluate.BLACK)
        # E4 should be flipped from black to black (wait, D4 is white at 27)
        # D3 flips D4
        self.assertEqual(new_board[27], evaluate.BLACK)

    def test_count_pieces(self):
        """Test counting pieces on the board."""
        board = evaluate.create_initial_board()
        black, white = evaluate.count_pieces(board)
        self.assertEqual(black, 2)
        self.assertEqual(white, 2)

    def test_random_player_select_move(self):
        """Test random player move selection (Req 10.6)."""
        player = evaluate.RandomPlayer(seed=42)
        board = evaluate.create_initial_board()
        move = player.select_move(board, evaluate.BLACK)
        # Should select one of the legal moves
        legal_moves = evaluate.get_legal_moves(board, evaluate.BLACK)
        self.assertIn(move, legal_moves)

    def test_random_player_no_moves(self):
        """Test random player with no legal moves."""
        player = evaluate.RandomPlayer(seed=42)
        # Create board with no moves
        board = [evaluate.EMPTY] * 64
        board[0] = evaluate.BLACK
        move = player.select_move(board, evaluate.BLACK)
        self.assertIsNone(move)

    def test_heuristic_player_select_move(self):
        """Test heuristic player move selection (Req 10.7)."""
        player = evaluate.HeuristicPlayer(seed=42)
        board = evaluate.create_initial_board()
        move = player.select_move(board, evaluate.BLACK)
        # Should select one of the legal moves
        legal_moves = evaluate.get_legal_moves(board, evaluate.BLACK)
        self.assertIn(move, legal_moves)

    def test_heuristic_player_prefers_corners(self):
        """Test heuristic player corner preference (Req 10.7)."""
        player = evaluate.HeuristicPlayer(seed=42)
        # Create a board where a corner is available
        board = [evaluate.EMPTY] * 64
        # Set up so corner 0 is a legal move
        board[1] = evaluate.WHITE
        board[2] = evaluate.BLACK
        board[9] = evaluate.WHITE
        board[18] = evaluate.BLACK

        # If corner is in legal moves, heuristic should choose it
        legal_moves = evaluate.get_legal_moves(board, evaluate.BLACK)
        if 0 in legal_moves:
            move = player.select_move(board, evaluate.BLACK)
            self.assertEqual(move, 0)

    def test_play_game_completes(self):
        """Test that play_game completes and returns result."""
        black_player = evaluate.RandomPlayer(seed=42)
        white_player = evaluate.RandomPlayer(seed=43)
        result = evaluate.play_game(black_player, white_player)
        self.assertIsInstance(result, evaluate.GameResult)
        self.assertGreater(result.black_score + result.white_score, 0)
        self.assertGreater(result.move_count, 0)
        # Exactly one of these should be true
        self.assertEqual(
            result.black_won + result.white_won + result.draw,
            1
        )

    def test_evaluation_result_dataclass(self):
        """Test EvaluationResult dataclass."""
        result = evaluate.EvaluationResult(
            opponent_name="Test",
            games_played=100,
            model_wins=60,
            opponent_wins=35,
            draws=5,
            model_win_rate=0.6,
            avg_stone_diff=5.2,
            avg_move_count=50.0
        )
        self.assertEqual(result.opponent_name, "Test")
        self.assertEqual(result.games_played, 100)
        self.assertEqual(result.model_win_rate, 0.6)


class TestSignalHandling(unittest.TestCase):
    """Tests for signal handling in train.py (Req 10.8)."""

    def test_signal_handler_sets_flag(self):
        """Test that signal handler sets shutdown flag."""
        train._shutdown_requested = False
        train._training_manager = None
        # Simulate signal
        train.signal_handler(2, None)  # SIGINT
        self.assertTrue(train._shutdown_requested)
        # Reset for other tests
        train._shutdown_requested = False

    def test_signal_handler_double_interrupt(self):
        """Test that double interrupt triggers force exit."""
        train._shutdown_requested = True
        train._training_manager = None
        with self.assertRaises(SystemExit):
            train.signal_handler(2, None)
        # Reset
        train._shutdown_requested = False


class TestLogging(unittest.TestCase):
    """Tests for logging functionality in train.py (Req 10.3)."""

    def _cleanup_logger(self, logger):
        """Close all handlers on the logger to release file handles."""
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    def test_setup_logging_creates_logger(self):
        """Test that setup_logging creates a logger."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            logger = train.setup_logging(tmpdir, "info")
            try:
                self.assertIsNotNone(logger)
                self.assertEqual(logger.name, "prismind.train")
            finally:
                self._cleanup_logger(logger)

    def test_setup_logging_creates_log_directory(self):
        """Test that setup_logging creates the log directory."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            log_dir = os.path.join(tmpdir, "new_logs")
            logger = train.setup_logging(log_dir, "info")
            try:
                self.assertTrue(os.path.exists(log_dir))
            finally:
                self._cleanup_logger(logger)

    def test_setup_logging_creates_log_file(self):
        """Test that setup_logging creates a log file."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            logger = train.setup_logging(tmpdir, "info")
            try:
                log_files = list(Path(tmpdir).glob("training_*.log"))
                self.assertEqual(len(log_files), 1)
            finally:
                self._cleanup_logger(logger)


if __name__ == "__main__":
    unittest.main()
