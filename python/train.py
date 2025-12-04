#!/usr/bin/env python3
"""
train.py - Prismind Othello AI Training Script

This script provides a command-line interface for training the Prismind Othello AI
using TD(lambda) reinforcement learning through self-play.

Features:
- Command-line arguments for target games, checkpoint interval, search time, epsilon schedule
- Resume from latest checkpoint with --resume flag
- Progress output to both console and log file
- Graceful shutdown on Ctrl+C with checkpoint save

Requirements Coverage:
- Req 10.1: Command-line arguments for training configuration
- Req 10.2: --resume flag to continue from latest checkpoint
- Req 10.3: Output progress to console and log file
- Req 10.8: Signal handling for graceful shutdown

Usage:
    python train.py --target-games 1000000 --checkpoint-interval 10000
    python train.py --resume
    python train.py --target-games 100000 --epsilon 0.15 --search-time 20

Author: Prismind Project
License: MIT
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import Callable, List, Optional, Protocol, Tuple, Union

# Try to import prismind module
# This is done conditionally to allow testing of script utilities without the full library
_PRISMIND_AVAILABLE = False
_PyTrainingManager: Optional[type] = None

try:
    from prismind import PyTrainingManager as _ImportedTrainingManager

    _PyTrainingManager = _ImportedTrainingManager
    _PRISMIND_AVAILABLE = True
except ImportError:
    # Module not available - utilities can still be tested
    # Main execution will fail gracefully
    pass


class TrainingManagerProtocol(Protocol):
    """Protocol for training manager objects."""

    def set_progress_callback(self, callback: "Callable[[int, float, float, float], None]") -> None:
        """Set progress callback."""
        ...

    def resume_training(self) -> None:
        """Resume from checkpoint."""
        ...

    def game_count(self) -> int:
        """Get current game count."""
        ...

    def pause_training(self) -> None:
        """Pause training."""
        ...

    def start_training(
        self,
        target_games: int,
        checkpoint_interval: int,
        callback_interval: int,
        search_time_ms: int,
        epsilon: float,
        eval_interval_games: Optional[int] = None,
        eval_sample_games: Optional[int] = None,
    ) -> "TrainingResult":
        """Start training."""
        ...


class TrainingResult(Protocol):
    """Protocol for training result."""

    games_completed: int
    final_stone_diff: float
    black_win_rate: float
    white_win_rate: float
    draw_rate: float
    games_per_second: float
    error_count: int


# Global flag for graceful shutdown
_shutdown_requested = False
_training_manager: Optional[TrainingManagerProtocol] = None


def setup_logging(log_dir: str, log_level: str = "info") -> logging.Logger:
    """
    Set up logging to both console and file.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (debug, info, warning, error)

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"training_{timestamp}.log"

    # Set log level
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    level = level_map.get(log_level.lower(), logging.INFO)

    # Create logger
    logger = logging.getLogger("prismind.train")
    logger.setLevel(level)

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging to: {log_file}")

    return logger


def signal_handler(signum: int, _frame: Optional[FrameType]) -> None:
    """
    Handle interrupt signals (Ctrl+C) for graceful shutdown.

    This handler sets a global flag that the training loop checks,
    allowing for a clean checkpoint save before exit.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global _shutdown_requested, _training_manager

    signal_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)

    if _shutdown_requested:
        # Second interrupt - force exit
        print("\nForce exit requested. Exiting immediately...")
        sys.exit(1)

    _shutdown_requested = True
    print(f"\n{signal_name} received. Requesting graceful shutdown...")
    print("Press Ctrl+C again to force exit.")

    # Try to pause training if manager is available
    if _training_manager is not None:
        try:
            _training_manager.pause_training()
        except Exception as e:
            print(f"Warning: Failed to pause training: {e}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_eta(seconds: float) -> str:
    """Format ETA in human-readable format."""
    if seconds == float("inf") or seconds < 0:
        return "N/A"
    return format_duration(seconds)


def progress_callback(games: int, stone_diff: float, win_rate: float, elapsed_secs: float) -> None:
    """
    Callback function for training progress updates.

    This function is called by PyTrainingManager at configurable intervals
    to report training progress.

    Args:
        games: Total games completed
        stone_diff: Average stone difference
        win_rate: Black win rate (0.0 to 1.0)
        elapsed_secs: Total elapsed time in seconds
    """
    # Get logger
    logger = logging.getLogger("prismind.train")

    # Calculate throughput
    games_per_sec = games / elapsed_secs if elapsed_secs > 0 else 0.0

    # Format progress message
    msg = (
        f"Games: {games:>10,} | "
        f"Stone diff: {stone_diff:+6.2f} | "
        f"Win rate: {win_rate:5.1%} | "
        f"Speed: {games_per_sec:.2f} g/s | "
        f"Elapsed: {format_duration(elapsed_secs)}"
    )

    logger.info(msg)


def parse_epsilon_schedule(
    schedule_str: str,
) -> List[Union[Tuple[int, float], Tuple[int, float, float]]]:
    """
    Parse epsilon schedule string into list of (games, epsilon) tuples.

    Format: "start:end:games" for linear decay, or "constant" for fixed value.
    Multiple phases can be separated by commas.

    Examples:
        "0.2" -> constant epsilon of 0.2
        "0.3:0.05:500000" -> decay from 0.3 to 0.05 over 500k games
        "0.3:0.1:200000,0.1:0.05:500000" -> two-phase decay

    Args:
        schedule_str: Epsilon schedule string

    Returns:
        List of (games_threshold, epsilon_value) tuples
    """
    schedule: List[Union[Tuple[int, float], Tuple[int, float, float]]] = []

    for phase in schedule_str.split(","):
        parts = phase.strip().split(":")

        if len(parts) == 1:
            # Constant epsilon
            epsilon = float(parts[0])
            schedule.append((0, epsilon))
        elif len(parts) == 3:
            # Linear decay: start:end:games
            start_eps = float(parts[0])
            end_eps = float(parts[1])
            games = int(parts[2])
            schedule.append((games, start_eps, end_eps))
        else:
            raise ValueError(f"Invalid epsilon schedule format: {phase}")

    return schedule


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Train Prismind Othello AI using TD(lambda) reinforcement learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train 1 million games with default settings
    python train.py --target-games 1000000

    # Resume from latest checkpoint
    python train.py --resume --target-games 1000000

    # Custom epsilon schedule (decay from 0.3 to 0.05)
    python train.py --target-games 1000000 --epsilon "0.3:0.05:500000"

    # Quick test run
    python train.py --target-games 1000 --checkpoint-interval 500
        """,
    )

    # Training parameters
    parser.add_argument(
        "--target-games",
        "-t",
        type=int,
        default=1_000_000,
        help="Target number of games to train (default: 1,000,000)",
    )

    parser.add_argument(
        "--checkpoint-interval",
        "-c",
        type=int,
        default=10_000,
        help="Games between checkpoint saves (default: 10,000)",
    )

    parser.add_argument(
        "--callback-interval",
        type=int,
        default=100,
        help="Games between progress callbacks (default: 100)",
    )

    parser.add_argument(
        "--search-time",
        "-s",
        type=int,
        default=15,
        help="Search time per move in milliseconds (default: 15)",
    )

    parser.add_argument(
        "--epsilon",
        "-e",
        type=str,
        default="0.1",
        help="Epsilon for exploration. Format: constant value, or 'start:end:games' for decay (default: 0.1)",
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help="Games between deterministic evaluation samples (0 disables)",
    )

    parser.add_argument(
        "--eval-games",
        type=int,
        default=32,
        help="Number of games per deterministic evaluation sample (default: 32)",
    )

    # Resume options
    parser.add_argument(
        "--resume", "-r", action="store_true", help="Resume training from latest checkpoint"
    )

    # Directory paths
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoint files (default: checkpoints)",
    )

    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory for log files (default: logs)"
    )

    parser.add_argument(
        "--pattern-file",
        type=str,
        default="patterns.csv",
        help="Path to pattern definitions file (default: patterns.csv)",
    )

    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output (only show errors)"
    )

    return parser


def main() -> int:
    """
    Main entry point for the training script.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    global _training_manager, _shutdown_requested

    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Set up logging
    log_level = "warning" if args.quiet else args.log_level
    logger = setup_logging(args.log_dir, log_level)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 60)
    logger.info("Prismind Training Session")
    logger.info("=" * 60)

    # Log configuration
    logger.info(f"Target games: {args.target_games:,}")
    logger.info(f"Checkpoint interval: {args.checkpoint_interval:,}")
    logger.info(f"Search time: {args.search_time}ms")
    logger.info(f"Epsilon: {args.epsilon}")
    if args.eval_interval > 0 and args.eval_games > 0:
        logger.info(
            f"Deterministic eval: every {args.eval_interval:,} games ({args.eval_games} games/sample)"
        )
    else:
        logger.info("Deterministic eval: disabled")
    logger.info(f"Resume mode: {args.resume}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info(f"Log dir: {args.log_dir}")

    try:
        # Check if prismind is available
        if not _PRISMIND_AVAILABLE or _PyTrainingManager is None:
            logger.error("prismind module not available.")
            logger.error("Make sure the prismind library is built and installed.")
            logger.error("Run: maturin develop --release")
            return 1

        # Create checkpoint directory
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Create training manager
        logger.info("Initializing training manager...")
        training_mgr = _PyTrainingManager(
            checkpoint_dir=args.checkpoint_dir, log_dir=args.log_dir, pattern_file=args.pattern_file
        )
        _training_manager = training_mgr

        # Set progress callback
        training_mgr.set_progress_callback(progress_callback)

        # Handle resume
        if args.resume:
            logger.info("Attempting to resume from latest checkpoint...")
            try:
                training_mgr.resume_training()
                current_games = training_mgr.game_count()
                logger.info(f"Resumed from checkpoint at {current_games:,} games")
            except RuntimeError as e:
                logger.warning(f"No checkpoint to resume from: {e}")
                logger.info("Starting fresh training session")

        # Parse epsilon schedule
        try:
            epsilon_schedule = parse_epsilon_schedule(args.epsilon)
            # For now, use the first (or constant) epsilon value
            if len(epsilon_schedule) == 1:
                epsilon = (
                    epsilon_schedule[0][1]
                    if len(epsilon_schedule[0]) == 2
                    else epsilon_schedule[0][0]
                )
            else:
                epsilon = epsilon_schedule[0][1] if len(epsilon_schedule[0]) > 1 else 0.1
        except ValueError as e:
            logger.error(f"Invalid epsilon schedule: {e}")
            return 1

        # Start training
        logger.info("Starting training...")
        start_time = time.time()

        result = training_mgr.start_training(
            target_games=args.target_games,
            checkpoint_interval=args.checkpoint_interval,
            callback_interval=args.callback_interval,
            search_time_ms=args.search_time,
            epsilon=epsilon,
            eval_interval_games=args.eval_interval,
            eval_sample_games=args.eval_games,
        )

        # Log completion
        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("Training Complete")
        logger.info("=" * 60)
        logger.info(f"Games completed: {result.games_completed:,}")
        logger.info(f"Final stone diff: {result.final_stone_diff:+.2f}")
        logger.info(f"Black win rate: {result.black_win_rate:.1%}")
        logger.info(f"White win rate: {result.white_win_rate:.1%}")
        logger.info(f"Draw rate: {result.draw_rate:.1%}")
        logger.info(f"Throughput: {result.games_per_second:.2f} games/sec")
        logger.info(f"Total time: {format_duration(elapsed)}")
        logger.info(f"Errors: {result.error_count}")

        if _shutdown_requested:
            logger.info("Training was interrupted by user request")
            return 130  # Standard exit code for SIGINT

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Full traceback:")
        return 1

    finally:
        # Cleanup
        _training_manager = None


if __name__ == "__main__":
    sys.exit(main())
