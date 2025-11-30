#!/usr/bin/env python3
"""
evaluate.py - Prismind Model Evaluation Script

This script evaluates trained Prismind Othello AI models by testing them
against baseline opponents: random player and simple heuristic player.

Features:
- Test against random player and report win rate
- Test against simple heuristic (corner/edge priority) and report win rate
- Accept checkpoint path as command-line argument
- Detailed statistics and analysis

Requirements Coverage:
- Req 10.6: Test trained model against random player and report win rate
- Req 10.7: Test trained model against simple heuristic and report win rate

Usage:
    python evaluate.py --checkpoint checkpoints/checkpoint_100000.bin
    python evaluate.py --checkpoint checkpoints/checkpoint_100000.bin --games 1000
    python evaluate.py --checkpoint latest

Author: Prismind Project
License: MIT
"""

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

# Try to import prismind module
# This is done conditionally to allow testing of script utilities without the full library
_PRISMIND_AVAILABLE = False
_PyEvaluator: Optional[type] = None
_PyCheckpointManager: Optional[type] = None

try:
    from prismind import PyCheckpointManager as _ImportedCheckpointManager
    from prismind import PyEvaluator as _ImportedEvaluator

    _PyEvaluator = _ImportedEvaluator
    _PyCheckpointManager = _ImportedCheckpointManager
    _PRISMIND_AVAILABLE = True
except ImportError:
    # Module not available - utilities can still be tested
    pass


# Board constants
EMPTY = 0
BLACK = 1
WHITE = 2

# Direction offsets for move calculation
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Corner positions (highly strategic)
CORNERS = {0, 7, 56, 63}

# Edge positions (excluding corners)
EDGES = {
    1,
    2,
    3,
    4,
    5,
    6,  # Top edge
    8,
    16,
    24,
    32,
    40,
    48,  # Left edge
    15,
    23,
    31,
    39,
    47,
    55,  # Right edge
    57,
    58,
    59,
    60,
    61,
    62,  # Bottom edge
}

# Positions adjacent to corners (dangerous X-squares and C-squares)
DANGER_SQUARES = {
    1,
    8,
    9,  # Near corner 0
    6,
    14,
    15,  # Near corner 7
    48,
    49,
    57,  # Near corner 56
    54,
    55,
    62,  # Near corner 63
}


@dataclass
class GameResult:
    """Result of a single game."""

    black_score: int
    white_score: int
    black_won: bool
    white_won: bool
    draw: bool
    move_count: int


@dataclass
class EvaluationResult:
    """Result of evaluation against an opponent."""

    opponent_name: str
    games_played: int
    model_wins: int
    opponent_wins: int
    draws: int
    model_win_rate: float
    avg_stone_diff: float
    avg_move_count: float


def pos_to_coords(pos: int) -> Tuple[int, int]:
    """Convert position index to row, column coordinates."""
    return pos // 8, pos % 8


def coords_to_pos(row: int, col: int) -> int:
    """Convert row, column coordinates to position index."""
    return row * 8 + col


def is_valid_pos(row: int, col: int) -> bool:
    """Check if position is on the board."""
    return 0 <= row < 8 and 0 <= col < 8


def get_opponent(player: int) -> int:
    """Get the opponent's color."""
    return WHITE if player == BLACK else BLACK


def get_legal_moves(board: List[int], player: int) -> List[int]:
    """
    Get all legal moves for a player.

    Args:
        board: 64-element board array
        player: Current player (BLACK or WHITE)

    Returns:
        List of legal move positions
    """
    opponent = get_opponent(player)
    legal_moves = []

    for pos in range(64):
        if board[pos] != EMPTY:
            continue

        # Check each direction
        for dr, dc in DIRECTIONS:
            row, col = pos_to_coords(pos)
            r, c = row + dr, col + dc
            found_opponent = False

            while is_valid_pos(r, c):
                idx = coords_to_pos(r, c)
                if board[idx] == opponent:
                    found_opponent = True
                    r += dr
                    c += dc
                elif board[idx] == player and found_opponent:
                    legal_moves.append(pos)
                    break
                else:
                    break

    return list(set(legal_moves))


def make_move(board: List[int], pos: int, player: int) -> List[int]:
    """
    Make a move on the board.

    Args:
        board: 64-element board array
        pos: Position to place piece
        player: Current player

    Returns:
        New board state after move
    """
    new_board = board.copy()
    new_board[pos] = player
    opponent = get_opponent(player)

    # Flip pieces in each direction
    for dr, dc in DIRECTIONS:
        row, col = pos_to_coords(pos)
        r, c = row + dr, col + dc
        to_flip = []

        while is_valid_pos(r, c):
            idx = coords_to_pos(r, c)
            if new_board[idx] == opponent:
                to_flip.append(idx)
                r += dr
                c += dc
            elif new_board[idx] == player:
                # Found our piece - flip all in between
                for flip_pos in to_flip:
                    new_board[flip_pos] = player
                break
            else:
                break

    return new_board


def count_pieces(board: List[int]) -> Tuple[int, int]:
    """Count black and white pieces on the board."""
    black = sum(1 for cell in board if cell == BLACK)
    white = sum(1 for cell in board if cell == WHITE)
    return black, white


def create_initial_board() -> List[int]:
    """Create the initial Othello board position."""
    board = [EMPTY] * 64
    # Standard starting position
    board[27] = WHITE  # D4
    board[28] = BLACK  # E4
    board[35] = BLACK  # D5
    board[36] = WHITE  # E5
    return board


class PlayerProtocol(Protocol):
    """Protocol for player objects."""

    def select_move(self, board: List[int], player: int) -> Optional[int]:
        """Select a move given the board state and current player."""
        ...


class EvaluatorProtocol(Protocol):
    """Protocol for evaluator objects."""

    def evaluate(self, board: List[int], player: int) -> float:
        """Evaluate a board position."""
        ...


class RandomPlayer:
    """
    Random player that selects moves uniformly at random.

    This serves as a baseline opponent to measure basic competence.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed."""
        self.rng = random.Random(seed)

    def select_move(self, board: List[int], player: int) -> Optional[int]:
        """Select a random legal move."""
        legal_moves = get_legal_moves(board, player)
        if not legal_moves:
            return None
        return self.rng.choice(legal_moves)


class HeuristicPlayer:
    """
    Simple heuristic player with corner and edge priority.

    Move selection priority:
    1. Corners (highest priority)
    2. Edges (excluding positions adjacent to empty corners)
    3. Center squares
    4. Avoid X-squares and C-squares near empty corners

    This provides a slightly stronger baseline than random play.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for tie-breaking."""
        self.rng = random.Random(seed)

    def _is_corner_empty(self, board: List[int], corner: int) -> bool:
        """Check if a corner is empty."""
        return board[corner] == EMPTY

    def _get_move_priority(self, board: List[int], pos: int) -> int:
        """
        Get priority score for a move (higher is better).

        Priority levels:
        - 1000: Corners
        - 100: Safe edges (not adjacent to empty corners)
        - 50: Other edges
        - 10: Center squares
        - 1: X-squares and C-squares near empty corners
        """
        # Corners are always best
        if pos in CORNERS:
            return 1000

        # Check if position is a danger square near empty corner
        if pos in DANGER_SQUARES:
            # Check which corner this is near
            if pos in {1, 8, 9} and self._is_corner_empty(board, 0):
                return 1
            if pos in {6, 14, 15} and self._is_corner_empty(board, 7):
                return 1
            if pos in {48, 49, 57} and self._is_corner_empty(board, 56):
                return 1
            if pos in {54, 55, 62} and self._is_corner_empty(board, 63):
                return 1

        # Edges are good (but check for nearby empty corners)
        if pos in EDGES:
            return 100

        # Center squares are okay
        return 10

    def select_move(self, board: List[int], player: int) -> Optional[int]:
        """Select move based on positional heuristics."""
        legal_moves = get_legal_moves(board, player)
        if not legal_moves:
            return None

        # Sort moves by priority
        moves_with_priority = [(pos, self._get_move_priority(board, pos)) for pos in legal_moves]
        moves_with_priority.sort(key=lambda x: x[1], reverse=True)

        # Get all moves with the highest priority
        best_priority = moves_with_priority[0][1]
        best_moves = [pos for pos, p in moves_with_priority if p == best_priority]

        # Random tie-breaking among best moves
        return self.rng.choice(best_moves)


class ModelPlayer:
    """
    Player using the trained Prismind model.

    Uses the PyEvaluator to evaluate positions and selects the move
    with the highest evaluation.
    """

    def __init__(self, evaluator: EvaluatorProtocol):
        """Initialize with a PyEvaluator instance."""
        self.evaluator = evaluator

    def select_move(self, board: List[int], player: int) -> Optional[int]:
        """Select the move with the highest evaluation."""
        legal_moves = get_legal_moves(board, player)
        if not legal_moves:
            return None

        best_move = None
        best_score = float("-inf")

        for move in legal_moves:
            # Make the move and evaluate the resulting position
            new_board = make_move(board, move, player)

            # Evaluate from the model's perspective
            # Positive scores favor black
            score = self.evaluator.evaluate(new_board, player)

            # If playing as white, negate the score
            if player == WHITE:
                score = -score

            if score > best_score:
                best_score = score
                best_move = move

        return best_move


def play_game(
    black_player: PlayerProtocol, white_player: PlayerProtocol, max_moves: int = 100
) -> GameResult:
    """
    Play a single game between two players.

    Args:
        black_player: Player object for black
        white_player: Player object for white
        max_moves: Maximum moves before game is abandoned

    Returns:
        GameResult with game outcome
    """
    board = create_initial_board()
    current_player = BLACK
    move_count = 0
    consecutive_passes = 0

    while move_count < max_moves:
        player = black_player if current_player == BLACK else white_player
        move = player.select_move(board, current_player)

        if move is None:
            consecutive_passes += 1
            if consecutive_passes >= 2:
                # Both players passed - game over
                break
        else:
            consecutive_passes = 0
            board = make_move(board, move, current_player)
            move_count += 1

        current_player = get_opponent(current_player)

    # Count final scores
    black_score, white_score = count_pieces(board)

    return GameResult(
        black_score=black_score,
        white_score=white_score,
        black_won=black_score > white_score,
        white_won=white_score > black_score,
        draw=black_score == white_score,
        move_count=move_count,
    )


def evaluate_against_opponent(
    model_player: ModelPlayer,
    opponent: PlayerProtocol,
    opponent_name: str,
    num_games: int,
    verbose: bool = False,
) -> EvaluationResult:
    """
    Evaluate the model against an opponent.

    Plays games with the model as both black and white to be fair.

    Args:
        model_player: The trained model player
        opponent: The opponent player
        opponent_name: Name of the opponent for reporting
        num_games: Total number of games to play
        model_plays_black_first: If True, model plays black in first game
        verbose: Print progress during evaluation

    Returns:
        EvaluationResult with statistics
    """
    model_wins = 0
    opponent_wins = 0
    draws = 0
    total_stone_diff = 0
    total_moves = 0

    games_as_black = num_games // 2
    games_as_white = num_games - games_as_black

    if verbose:
        print(f"\nEvaluating against {opponent_name}...")
        print(f"  Playing {games_as_black} games as Black, {games_as_white} as White")

    # Play games as black
    for i in range(games_as_black):
        result = play_game(model_player, opponent)
        total_moves += result.move_count

        if result.black_won:
            model_wins += 1
            total_stone_diff += result.black_score - result.white_score
        elif result.white_won:
            opponent_wins += 1
            total_stone_diff += result.black_score - result.white_score
        else:
            draws += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"    Black games: {i + 1}/{games_as_black}")

    # Play games as white
    for i in range(games_as_white):
        result = play_game(opponent, model_player)
        total_moves += result.move_count

        if result.white_won:
            model_wins += 1
            total_stone_diff += result.white_score - result.black_score
        elif result.black_won:
            opponent_wins += 1
            total_stone_diff += result.white_score - result.black_score
        else:
            draws += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"    White games: {i + 1}/{games_as_white}")

    return EvaluationResult(
        opponent_name=opponent_name,
        games_played=num_games,
        model_wins=model_wins,
        opponent_wins=opponent_wins,
        draws=draws,
        model_win_rate=model_wins / num_games if num_games > 0 else 0,
        avg_stone_diff=total_stone_diff / num_games if num_games > 0 else 0,
        avg_move_count=total_moves / num_games if num_games > 0 else 0,
    )


def print_evaluation_result(result: EvaluationResult) -> None:
    """Print evaluation result in a formatted manner."""
    print(f"\n{'=' * 50}")
    print(f"Results vs {result.opponent_name}")
    print(f"{'=' * 50}")
    print(f"  Games Played:     {result.games_played:,}")
    print(f"  Model Wins:       {result.model_wins:,} ({result.model_win_rate:.1%})")
    print(
        f"  Opponent Wins:    {result.opponent_wins:,} ({result.opponent_wins / result.games_played:.1%})"
    )
    print(f"  Draws:            {result.draws:,} ({result.draws / result.games_played:.1%})")
    print(f"  Avg Stone Diff:   {result.avg_stone_diff:+.1f}")
    print(f"  Avg Move Count:   {result.avg_move_count:.1f}")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Prismind model against baseline opponents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate specific checkpoint
    python evaluate.py --checkpoint checkpoints/checkpoint_100000.bin

    # Evaluate latest checkpoint
    python evaluate.py --checkpoint latest

    # Quick evaluation with 100 games
    python evaluate.py --checkpoint latest --games 100

    # Only test against random player
    python evaluate.py --checkpoint latest --random-only

    # Only test against heuristic player
    python evaluate.py --checkpoint latest --heuristic-only
        """,
    )

    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        required=True,
        help="Path to checkpoint file, or 'latest' for most recent checkpoint",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoint files when using 'latest' (default: checkpoints)",
    )

    parser.add_argument(
        "--games",
        "-g",
        type=int,
        default=500,
        help="Number of games to play against each opponent (default: 500)",
    )

    parser.add_argument(
        "--random-only", action="store_true", help="Only test against random player"
    )

    parser.add_argument(
        "--heuristic-only", action="store_true", help="Only test against heuristic player"
    )

    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed progress during evaluation"
    )

    parser.add_argument("--quiet", "-q", action="store_true", help="Only show final summary")

    return parser


def main() -> int:
    """
    Main entry point for the evaluation script.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Check if prismind is available
    if not _PRISMIND_AVAILABLE or _PyEvaluator is None or _PyCheckpointManager is None:
        print("Error: prismind module not available.")
        print("Make sure the prismind library is built and installed.")
        print("Run: maturin develop --release")
        return 1

    # Handle 'latest' checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path.lower() == "latest":
        try:
            checkpoint_manager = _PyCheckpointManager(checkpoint_dir=args.checkpoint_dir)
            checkpoints = checkpoint_manager.list_checkpoints()
            if not checkpoints:
                print(f"Error: No checkpoints found in {args.checkpoint_dir}")
                return 1
            checkpoint_path = checkpoints[-1][0]
            print(f"Using latest checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Error finding latest checkpoint: {e}")
            return 1
    elif not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return 1

    # Create evaluator with checkpoint
    if not args.quiet:
        print(f"\nLoading model from: {checkpoint_path}")

    try:
        evaluator = _PyEvaluator(checkpoint_path=checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 1

    # Create model player
    model_player = ModelPlayer(evaluator)

    # Set random seed
    seed = args.seed

    # Print header
    if not args.quiet:
        print("\n" + "=" * 60)
        print("       PRISMIND MODEL EVALUATION")
        print("=" * 60)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Games per opponent: {args.games}")
        if seed is not None:
            print(f"Random seed: {seed}")
        print("=" * 60)

    results = []
    start_time = time.time()

    # Test against random player
    if not args.heuristic_only:
        random_player = RandomPlayer(seed=seed)
        result = evaluate_against_opponent(
            model_player,
            random_player,
            "Random Player",
            args.games,
            verbose=args.verbose and not args.quiet,
        )
        results.append(result)
        if not args.quiet:
            print_evaluation_result(result)

    # Test against heuristic player
    if not args.random_only:
        heuristic_player = HeuristicPlayer(seed=seed)
        result = evaluate_against_opponent(
            model_player,
            heuristic_player,
            "Heuristic Player (Corner/Edge Priority)",
            args.games,
            verbose=args.verbose and not args.quiet,
        )
        results.append(result)
        if not args.quiet:
            print_evaluation_result(result)

    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("                    SUMMARY")
    print("=" * 60)

    for result in results:
        win_indicator = ""
        if result.model_win_rate >= 0.9:
            win_indicator = " [EXCELLENT]"
        elif result.model_win_rate >= 0.7:
            win_indicator = " [GOOD]"
        elif result.model_win_rate >= 0.5:
            win_indicator = " [PASSING]"
        else:
            win_indicator = " [NEEDS IMPROVEMENT]"

        print(f"  vs {result.opponent_name}: {result.model_win_rate:.1%}{win_indicator}")

    print(f"\nTotal evaluation time: {elapsed:.1f}s")

    # Return success if model beats random player with >50% win rate
    if results:
        random_result = next((r for r in results if "Random" in r.opponent_name), None)
        if random_result and random_result.model_win_rate < 0.5:
            print("\nWARNING: Model does not beat random player!")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
