#!/usr/bin/env python3
"""
monitor.py - Prismind Training Monitor Dashboard

This script provides a real-time terminal dashboard for monitoring
Prismind Othello AI training progress.

Features:
- Real-time training statistics display in formatted terminal dashboard
- Training curve visualization (stone difference and win rate over games)
- Connection to training via progress callbacks
- Refreshable statistics display

Requirements Coverage:
- Req 10.4: Real-time training statistics in formatted dashboard
- Req 10.5: Training curves showing stone difference and win rate

Usage:
    python monitor.py
    python monitor.py --checkpoint-dir checkpoints --refresh 5
    python monitor.py --plot

Author: Prismind Project
License: MIT
"""

import argparse
import contextlib
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Try to import prismind module
# This is done conditionally to allow testing of script utilities without the full library
_PRISMIND_AVAILABLE = False
PyStatisticsManager = None
PyCheckpointManager = None

try:
    from prismind import PyCheckpointManager, PyStatisticsManager

    _PRISMIND_AVAILABLE = True
except ImportError:
    # Module not available - utilities can still be tested
    pass


# ANSI color codes for terminal formatting
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"


def supports_color() -> bool:
    """Check if the terminal supports ANSI colors."""
    # Check for NO_COLOR environment variable
    if os.environ.get("NO_COLOR"):
        return False

    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    # Check for Windows
    if sys.platform == "win32":
        # Enable ANSI colors on Windows 10+
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            pass
    return False


# Disable colors if not supported
if not supports_color():
    for attr in dir(Colors):
        if not attr.startswith("_"):
            setattr(Colors, attr, "")


def clear_screen() -> None:
    """Clear the terminal screen."""
    if sys.platform == "win32":
        os.system("cls")
    else:
        print("\033[2J\033[H", end="")


def format_number(n: float, decimals: int = 0) -> str:
    """Format number with comma separators."""
    if decimals > 0:
        return f"{n:,.{decimals}f}"
    return f"{int(n):,}"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 0:
        return "N/A"

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


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value * 100:.1f}%"


def format_mb(bytes_value: float) -> str:
    """Format bytes as megabytes."""
    return f"{bytes_value:.1f} MB"


def progress_bar(value: float, width: int = 40, label: str = "") -> str:
    """
    Create a text-based progress bar.

    Args:
        value: Progress value (0.0 to 1.0)
        width: Width of the bar in characters
        label: Optional label to display

    Returns:
        Formatted progress bar string
    """
    value = max(0.0, min(1.0, value))
    filled = int(width * value)
    empty = width - filled

    bar = f"[{'=' * filled}{' ' * empty}]"

    if label:
        return f"{label}: {bar} {value * 100:5.1f}%"
    return f"{bar} {value * 100:5.1f}%"


def ascii_chart(
    data: List[Tuple[int, float]],
    width: int = 60,
    height: int = 15,
    title: str = "",
) -> str:
    """
    Create a simple ASCII line chart.

    Args:
        data: List of (x, y) tuples
        width: Chart width in characters
        height: Chart height in lines
        title: Chart title
        y_label: Y-axis label

    Returns:
        ASCII chart string
    """
    if not data:
        return "No data to plot"

    # Extract values
    x_vals = [d[0] for d in data]
    y_vals = [d[1] for d in data]

    # Calculate ranges
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    # Handle constant y values
    if y_max == y_min:
        y_min -= 1
        y_max += 1

    # Build chart
    lines = []

    # Title
    if title:
        lines.append(f"{Colors.BOLD}{title}{Colors.RESET}")
        lines.append("")

    # Chart area
    chart_width = width - 8  # Leave room for y-axis labels
    chart_height = height - 3  # Leave room for x-axis

    # Create empty chart
    chart = [[" " for _ in range(chart_width)] for _ in range(chart_height)]

    # Plot data points
    for x, y in data:
        # Normalize to chart coordinates
        if x_max > x_min:
            cx = int((x - x_min) / (x_max - x_min) * (chart_width - 1))
        else:
            cx = chart_width // 2

        cy = int((y - y_min) / (y_max - y_min) * (chart_height - 1))
        cy = chart_height - 1 - cy  # Flip y-axis

        # Bounds check
        cx = max(0, min(chart_width - 1, cx))
        cy = max(0, min(chart_height - 1, cy))

        chart[cy][cx] = "*"

    # Build output with y-axis labels
    for i, row in enumerate(chart):
        # Y-axis value
        y_val = y_max - (i / (chart_height - 1)) * (y_max - y_min)
        label = f"{y_val:6.1f} |"
        lines.append(f"{Colors.DIM}{label}{Colors.RESET}{''.join(row)}")

    # X-axis
    lines.append(f"       +{'-' * chart_width}")
    x_labels = f"        {x_min:<{chart_width // 2}}{x_max:>{chart_width // 2}}"
    lines.append(f"{Colors.DIM}{x_labels}{Colors.RESET}")

    return "\n".join(lines)


def print_header() -> None:
    """Print the dashboard header."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("=" * 70)
    print("         PRISMIND TRAINING MONITOR DASHBOARD")
    print(f"                    {timestamp}")
    print("=" * 70)
    print(Colors.RESET)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}--- {title} ---{Colors.RESET}\n")


def print_kv(key: str, value: str, width: int = 25) -> None:
    """Print a key-value pair."""
    print(f"  {Colors.DIM}{key:<{width}}{Colors.RESET}{value}")


def print_status(status: str, ok: bool = True) -> None:
    """Print a status indicator."""
    color = Colors.GREEN if ok else Colors.RED
    symbol = "[OK]" if ok else "[!!]"
    print(f"  {color}{symbol}{Colors.RESET} {status}")


def display_dashboard(
    stats_manager: Any,
    checkpoint_manager: Optional[Any] = None,
    show_charts: bool = False,
    history: Optional[List[Tuple[int, float, float]]] = None,
) -> None:
    """
    Display the training dashboard.

    Args:
        stats_manager: Statistics manager instance
        checkpoint_manager: Optional checkpoint manager for checkpoint info
        show_charts: Whether to show ASCII charts
        history: Optional training history for charts [(games, stone_diff, win_rate)]
    """
    clear_screen()
    print_header()

    # Training Statistics
    print_section("Training Statistics")
    try:
        stats = stats_manager.get_statistics()
        print_kv("Games Completed", format_number(stats.get("games_completed", 0)))
        print_kv("Elapsed Time", format_duration(stats.get("elapsed_secs", 0)))
        print_kv("Throughput", f"{stats.get('games_per_sec', 0):.2f} games/sec")
    except Exception as e:
        print(f"  {Colors.RED}Error getting statistics: {e}{Colors.RESET}")

    # Convergence Metrics
    print_section("Convergence Metrics")
    try:
        convergence = stats_manager.get_convergence_metrics()
        print_kv("Stone Diff (Avg)", f"{convergence.get('stone_diff_avg', 0):+.2f}")
        print_kv("Eval Variance", f"{convergence.get('eval_variance', 0):.4f}")
        print_kv(
            "Pattern Coverage", format_percentage(convergence.get("pattern_coverage", 0) / 100)
        )

        stagnating = convergence.get("stagnation_detected", False)
        stag_color = Colors.RED if stagnating else Colors.GREEN
        stag_text = "DETECTED" if stagnating else "None"
        print(
            f"  {Colors.DIM}{'Stagnation':<25}{Colors.RESET}{stag_color}{stag_text}{Colors.RESET}"
        )

        print_kv(
            "Games Since Improvement", format_number(convergence.get("games_since_improvement", 0))
        )
    except Exception as e:
        print(f"  {Colors.RED}Error getting convergence metrics: {e}{Colors.RESET}")

    # Pattern Coverage Details
    print_section("Pattern Update Coverage")
    try:
        coverage = stats_manager.get_pattern_coverage()
        print_kv("Unique Entries Updated", format_number(coverage.get("unique_entries_updated", 0)))
        print_kv("Total Updates", format_number(coverage.get("total_updates", 0)))
        print_kv("Coverage %", format_percentage(coverage.get("entry_coverage_pct", 0) / 100))
        print_kv("Avg Updates/Entry", f"{coverage.get('avg_updates_per_entry', 0):.2f}")

        warnings = coverage.get("warnings", [])
        if warnings:
            print(f"\n  {Colors.YELLOW}Warnings:{Colors.RESET}")
            for warning in warnings:
                print(f"    {Colors.YELLOW}! {warning}{Colors.RESET}")
    except Exception as e:
        print(f"  {Colors.RED}Error getting pattern coverage: {e}{Colors.RESET}")

    # Memory Usage
    print_section("Memory Usage")
    try:
        memory = stats_manager.get_memory_report()
        total_mb = memory.get("total_mb", 0)
        budget_mb = memory.get("budget_mb", 600)
        within_budget = memory.get("within_budget", True)

        print_kv("Pattern Tables", format_mb(memory.get("pattern_tables_mb", 0)))
        print_kv("Adam Optimizer", format_mb(memory.get("adam_state_mb", 0)))
        print_kv("Transposition Table", format_mb(memory.get("tt_mb", 0)))
        print_kv("Game History", format_mb(memory.get("game_history_mb", 0)))
        print_kv("Misc Overhead", format_mb(memory.get("misc_mb", 0)))
        print()
        print_kv("Total", format_mb(total_mb))
        print_kv("Budget", format_mb(budget_mb))

        # Memory progress bar
        usage_pct = total_mb / budget_mb if budget_mb > 0 else 0
        bar_color = Colors.GREEN if within_budget else Colors.RED
        print(f"\n  {bar_color}{progress_bar(usage_pct, 50)}{Colors.RESET}")

        status_text = "Within budget" if within_budget else "OVER BUDGET!"
        print_status(status_text, within_budget)
    except Exception as e:
        print(f"  {Colors.RED}Error getting memory report: {e}{Colors.RESET}")

    # Checkpoints
    if checkpoint_manager:
        print_section("Checkpoints")
        try:
            checkpoints = checkpoint_manager.list_checkpoints()
            print_kv("Available Checkpoints", str(len(checkpoints)))
            if checkpoints:
                latest = checkpoints[-1]
                print_kv("Latest Checkpoint", f"{latest[1]:,} games")
                print_kv("Checkpoint Size", f"{latest[3] / (1024 * 1024):.1f} MB")
        except Exception as e:
            print(f"  {Colors.RED}Error getting checkpoint info: {e}{Colors.RESET}")

    # Training Charts
    if show_charts and history and len(history) > 1:
        print_section("Training Progress Charts")

        # Stone difference chart
        stone_diff_data = [(h[0], h[1]) for h in history]
        print(
            ascii_chart(stone_diff_data, width=60, height=10, title="Stone Difference Over Games")
        )
        print()

        # Win rate chart
        win_rate_data = [(h[0], h[2] * 100) for h in history]
        print(ascii_chart(win_rate_data, width=60, height=10, title="Win Rate % Over Games"))

    # Footer
    print(f"\n{Colors.DIM}Press Ctrl+C to exit | Refresh rate: 5 seconds{Colors.RESET}")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Monitor Prismind Othello AI training progress.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic monitoring
    python monitor.py

    # Monitor with charts
    python monitor.py --plot

    # Custom refresh rate
    python monitor.py --refresh 10

    # One-shot display (no refresh)
    python monitor.py --once
        """,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoint files (default: checkpoints)",
    )

    parser.add_argument(
        "--refresh", "-r", type=int, default=5, help="Refresh interval in seconds (default: 5)"
    )

    parser.add_argument("--plot", "-p", action="store_true", help="Show training progress charts")

    parser.add_argument(
        "--once", "-o", action="store_true", help="Display once and exit (no continuous refresh)"
    )

    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    return parser


def main() -> int:
    """
    Main entry point for the monitor script.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith("_"):
                setattr(Colors, attr, "")

    try:
        # Check if prismind is available
        if not _PRISMIND_AVAILABLE:
            print("Error: prismind module not available.")
            print("Make sure the prismind library is built and installed.")
            print("Run: maturin develop --release")
            return 1

        # Create statistics manager
        assert PyStatisticsManager is not None, "prismind module not available"
        stats_manager = PyStatisticsManager()  # type: ignore[unreachable]

        # Create checkpoint manager if directory exists
        checkpoint_manager = None
        if PyCheckpointManager is not None and Path(args.checkpoint_dir).exists():
            with contextlib.suppress(Exception):
                checkpoint_manager = PyCheckpointManager(checkpoint_dir=args.checkpoint_dir)

        # Training history for charts
        history: List[Tuple[int, float, float]] = []

        if args.once:
            # One-shot display
            display_dashboard(
                stats_manager, checkpoint_manager, show_charts=args.plot, history=history
            )
            return 0

        # Continuous monitoring
        print("Starting training monitor...")
        print("Press Ctrl+C to exit.")

        while True:
            try:
                # Update history for charts
                try:
                    stats = stats_manager.get_statistics()
                    convergence = stats_manager.get_convergence_metrics()

                    games = stats.get("games_completed", 0)
                    stone_diff = convergence.get("stone_diff_avg", 0)
                    # Approximate win rate from stone diff
                    win_rate = 0.5 + (stone_diff / 64) * 0.5  # Rough approximation

                    if games > 0:
                        history.append((games, stone_diff, win_rate))

                    # Keep history bounded
                    if len(history) > 100:
                        history = history[-100:]
                except Exception:
                    pass

                # Display dashboard
                display_dashboard(
                    stats_manager,
                    checkpoint_manager,
                    show_charts=args.plot,
                    history=history if len(history) > 1 else None,
                )

                # Wait for refresh interval
                time.sleep(args.refresh)

            except KeyboardInterrupt:
                clear_screen()
                print("\nMonitor stopped.")
                return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
