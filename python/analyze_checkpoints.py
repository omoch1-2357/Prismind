#!/usr/bin/env python3
"""
Analyze checkpoint evaluation table statistics to check for weight divergence.
"""

import struct
import sys
from pathlib import Path


class EvalStats:
    """Evaluation table statistics."""

    def __init__(self) -> None:
        self.total_entries: int = 0
        self.min_value: int = 65535
        self.max_value: int = 0
        self.sum_value: int = 0
        self.at_center: int = 0
        self.below_center: int = 0
        self.above_center: int = 0
        self.extreme_low: int = 0
        self.extreme_high: int = 0
        self.mean_value: float = 0.0
        self.mean_stone_diff: float = 0.0
        self.sample_per_stage: list[dict[str, float]] = []


def read_checkpoint_header(path: Path) -> dict[str, int]:
    """Read checkpoint header and metadata."""
    with open(path, "rb") as f:
        # Read magic header (24 bytes)
        magic = f.read(24)
        if magic != b"PRISMIND_CHECKPOINT_V2\x00\x00":
            # Try V1 format
            f.seek(0)
            magic = f.read(24)
            if magic != b"OTHELLO_AI_CHECKPOINT_V1":
                raise ValueError(f"Unknown checkpoint format: {magic!r}")

        # Read metadata
        game_count = struct.unpack("<Q", f.read(8))[0]
        elapsed_secs = struct.unpack("<Q", f.read(8))[0]
        adam_timestep = struct.unpack("<Q", f.read(8))[0]

        return {
            "game_count": game_count,
            "elapsed_secs": elapsed_secs,
            "adam_timestep": adam_timestep,
            "header_size": f.tell(),
        }


def analyze_eval_table(path: Path) -> EvalStats:
    """Analyze evaluation table weights."""
    num_stages = 30
    center = 32768
    scale = 256.0

    stats = EvalStats()

    with open(path, "rb") as f:
        # Skip header (24 + 8 + 8 + 8 = 48 bytes) + possible extended metadata
        f.seek(0)
        magic = f.read(24)

        # Check for V2 format with extended header
        if magic == b"PRISMIND_CHECKPOINT_V2\x00\x00":
            # V2 format: header + metadata + optional extended
            f.seek(48)  # Skip to after basic metadata
            # Read extended metadata size if present
            try:
                ext_size = struct.unpack("<Q", f.read(8))[0]
                if ext_size > 0 and ext_size < 1000000:  # Sanity check
                    f.seek(56 + ext_size)  # Skip extended metadata
                else:
                    f.seek(48)  # No extended metadata
            except Exception:
                f.seek(48)
        else:
            # V1 format
            f.seek(48)

        # Pattern sizes (3^k where k is number of cells)
        # P01-P14 cell counts: 10, 10, 10, 10, 10, 8, 8, 8, 8, 7, 6, 6, 5, 4
        pattern_cell_counts = [10, 10, 10, 10, 10, 8, 8, 8, 8, 7, 6, 6, 5, 4]
        pattern_sizes = [3**k for k in pattern_cell_counts]

        # Read all entries
        for stage in range(num_stages):
            stage_values: list[int] = []
            for _pattern_id, pattern_size in enumerate(pattern_sizes):
                for _idx in range(pattern_size):
                    try:
                        data = f.read(2)
                        if len(data) < 2:
                            break
                        value = struct.unpack("<H", data)[0]

                        stats.total_entries += 1
                        stats.sum_value += value
                        stats.min_value = min(stats.min_value, value)
                        stats.max_value = max(stats.max_value, value)

                        if value == center:
                            stats.at_center += 1
                        elif value < center:
                            stats.below_center += 1
                        else:
                            stats.above_center += 1

                        # Check for extreme values (stone diff > ±64)
                        if value < 16384:  # stone diff < -64
                            stats.extreme_low += 1
                        elif value > 49152:  # stone diff > +64
                            stats.extreme_high += 1

                        stage_values.append(value)
                    except Exception:
                        break

            if stage_values:
                stats.sample_per_stage.append(
                    {
                        "stage": float(stage),
                        "count": float(len(stage_values)),
                        "min": float(min(stage_values)),
                        "max": float(max(stage_values)),
                        "mean": sum(stage_values) / len(stage_values),
                    }
                )

        if stats.total_entries > 0:
            stats.mean_value = stats.sum_value / stats.total_entries
            stats.mean_stone_diff = (stats.mean_value - center) / scale

    return stats


def main() -> int:
    """Main entry point."""
    checkpoint_dir = Path("checkpoints")

    if not checkpoint_dir.exists():
        print("Checkpoints directory not found")
        return 1

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.bin"))

    if not checkpoints:
        print("No checkpoints found")
        return 1

    print("=" * 70)
    print("CHECKPOINT EVALUATION TABLE ANALYSIS")
    print("=" * 70)

    for cp_path in checkpoints:
        print(f"\n{'='*70}")
        print(f"Checkpoint: {cp_path.name}")
        print("-" * 70)

        try:
            header = read_checkpoint_header(cp_path)
            print(f"Games: {header['game_count']:,}")
            print(f"Elapsed: {header['elapsed_secs']:,} seconds")

            stats = analyze_eval_table(cp_path)

            print("\nEvaluation Table Statistics:")
            print(f"  Total entries: {stats.total_entries:,}")
            print(f"  Min value: {stats.min_value} (stone diff: {(stats.min_value-32768)/256:.2f})")
            print(f"  Max value: {stats.max_value} (stone diff: {(stats.max_value-32768)/256:.2f})")
            print(f"  Mean value: {stats.mean_value:.1f} (stone diff: {stats.mean_stone_diff:.3f})")

            print("\nValue Distribution:")
            pct_center = 100 * stats.at_center / stats.total_entries
            pct_below = 100 * stats.below_center / stats.total_entries
            pct_above = 100 * stats.above_center / stats.total_entries
            print(f"  At center (32768): {stats.at_center:,} ({pct_center:.2f}%)")
            print(f"  Below center: {stats.below_center:,} ({pct_below:.2f}%)")
            print(f"  Above center: {stats.above_center:,} ({pct_above:.2f}%)")

            print("\nExtreme Values (|stone diff| > 64):")
            pct_low = 100 * stats.extreme_low / stats.total_entries
            pct_high = 100 * stats.extreme_high / stats.total_entries
            print(f"  Extreme low (<16384): {stats.extreme_low:,} ({pct_low:.3f}%)")
            print(f"  Extreme high (>49152): {stats.extreme_high:,} ({pct_high:.3f}%)")

            # Check for danger signs
            extreme_pct = 100 * (stats.extreme_low + stats.extreme_high) / stats.total_entries
            if extreme_pct > 5:
                print(f"\n  ⚠️  WARNING: {extreme_pct:.2f}% of entries have extreme values!")
            else:
                print(f"\n  ✓ Extreme values are within acceptable range ({extreme_pct:.3f}%)")

            # Stage breakdown (first 5 and last 5)
            if stats.sample_per_stage:
                print("\nPer-Stage Statistics (first/last 3 stages):")
                stages_to_show = stats.sample_per_stage[:3] + stats.sample_per_stage[-3:]
                for s in stages_to_show:
                    s_mean = s["mean"]
                    s_min = s["min"]
                    s_max = s["max"]
                    s_stage = int(s["stage"])
                    mean_diff = (s_mean - 32768) / 256
                    min_diff = (s_min - 32768) / 256
                    max_diff = (s_max - 32768) / 256
                    print(
                        f"  Stage {s_stage:2d}: mean={mean_diff:+.2f}, "
                        f"range=[{min_diff:+.1f}, {max_diff:+.1f}]"
                    )

        except Exception as e:
            print(f"Error analyzing checkpoint: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
