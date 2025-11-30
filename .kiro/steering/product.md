# Product Overview

Prismind is a high-performance Othello (Reversi) AI engine written in Rust. It combines pattern-based evaluation with reinforcement learning to achieve competitive gameplay strength through self-play training.

## Core Capabilities

- **BitBoard Engine**: Efficient 64-bit board representation for fast move generation and game state management
- **Pattern Evaluation**: 14 positional patterns extracted across 4 rotations (56 pattern instances) for nuanced position assessment
- **Search System**: MTD(f) search with iterative deepening, transposition tables, and move ordering for optimal play
- **Self-Learning**: TD(lambda)-Leaf reinforcement learning with Adam optimization for continuous improvement through self-play

## Target Use Cases

- Training AI models for Othello through large-scale self-play (1 million+ games)
- Research platform for pattern-based evaluation and reinforcement learning techniques
- High-performance Othello engine for competitive analysis and gameplay

## Value Proposition

- **Performance-First**: Designed for ARM64 (OCI Ampere A1) with optional NEON SIMD optimizations
- **Complete Learning Pipeline**: End-to-end training system from game generation to evaluation convergence
- **Production-Ready**: Comprehensive checkpoint management, graceful shutdown, and error recovery for long-running training sessions

---
_Focus on patterns and purpose, not exhaustive feature lists_
