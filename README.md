# Prismind

High-performance Othello AI engine with pattern-based evaluation and reinforcement learning.

## Features

- Pattern-based board evaluation using optimized lookup tables
- TD(λ) reinforcement learning with Adam optimizer
- PyO3 Python bindings for training orchestration
- Optimized for ARM64 (OCI Ampere A1) deployment

## Installation

### From Source (Rust)

```bash
cargo build --release
```

### Python Bindings

```bash
pip install maturin
maturin develop --features pyo3
```

## Usage

### Python Training

```python
from prismind import PyEvaluator, PyTrainingManager

# Create evaluator
evaluator = PyEvaluator()

# Evaluate a board position
score = evaluator.evaluate(board, player)
```

## License

MIT
