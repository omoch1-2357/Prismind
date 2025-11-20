use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use prismind::board::BitBoard;
use prismind::legal_moves;

/// Benchmark legal_moves() on initial position
fn bench_legal_moves_initial(c: &mut Criterion) {
    let board = BitBoard::new();

    c.bench_function("legal_moves_initial", |b| {
        b.iter(|| black_box(legal_moves(black_box(&board))))
    });
}

/// Benchmark legal_moves() on various board states
fn bench_legal_moves_various_states(c: &mut Criterion) {
    let mut group = c.benchmark_group("legal_moves_states");

    // Initial position (4 legal moves)
    let initial = BitBoard::new();
    group.bench_with_input(
        BenchmarkId::new("initial", "4_moves"),
        &initial,
        |b, board| b.iter(|| black_box(legal_moves(black_box(board)))),
    );

    // Mid-game position (more complex)
    // Create a board with stones in multiple positions
    let mut midgame = BitBoard::new();
    // This is just a representative mid-game state
    // In real scenarios, this would be from actual game progression
    midgame = midgame.rotate_90(); // Simulate some game state
    group.bench_with_input(
        BenchmarkId::new("midgame", "rotated"),
        &midgame,
        |b, board| b.iter(|| black_box(legal_moves(black_box(board)))),
    );

    // Note: Creating custom board states requires public API
    // For now, we use standard positions for benchmarking
    let flipped = initial.flip();
    group.bench_with_input(
        BenchmarkId::new("flipped", "white_turn"),
        &flipped,
        |b, board| b.iter(|| black_box(legal_moves(black_box(board)))),
    );

    group.finish();
}

/// Benchmark legal_moves() with 1000 iterations to measure statistics
fn bench_legal_moves_statistics(c: &mut Criterion) {
    let board = BitBoard::new();

    let mut group = c.benchmark_group("legal_moves_stats");
    group.sample_size(1000); // Exactly 1000 iterations as specified

    group.bench_function("legal_moves_1000_iters", |b| {
        b.iter(|| black_box(legal_moves(black_box(&board))))
    });

    group.finish();
}

/// Benchmark comparison: legal_moves vs baseline operations
fn bench_legal_moves_vs_baseline(c: &mut Criterion) {
    let board = BitBoard::new();

    let mut group = c.benchmark_group("legal_moves_comparison");

    // Benchmark legal_moves
    group.bench_function("legal_moves", |b| {
        b.iter(|| black_box(legal_moves(black_box(&board))))
    });

    // Benchmark baseline: simple bit operations for comparison
    group.bench_function("baseline_bit_ops", |b| {
        b.iter(|| {
            let player = black_box(board.current_player());
            let opponent = black_box(board.opponent());
            let empty = black_box(!(player | opponent));
            black_box(player & opponent & empty)
        })
    });

    group.finish();
}

/// Benchmark legal_moves() on rotated positions (simulating different game states)
fn bench_legal_moves_rotated(c: &mut Criterion) {
    let mut group = c.benchmark_group("legal_moves_rotated");

    let board = BitBoard::new();

    // Test on 90-degree rotated board
    let rot90 = board.rotate_90();
    group.bench_with_input(
        BenchmarkId::new("rotated", "90_degrees"),
        &rot90,
        |b, board| b.iter(|| black_box(legal_moves(black_box(board)))),
    );

    // Test on 180-degree rotated board
    let rot180 = board.rotate_180();
    group.bench_with_input(
        BenchmarkId::new("rotated", "180_degrees"),
        &rot180,
        |b, board| b.iter(|| black_box(legal_moves(black_box(board)))),
    );

    // Test on 270-degree rotated board
    let rot270 = board.rotate_270();
    group.bench_with_input(
        BenchmarkId::new("rotated", "270_degrees"),
        &rot270,
        |b, board| b.iter(|| black_box(legal_moves(black_box(board)))),
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_legal_moves_initial,
    bench_legal_moves_various_states,
    bench_legal_moves_statistics,
    bench_legal_moves_vs_baseline,
    bench_legal_moves_rotated
);
criterion_main!(benches);
