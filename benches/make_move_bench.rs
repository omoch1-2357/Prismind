use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use prismind::board::BitBoard;
use prismind::{legal_moves, make_move, undo_move};

/// Benchmark make_move() on initial position
fn bench_make_move_initial(c: &mut Criterion) {
    let board = BitBoard::new();
    let moves = legal_moves(&board);
    let first_move = moves.trailing_zeros() as u8;

    c.bench_function("make_move_initial", |b| {
        b.iter(|| {
            let mut board_copy = black_box(board);
            black_box(make_move(&mut board_copy, black_box(first_move)))
        })
    });
}

/// Benchmark make_move() with undo_move() roundtrip
fn bench_make_undo_roundtrip(c: &mut Criterion) {
    let board = BitBoard::new();
    let moves = legal_moves(&board);
    let first_move = moves.trailing_zeros() as u8;

    c.bench_function("make_undo_roundtrip", |b| {
        b.iter(|| {
            let mut board_copy = black_box(board);
            let undo_info = make_move(&mut board_copy, black_box(first_move)).unwrap();
            undo_move(&mut board_copy, black_box(undo_info));
            black_box(board_copy)
        })
    });
}

/// Benchmark make_move() on various board states
fn bench_make_move_various_states(c: &mut Criterion) {
    let mut group = c.benchmark_group("make_move_states");

    // Initial position
    let initial = BitBoard::new();
    let moves_initial = legal_moves(&initial);
    let move_initial = moves_initial.trailing_zeros() as u8;

    group.bench_with_input(
        BenchmarkId::new("initial", "first_move"),
        &(initial, move_initial),
        |b, (board, mov)| {
            b.iter(|| {
                let mut board_copy = black_box(*board);
                black_box(make_move(&mut board_copy, black_box(*mov)))
            })
        },
    );

    // After one move (white's turn)
    let mut after_one = initial;
    make_move(&mut after_one, move_initial).unwrap();
    let moves_white = legal_moves(&after_one);
    let move_white = moves_white.trailing_zeros() as u8;

    group.bench_with_input(
        BenchmarkId::new("after_one", "white_turn"),
        &(after_one, move_white),
        |b, (board, mov)| {
            b.iter(|| {
                let mut board_copy = black_box(*board);
                black_box(make_move(&mut board_copy, black_box(*mov)))
            })
        },
    );

    group.finish();
}

/// Benchmark make_move() with 1000 iterations to measure statistics
/// Target: 1.5μs以内 (average time, standard deviation, p99 percentile)
fn bench_make_move_statistics(c: &mut Criterion) {
    let board = BitBoard::new();
    let moves = legal_moves(&board);
    let first_move = moves.trailing_zeros() as u8;

    let mut group = c.benchmark_group("make_move_stats");
    group.sample_size(1000); // Exactly 1000 iterations as specified

    group.bench_function("make_move_1000_iters", |b| {
        b.iter(|| {
            let mut board_copy = black_box(board);
            black_box(make_move(&mut board_copy, black_box(first_move)))
        })
    });

    group.finish();
}

/// Benchmark different types of moves (horizontal, vertical, diagonal flips)
fn bench_make_move_flip_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("make_move_flip_types");

    // Initial board - vertical flip (D3 position)
    let board = BitBoard::new();
    group.bench_with_input(
        BenchmarkId::new("flip", "vertical_D3"),
        &(board, 19), // D3 causes vertical flip
        |b, (board, mov)| {
            b.iter(|| {
                let mut board_copy = black_box(*board);
                black_box(make_move(&mut board_copy, black_box(*mov)))
            })
        },
    );

    // Initial board - horizontal flip (C4 position)
    group.bench_with_input(
        BenchmarkId::new("flip", "horizontal_C4"),
        &(board, 26), // C4 causes horizontal flip
        |b, (board, mov)| {
            b.iter(|| {
                let mut board_copy = black_box(*board);
                black_box(make_move(&mut board_copy, black_box(*mov)))
            })
        },
    );

    group.finish();
}

/// Benchmark comparison: make_move vs legal_moves overhead
fn bench_make_move_vs_legal_moves(c: &mut Criterion) {
    let board = BitBoard::new();
    let moves = legal_moves(&board);
    let first_move = moves.trailing_zeros() as u8;

    let mut group = c.benchmark_group("make_move_comparison");

    // Benchmark legal_moves (for comparison)
    group.bench_function("legal_moves_only", |b| {
        b.iter(|| black_box(legal_moves(black_box(&board))))
    });

    // Benchmark make_move (includes legal_moves check internally)
    group.bench_function("make_move_full", |b| {
        b.iter(|| {
            let mut board_copy = black_box(board);
            black_box(make_move(&mut board_copy, black_box(first_move)))
        })
    });

    group.finish();
}

/// Benchmark make_move error handling (illegal move detection)
fn bench_make_move_error_handling(c: &mut Criterion) {
    let board = BitBoard::new();

    let mut group = c.benchmark_group("make_move_errors");

    // Benchmark illegal move rejection
    group.bench_function("illegal_move_error", |b| {
        b.iter(|| {
            let mut board_copy = black_box(board);
            // Position 0 (A1) is not a legal move in the initial position
            black_box(make_move(&mut board_copy, black_box(0)))
        })
    });

    // Benchmark out-of-bounds error
    group.bench_function("out_of_bounds_error", |b| {
        b.iter(|| {
            let mut board_copy = black_box(board);
            black_box(make_move(&mut board_copy, black_box(64)))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_make_move_initial,
    bench_make_undo_roundtrip,
    bench_make_move_various_states,
    bench_make_move_statistics,
    bench_make_move_flip_types,
    bench_make_move_vs_legal_moves,
    bench_make_move_error_handling
);
criterion_main!(benches);
