use criterion::{Criterion, criterion_group, criterion_main};
use prismind::board::BitBoard;
use prismind::pattern::{Pattern, extract_all_patterns};
use std::hint::black_box;

/// ベンチマーク用のテストパターンを作成
fn create_benchmark_patterns() -> Vec<Pattern> {
    vec![
        Pattern::new(0, 10, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap(),
        Pattern::new(1, 10, vec![0, 8, 16, 24, 32, 40, 48, 56, 1, 9]).unwrap(),
        Pattern::new(2, 10, vec![0, 1, 8, 9, 16, 17, 24, 25, 32, 33]).unwrap(),
        Pattern::new(3, 10, vec![0, 9, 18, 27, 36, 45, 54, 63, 1, 10]).unwrap(),
        Pattern::new(4, 8, vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap(),
        Pattern::new(5, 8, vec![0, 8, 16, 24, 32, 40, 48, 56]).unwrap(),
        Pattern::new(6, 8, vec![0, 9, 18, 27, 36, 45, 54, 63]).unwrap(),
        Pattern::new(7, 8, vec![7, 14, 21, 28, 35, 42, 49, 56]).unwrap(),
        Pattern::new(8, 6, vec![0, 1, 2, 3, 4, 5]).unwrap(),
        Pattern::new(9, 6, vec![0, 8, 16, 24, 32, 40]).unwrap(),
        Pattern::new(10, 5, vec![0, 1, 2, 3, 4]).unwrap(),
        Pattern::new(11, 5, vec![0, 8, 16, 24, 32]).unwrap(),
        Pattern::new(12, 4, vec![0, 1, 2, 3]).unwrap(),
        Pattern::new(13, 4, vec![0, 8, 16, 24]).unwrap(),
    ]
}

fn bench_extract_all_patterns_initial_board(c: &mut Criterion) {
    let board = BitBoard::new();
    let patterns = create_benchmark_patterns();

    c.bench_function("extract_all_patterns_initial", |b| {
        b.iter(|| {
            black_box(extract_all_patterns(
                black_box(&board),
                black_box(&patterns),
            ))
        })
    });
}

fn bench_extract_all_patterns_midgame_board(c: &mut Criterion) {
    use prismind::board::{legal_moves, make_move};

    // 中盤の盤面を作成（何手か進めた状態）
    let mut board = BitBoard::new();
    for _ in 0..5 {
        let moves = legal_moves(&board);
        if moves != 0 {
            let _ = make_move(&mut board, moves.trailing_zeros() as u8);
        }
    }

    let patterns = create_benchmark_patterns();

    c.bench_function("extract_all_patterns_midgame", |b| {
        b.iter(|| {
            black_box(extract_all_patterns(
                black_box(&board),
                black_box(&patterns),
            ))
        })
    });
}

fn bench_extract_all_patterns_various_boards(c: &mut Criterion) {
    use prismind::board::{legal_moves, make_move};

    let patterns = create_benchmark_patterns();

    // 複数の盤面でベンチマーク
    let mut boards = vec![BitBoard::new()];

    // いくつかの進行度の盤面を作成
    let mut current = BitBoard::new();
    for _ in 0..3 {
        let moves = legal_moves(&current);
        if moves != 0 {
            let _ = make_move(&mut current, moves.trailing_zeros() as u8);
            boards.push(current);
        }
    }

    let mut board_idx = 0;

    c.bench_function("extract_all_patterns_various", |b| {
        b.iter(|| {
            let board = &boards[board_idx % boards.len()];
            board_idx = (board_idx + 1) % boards.len();
            black_box(extract_all_patterns(black_box(board), black_box(&patterns)))
        })
    });
}

/// Benchmark extract_all_patterns() with 1000 iterations to measure statistics
/// Target: 25μs以内 (average time, standard deviation, p99 percentile)
/// Task 14.2: 評価システムの最終パフォーマンス検証
fn bench_extract_all_patterns_statistics(c: &mut Criterion) {
    let board = BitBoard::new();
    let patterns = create_benchmark_patterns();

    let mut group = c.benchmark_group("extract_all_patterns_stats");
    group.sample_size(1000); // Exactly 1000 iterations as specified

    group.bench_function("extract_all_patterns_1000_iters", |b| {
        b.iter(|| {
            black_box(extract_all_patterns(
                black_box(&board),
                black_box(&patterns),
            ))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_extract_all_patterns_initial_board,
    bench_extract_all_patterns_midgame_board,
    bench_extract_all_patterns_various_boards,
    bench_extract_all_patterns_statistics
);
criterion_main!(benches);
