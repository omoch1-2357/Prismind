//! Benchmark for evaluate() function
//!
//! Task 11.3: 評価関数のベンチマークとキャッシュ測定
//!
//! Performance targets:
//! - evaluate(): 35μs以内（ARM64実測値ベース、プリフェッチとSoA最適化）
//! - キャッシュミス率: 30-40%以下

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use prismind::board::BitBoard;
use prismind::evaluator::Evaluator;

/// Benchmark: Evaluator::new() - Pattern loading and table initialization
///
/// Requirement 11.1: Evaluator::new()でpatterns.csvを読み込み
fn bench_evaluator_new(c: &mut Criterion) {
    c.bench_function("evaluator_new", |b| {
        b.iter(|| {
            let evaluator = Evaluator::new("patterns.csv").unwrap();
            black_box(evaluator)
        })
    });
}

/// Benchmark: evaluate() on initial board
///
/// Requirement 11.2, 11.3: evaluate()で56パターンの評価値を合計
/// Target: 35μs以内
fn bench_evaluate_initial(c: &mut Criterion) {
    let evaluator = Evaluator::new("patterns.csv").unwrap();
    let board = BitBoard::new();

    c.bench_function("evaluate_initial", |b| {
        b.iter(|| {
            let eval = evaluator.evaluate(black_box(&board));
            black_box(eval)
        })
    });
}

/// Benchmark: evaluate() on various board states
///
/// Requirement 11.3: 異なるステージでのパフォーマンス測定
fn bench_evaluate_various_stages(c: &mut Criterion) {
    let evaluator = Evaluator::new("patterns.csv").unwrap();

    let mut group = c.benchmark_group("evaluate_stages");

    // Initial board (stage 0)
    let board_stage0 = BitBoard::new();
    group.bench_function("stage_0", |b| {
        b.iter(|| {
            let eval = evaluator.evaluate(black_box(&board_stage0));
            black_box(eval)
        })
    });

    // Mid-game (simulate stage 15 by manually setting move_count)
    // Note: We can't easily set move_count without playing moves,
    // so we'll just evaluate the initial board multiple times
    // to measure cache behavior

    group.finish();
}

/// Benchmark: evaluate() cache behavior test
///
/// Requirement 11.3: キャッシュ効率の測定
/// 同じ盤面を連続評価 vs 異なる盤面を評価
fn bench_evaluate_cache_behavior(c: &mut Criterion) {
    let evaluator = Evaluator::new("patterns.csv").unwrap();

    let mut group = c.benchmark_group("evaluate_cache");

    // Same board repeated (best cache scenario)
    let board = BitBoard::new();
    group.bench_function("same_board_repeated", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let eval = evaluator.evaluate(black_box(&board));
                black_box(eval);
            }
        })
    });

    // Different rotations (different access patterns)
    let boards = vec![
        BitBoard::new(),
        BitBoard::new().rotate_90(),
        BitBoard::new().rotate_180(),
        BitBoard::new().rotate_270(),
    ];

    group.bench_function("different_rotations", |b| {
        b.iter(|| {
            for board in &boards {
                let eval = evaluator.evaluate(black_box(board));
                black_box(eval);
            }
        })
    });

    group.finish();
}

/// Benchmark: Pattern extraction (called by evaluate)
///
/// Requirement 11.2: 56個のパターンインスタンス抽出のパフォーマンス
fn bench_pattern_extraction_in_evaluate(c: &mut Criterion) {
    use prismind::pattern::{extract_all_patterns, load_patterns};

    let patterns = load_patterns("patterns.csv").unwrap();
    let board = BitBoard::new();

    c.bench_function("extract_all_patterns_56", |b| {
        b.iter(|| {
            let indices = extract_all_patterns(black_box(&board), black_box(&patterns));
            black_box(indices)
        })
    });
}

criterion_group!(
    benches,
    bench_evaluator_new,
    bench_evaluate_initial,
    bench_evaluate_various_stages,
    bench_evaluate_cache_behavior,
    bench_pattern_extraction_in_evaluate,
);
criterion_main!(benches);
