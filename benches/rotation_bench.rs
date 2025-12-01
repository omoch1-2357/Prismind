use criterion::{Criterion, criterion_group, criterion_main};
use prismind::board::BitBoard;
use std::hint::black_box;

fn bench_rotate_90(c: &mut Criterion) {
    let board = BitBoard::new();

    c.bench_function("rotate_90", |b| b.iter(|| black_box(board.rotate_90())));
}

fn bench_rotate_180(c: &mut Criterion) {
    let board = BitBoard::new();

    c.bench_function("rotate_180", |b| b.iter(|| black_box(board.rotate_180())));
}

fn bench_rotate_270(c: &mut Criterion) {
    let board = BitBoard::new();

    c.bench_function("rotate_270", |b| b.iter(|| black_box(board.rotate_270())));
}

fn bench_four_rotations(c: &mut Criterion) {
    let board = BitBoard::new();

    c.bench_function("four_rotations_90", |b| {
        b.iter(|| {
            let board = black_box(board);
            let rot1 = board.rotate_90();
            let rot2 = rot1.rotate_90();
            let rot3 = rot2.rotate_90();
            black_box(rot3.rotate_90())
        })
    });
}

/// Benchmark rotate_180() with 1000 iterations to measure statistics
/// Target: 200ns以内 (average time, standard deviation, p99 percentile)
fn bench_rotate_180_statistics(c: &mut Criterion) {
    let board = BitBoard::new();

    let mut group = c.benchmark_group("rotate_180_stats");
    group.sample_size(1000); // Exactly 1000 iterations as specified

    group.bench_function("rotate_180_1000_iters", |b| {
        b.iter(|| black_box(board.rotate_180()))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rotate_90,
    bench_rotate_180,
    bench_rotate_270,
    bench_four_rotations,
    bench_rotate_180_statistics
);
criterion_main!(benches);
