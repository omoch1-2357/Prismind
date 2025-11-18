use criterion::{black_box, criterion_group, criterion_main, Criterion};
use prismind::board::BitBoard;

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

criterion_group!(
    benches,
    bench_rotate_90,
    bench_rotate_180,
    bench_rotate_270,
    bench_four_rotations
);
criterion_main!(benches);
