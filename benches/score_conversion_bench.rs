use criterion::{Criterion, criterion_group, criterion_main};
use prismind::evaluator::{score_to_u16, u16_to_score};
use std::hint::black_box;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use criterion::BenchmarkId;

#[cfg(target_arch = "aarch64")]
use prismind::evaluator::u16_to_score_simd;

#[cfg(target_arch = "x86_64")]
use prismind::evaluator::u16_to_score_simd;

#[cfg(target_arch = "x86_64")]
use prismind::x86_64::{u16_to_score_simd_avx2, u16_to_score_simd_sse};

/// Benchmark u16 → f32 conversion (scalar version)
fn bench_u16_to_score_scalar(c: &mut Criterion) {
    c.bench_function("u16_to_score_scalar", |b| {
        let value = black_box(40000u16);
        b.iter(|| u16_to_score(value))
    });
}

/// Benchmark f32 → u16 conversion
fn bench_score_to_u16(c: &mut Criterion) {
    c.bench_function("score_to_u16", |b| {
        let score = black_box(25.5f32);
        b.iter(|| score_to_u16(score))
    });
}

/// Benchmark round-trip conversion
fn bench_round_trip_conversion(c: &mut Criterion) {
    c.bench_function("round_trip_conversion", |b| {
        let original = black_box(32768u16);
        b.iter(|| {
            let score = u16_to_score(original);
            score_to_u16(score)
        })
    });
}

/// Benchmark batch conversion (8 values using scalar version)
fn bench_batch_conversion_scalar(c: &mut Criterion) {
    c.bench_function("batch_8_values_scalar", |b| {
        let values: [u16; 8] = black_box([0, 10000, 20000, 32768, 40000, 50000, 60000, 65535]);
        b.iter(|| {
            let mut scores = [0.0f32; 8];
            for i in 0..8 {
                scores[i] = u16_to_score(values[i]);
            }
            scores
        })
    });
}

/// Benchmark SIMD version (ARM64 only)
#[cfg(target_arch = "aarch64")]
fn bench_u16_to_score_simd_arm64(c: &mut Criterion) {
    c.bench_function("u16_to_score_simd_arm64", |b| {
        let values: [u16; 8] = black_box([0, 10000, 20000, 32768, 40000, 50000, 60000, 65535]);
        b.iter(|| u16_to_score_simd(&values))
    });
}

/// Compare scalar vs SIMD performance (ARM64 only)
#[cfg(target_arch = "aarch64")]
fn bench_scalar_vs_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_vs_simd");

    let values: [u16; 8] = [0, 10000, 20000, 32768, 40000, 50000, 60000, 65535];

    group.bench_with_input(BenchmarkId::new("scalar", 8), &values, |b, values| {
        b.iter(|| {
            let mut scores = [0.0f32; 8];
            for i in 0..8 {
                scores[i] = u16_to_score(values[i]);
            }
            scores
        })
    });

    group.bench_with_input(BenchmarkId::new("simd", 8), &values, |b, values| {
        b.iter(|| u16_to_score_simd(black_box(values)))
    });

    group.finish();
}

/// Benchmark SIMD version (x86-64)
#[cfg(target_arch = "x86_64")]
fn bench_u16_to_score_simd_x86_64(c: &mut Criterion) {
    c.bench_function("u16_to_score_simd_x86_64", |b| {
        let values: [u16; 8] = black_box([0, 10000, 20000, 32768, 40000, 50000, 60000, 65535]);
        b.iter(|| u16_to_score_simd(&values))
    });
}

/// Benchmark SSE version (x86-64, if available)
#[cfg(target_arch = "x86_64")]
fn bench_u16_to_score_simd_sse(c: &mut Criterion) {
    if !is_x86_feature_detected!("sse4.1") {
        return;
    }
    c.bench_function("u16_to_score_simd_sse", |b| {
        let values: [u16; 8] = black_box([0, 10000, 20000, 32768, 40000, 50000, 60000, 65535]);
        b.iter(|| unsafe { u16_to_score_simd_sse(&values) })
    });
}

/// Benchmark AVX2 version (x86-64, if available)
#[cfg(target_arch = "x86_64")]
fn bench_u16_to_score_simd_avx2(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    c.bench_function("u16_to_score_simd_avx2", |b| {
        let values: [u16; 8] = black_box([0, 10000, 20000, 32768, 40000, 50000, 60000, 65535]);
        b.iter(|| unsafe { u16_to_score_simd_avx2(&values) })
    });
}

/// Compare scalar vs SIMD performance (x86-64)
#[cfg(target_arch = "x86_64")]
fn bench_scalar_vs_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_vs_simd");

    let values: [u16; 8] = [0, 10000, 20000, 32768, 40000, 50000, 60000, 65535];

    group.bench_with_input(BenchmarkId::new("scalar", 8), &values, |b, values| {
        b.iter(|| {
            let mut scores = [0.0f32; 8];
            for i in 0..8 {
                scores[i] = u16_to_score(values[i]);
            }
            scores
        })
    });

    group.bench_with_input(BenchmarkId::new("simd", 8), &values, |b, values| {
        b.iter(|| u16_to_score_simd(black_box(values)))
    });

    if is_x86_feature_detected!("sse4.1") {
        group.bench_with_input(BenchmarkId::new("sse4.1", 8), &values, |b, values| {
            b.iter(|| unsafe { u16_to_score_simd_sse(black_box(values)) })
        });
    }

    if is_x86_feature_detected!("avx2") {
        group.bench_with_input(BenchmarkId::new("avx2", 8), &values, |b, values| {
            b.iter(|| unsafe { u16_to_score_simd_avx2(black_box(values)) })
        });
    }

    group.finish();
}

// ARM64 configuration
#[cfg(target_arch = "aarch64")]
criterion_group!(
    benches,
    bench_u16_to_score_scalar,
    bench_score_to_u16,
    bench_round_trip_conversion,
    bench_batch_conversion_scalar,
    bench_u16_to_score_simd_arm64,
    bench_scalar_vs_simd
);

// x86-64 configuration (with SIMD benchmarks)
#[cfg(target_arch = "x86_64")]
criterion_group!(
    benches,
    bench_u16_to_score_scalar,
    bench_score_to_u16,
    bench_round_trip_conversion,
    bench_batch_conversion_scalar,
    bench_u16_to_score_simd_x86_64,
    bench_u16_to_score_simd_sse,
    bench_u16_to_score_simd_avx2,
    bench_scalar_vs_simd
);

// Non-ARM64/x86-64 configuration (no SIMD benchmarks)
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
criterion_group!(
    benches,
    bench_u16_to_score_scalar,
    bench_score_to_u16,
    bench_round_trip_conversion,
    bench_batch_conversion_scalar
);

criterion_main!(benches);
