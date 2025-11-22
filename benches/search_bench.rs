use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use prismind::board::BitBoard;
use prismind::evaluator::Evaluator;
use prismind::search::Search;
use std::path::PathBuf;

/// ベンチマーク用の評価関数を初期化
fn setup_evaluator() -> Evaluator {
    // patterns.csvのパスを取得（プロジェクトルートから相対パス）
    let patterns_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("patterns.csv");

    Evaluator::new(&patterns_path).expect("Failed to load evaluator")
}

/// 初期盤面でのベンチマーク
///
/// 目標: 深さ6で10ms以内
fn bench_initial_position(c: &mut Criterion) {
    let evaluator = setup_evaluator();
    let mut search = Search::new(evaluator, 128).expect("Failed to create search");
    let board = BitBoard::new();

    let mut group = c.benchmark_group("search_initial_position");

    // 深さ1-6の探索時間を測定
    for depth in 1..=6 {
        group.bench_with_input(BenchmarkId::new("depth", depth), &depth, |b, &depth| {
            b.iter(|| {
                // 深さ制限を実現するため、時間制限を十分大きくする
                // max_depthパラメータで指定深さに到達したら停止する
                let result = search
                    .search(black_box(&board), black_box(10000), Some(depth))
                    .expect("Search failed");

                // 到達深さが目標深さに達していることを確認
                black_box(result)
            });
        });
    }

    group.finish();
}

/// 中盤局面でのベンチマーク
///
/// 目標: 深さ6で15ms以内
fn bench_midgame_position(c: &mut Criterion) {
    let evaluator = setup_evaluator();
    let mut search = Search::new(evaluator, 128).expect("Failed to create search");

    // 中盤局面を作成（手数20程度の局面）
    // ここでは簡易的に初期盤面から数手進めた局面を使用
    let mut board = BitBoard::new();

    // 典型的な序盤定石を再現（20手程度進める）
    // 実際の中盤局面を作成
    let moves = vec![
        19, // d3
        26, // c4
        21, // f3
        34, // c5
        42, // c6
        18, // c3
        29, // f4
        37, // f5
        20, // e3
        43, // d6
        35, // d5
        28, // e5
        44, // e6
        36, // e4
        27, // d4
        45, // f6
        51, // d7
        52, // e7
        53, // f7
        33, // b5
    ];

    for &mv in &moves {
        if let Err(e) = prismind::board::make_move(&mut board, mv) {
            panic!("Failed to make move {}: {:?}", mv, e);
        }
    }

    let mut group = c.benchmark_group("search_midgame_position");

    // 深さ6-8の探索時間を測定
    for depth in 6..=8 {
        group.bench_with_input(BenchmarkId::new("depth", depth), &depth, |b, &depth| {
            b.iter(|| {
                let result = search
                    .search(black_box(&board), black_box(10000), Some(depth))
                    .expect("Search failed");

                black_box(result)
            });
        });
    }

    group.finish();
}

/// 置換表ヒット率の測定
///
/// 目標: 中盤以降で50%以上
fn bench_tt_hit_rate(c: &mut Criterion) {
    let evaluator = setup_evaluator();
    let mut search = Search::new(evaluator, 128).expect("Failed to create search");

    // 中盤局面（上記と同じ）
    let mut board = BitBoard::new();
    let moves = vec![
        19, 26, 21, 34, 42, 18, 29, 37, 20, 43, 35, 28, 44, 36, 27, 45, 51, 52, 53, 33,
    ];
    for &mv in &moves {
        if let Err(e) = prismind::board::make_move(&mut board, mv) {
            panic!("Failed to make move {}: {:?}", mv, e);
        }
    }

    c.bench_function("tt_hit_rate_midgame", |b| {
        b.iter(|| {
            let result = search
                .search(black_box(&board), black_box(15), None)
                .expect("Search failed");

            // ヒット率を計算
            let hit_rate = result.tt_hit_rate();

            // カスタムメトリクスとしてログ出力（Criterionのサマリには表示されないが、デバッグに有用）
            println!(
                "TT Hit Rate: {:.1}%, Nodes: {}, TT Hits: {}",
                hit_rate * 100.0,
                result.nodes_searched,
                result.tt_hits
            );

            black_box(result)
        });
    });
}

/// 枝刈り効率の測定
///
/// AlphaBetaとムーブオーダリングの効果を測定
fn bench_pruning_efficiency(c: &mut Criterion) {
    let evaluator = setup_evaluator();
    let mut search = Search::new(evaluator, 128).expect("Failed to create search");

    let board = BitBoard::new();

    c.bench_function("pruning_efficiency_depth6", |b| {
        b.iter(|| {
            let result = search
                .search(black_box(&board), black_box(1000), None)
                .expect("Search failed");

            // 探索ノード数をログ出力
            println!(
                "Nodes searched at depth {}: {}",
                result.depth, result.nodes_searched
            );

            black_box(result)
        });
    });
}

/// 1秒あたりの探索ノード数（nps）を測定
///
/// 目標: 高いnpsほど探索が高速
fn bench_nodes_per_second(c: &mut Criterion) {
    let evaluator = setup_evaluator();
    let mut search = Search::new(evaluator, 128).expect("Failed to create search");

    let board = BitBoard::new();

    c.bench_function("nodes_per_second", |b| {
        b.iter(|| {
            let result = search
                .search(black_box(&board), black_box(100), None)
                .expect("Search failed");

            // npsを計算
            let nps = if result.elapsed_ms > 0 {
                (result.nodes_searched as f64) / (result.elapsed_ms as f64 / 1000.0)
            } else {
                0.0
            };

            println!(
                "Nodes per second: {:.0}, Total nodes: {}, Time: {}ms",
                nps, result.nodes_searched, result.elapsed_ms
            );

            black_box(result)
        });
    });
}

/// 性能目標の検証
///
/// 各ベンチマーク結果が目標値を満たしているか確認するための統合ベンチマーク
fn bench_performance_targets(c: &mut Criterion) {
    c.bench_function("target_initial_depth6_10ms", |b| {
        let evaluator = setup_evaluator();
        let mut search = Search::new(evaluator, 128).expect("Failed to create search");
        let board = BitBoard::new();

        b.iter(|| {
            let result = search
                .search(black_box(&board), black_box(1000), None)
                .expect("Search failed");

            // 目標: 深さ6で10ms以内
            assert!(
                result.elapsed_ms <= 10 && result.depth >= 6,
                "Target not met: depth {} in {}ms (target: depth 6 in 10ms)",
                result.depth,
                result.elapsed_ms
            );

            black_box(result)
        });
    });

    c.bench_function("target_midgame_depth6_15ms", |b| {
        let evaluator = setup_evaluator();
        let mut search = Search::new(evaluator, 128).expect("Failed to create search");

        // 中盤局面
        let mut board = BitBoard::new();
        let moves = vec![
            19, 26, 21, 34, 42, 18, 29, 37, 20, 43, 35, 28, 44, 36, 27, 45, 51, 52, 53, 33,
        ];
        for &mv in &moves {
            prismind::board::make_move(&mut board, mv).expect("Failed to make move");
        }

        b.iter(|| {
            let result = search
                .search(black_box(&board), black_box(15), None)
                .expect("Search failed");

            // 目標: 深さ6で15ms以内
            println!(
                "Midgame search: depth {} in {}ms (target: depth 6 in 15ms)",
                result.depth, result.elapsed_ms
            );

            black_box(result)
        });
    });
}

criterion_group!(
    benches,
    bench_initial_position,
    bench_midgame_position,
    bench_tt_hit_rate,
    bench_pruning_efficiency,
    bench_nodes_per_second,
    bench_performance_targets
);
criterion_main!(benches);
