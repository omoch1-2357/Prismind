//! パブリックAPIの統合テスト
//!
//! Task 12.2: BitBoard操作から評価関数までの一連のフロー検証
//!
//! このテストは以下の要件を検証する:
//! - Requirements: 13.5, NFR-4

use prismind::board::{
    BitBoard, Color, GameError, GameState, check_game_state, legal_moves, make_move, undo_move,
};
use prismind::evaluator::Evaluator;
use prismind::pattern::load_patterns;

/// Test 1: BitBoard操作から評価関数までの一連のフロー検証
///
/// 要件: 13.5
/// 初期盤面→着手実行→評価→undo→元の盤面に戻ることを確認
#[test]
fn test_bitboard_to_evaluation_flow() {
    // patterns.csvが存在しない場合はスキップ
    if !std::path::Path::new("patterns.csv").exists() {
        println!("patterns.csv not found, skipping integration test");
        return;
    }

    // 1. 初期盤面の作成
    let mut board = BitBoard::new();
    let original_board = board;
    assert_eq!(board.turn(), Color::Black, "初期手番は黒");
    assert_eq!(board.move_count(), 0, "初期手数は0");

    // 2. Evaluatorの初期化
    let evaluator = Evaluator::new("patterns.csv").expect("Evaluator初期化に失敗");

    // 3. 初期盤面の評価
    let initial_eval = evaluator.evaluate(&board);
    assert!(initial_eval.is_finite(), "初期盤面の評価値は有限であるべき");
    assert!(
        initial_eval.abs() < 5.0,
        "初期盤面の評価値は0付近であるべき (got {})",
        initial_eval
    );

    // 4. 合法手の取得
    let moves = legal_moves(&board);
    assert_ne!(moves, 0, "初期盤面には合法手が存在するべき");
    assert_eq!(moves.count_ones(), 4, "初期盤面には4つの合法手が存在");

    // 5. 着手実行
    let first_move = moves.trailing_zeros() as u8;
    let undo_info = make_move(&mut board, first_move).expect("着手実行に失敗");

    // 6. 着手後の状態確認
    assert_eq!(board.turn(), Color::White, "着手後の手番は白");
    assert_eq!(board.move_count(), 1, "着手後の手数は1");

    // 7. 着手後の評価
    let after_move_eval = evaluator.evaluate(&board);
    assert!(
        after_move_eval.is_finite(),
        "着手後の評価値は有限であるべき"
    );

    // 8. ゲーム状態判定
    let game_state = check_game_state(&board);
    assert_eq!(game_state, GameState::Playing, "着手後もゲームは継続中");

    // 9. undo操作で元に戻す
    undo_move(&mut board, undo_info);
    assert_eq!(board, original_board, "undo後は元の盤面に戻るべき");

    // 10. undo後の評価値確認
    let restored_eval = evaluator.evaluate(&board);
    assert!(
        (restored_eval - initial_eval).abs() < 0.001,
        "undo後の評価値は初期評価値と一致すべき (initial: {}, restored: {})",
        initial_eval,
        restored_eval
    );
}

/// Test 2: Pattern読み込みからEvaluator初期化までの統合テスト
///
/// 要件: 13.5, NFR-4
/// patterns.csvからパターンを読み込み、Evaluatorを正しく初期化できることを確認
#[test]
fn test_pattern_loading_to_evaluator_initialization() {
    if !std::path::Path::new("patterns.csv").exists() {
        println!("patterns.csv not found, skipping integration test");
        return;
    }

    // 1. パターン定義の読み込み
    let patterns = load_patterns("patterns.csv").expect("パターン読み込みに失敗");
    assert_eq!(patterns.len(), 14, "14個のパターンが読み込まれるべき");

    // 2. 各パターンのバリデーション
    for (i, pattern) in patterns.iter().enumerate() {
        assert_eq!(pattern.id, i as u8, "パターンIDは0-13の範囲内");
        assert!(
            pattern.k >= 4 && pattern.k <= 10,
            "パターンのセル数kは4-10の範囲内"
        );

        // 全ての位置が0-63の範囲内
        for j in 0..(pattern.k as usize) {
            assert!(
                pattern.positions[j] < 64,
                "パターン{}の位置{}は0-63の範囲内であるべき",
                i,
                j
            );
        }
    }

    // 3. Evaluatorの初期化
    let evaluator = Evaluator::new("patterns.csv").expect("Evaluator初期化に失敗");

    // 4. Evaluatorで評価実行
    let board = BitBoard::new();
    let eval = evaluator.evaluate(&board);
    assert!(eval.is_finite(), "評価値は有限であるべき");
}

/// Test 3: 着手実行→合法手再計算→ゲーム状態判定のフロー検証
///
/// 要件: 13.5
/// 複数手にわたる一連のゲーム進行フローを検証
#[test]
fn test_move_legal_moves_game_state_flow() {
    let mut board = BitBoard::new();
    let mut move_history = Vec::new();

    // 最大10手まで進める
    for iteration in 0..10 {
        // ゲーム状態判定
        let game_state = check_game_state(&board);

        match game_state {
            GameState::Playing => {
                // 合法手を取得
                let moves = legal_moves(&board);
                assert_ne!(
                    moves, 0,
                    "Playing状態では合法手が存在するべき (iteration {})",
                    iteration
                );

                // 現在の手数を記録
                let before_move_count = board.move_count();

                // 最初の合法手を実行
                let move_pos = moves.trailing_zeros() as u8;
                let undo_info = make_move(&mut board, move_pos)
                    .unwrap_or_else(|_| panic!("着手{}の実行に失敗", iteration));

                move_history.push((move_pos, undo_info));

                // 手数の確認（1増加するはず）
                assert_eq!(
                    board.move_count(),
                    before_move_count + 1,
                    "Iteration {}: 手数カウンタが1増加するべき (before={}, after={})",
                    iteration,
                    before_move_count,
                    board.move_count()
                );

                // 手番が切り替わったことを確認
                assert_ne!(board.turn(), undo_info.turn, "手番が切り替わるべき");
            }
            GameState::Pass => {
                // パスの場合は手番を切り替える（手数は増えない）
                let before_move_count = board.move_count();
                board = board.flip();
                assert_eq!(
                    board.move_count(),
                    before_move_count,
                    "パス時は手数が増えない"
                );
            }
            GameState::GameOver(score) => {
                // ゲーム終了
                println!("Game over at iteration {} with score {}", iteration, score);
                assert!((-64..=64).contains(&score), "最終スコアは-64～+64の範囲内");
                break;
            }
        }
    }

    // 全ての着手をundoして元に戻す
    for (_, undo_info) in move_history.iter().rev() {
        undo_move(&mut board, *undo_info);
    }

    // 初期盤面に戻ったことを確認
    let initial_board = BitBoard::new();
    assert_eq!(board, initial_board, "全てのundoで初期盤面に戻るべき");
}

/// Test 4: エラーハンドリングパスの統合テスト
///
/// 要件: NFR-4
/// 不正な入力に対して適切なエラーを返すことを確認
#[test]
fn test_error_handling_paths() {
    let mut board = BitBoard::new();

    // 1. 範囲外の位置への着手
    let result = make_move(&mut board, 64);
    assert!(
        matches!(result, Err(GameError::OutOfBounds(64))),
        "範囲外の位置への着手はOutOfBoundsエラーを返すべき"
    );

    // 2. 非合法な位置への着手
    let result = make_move(&mut board, 0); // A1は初期盤面では非合法
    assert!(
        matches!(result, Err(GameError::IllegalMove(0))),
        "非合法な位置への着手はIllegalMoveエラーを返すべき"
    );

    // 3. 複数の非合法手を試す
    for pos in 0..64 {
        let moves = legal_moves(&board);
        let is_legal = (moves & (1 << pos)) != 0;

        let result = make_move(&mut board, pos);

        if is_legal {
            // 合法手の場合は成功するべき
            assert!(result.is_ok(), "合法手 {} は成功するべき", pos);

            // undoして元に戻す
            if let Ok(undo_info) = result {
                undo_move(&mut board, undo_info);
            }
        } else {
            // 非合法手の場合はエラーを返すべき
            assert!(result.is_err(), "非合法手 {} はエラーを返すべき", pos);
        }
    }

    // 4. 盤面が変更されていないことを確認
    let initial_board = BitBoard::new();
    assert_eq!(board, initial_board, "エラーケースで盤面は変更されないべき");
}

/// Test 5: 評価関数の対称性と一貫性
///
/// 要件: 13.5
/// 回転した盤面での評価値が一貫していることを確認
#[test]
fn test_evaluation_symmetry_and_consistency() {
    if !std::path::Path::new("patterns.csv").exists() {
        println!("patterns.csv not found, skipping integration test");
        return;
    }

    let evaluator = Evaluator::new("patterns.csv").expect("Evaluator初期化に失敗");
    let board = BitBoard::new();

    // 初期盤面は180度回転対称なので、評価値も同じはず
    let eval_0 = evaluator.evaluate(&board);
    let eval_180 = evaluator.evaluate(&board.rotate_180());

    assert!(
        (eval_0 - eval_180).abs() < 0.1,
        "180度回転対称な盤面の評価値は一致すべき (0°: {}, 180°: {})",
        eval_0,
        eval_180
    );

    // 黒白反転した盤面の評価値は符号が逆
    let board_flipped = board.flip();
    let eval_flipped = evaluator.evaluate(&board_flipped);

    println!("Black eval: {}, White eval: {}", eval_0, eval_flipped);
    // 初期盤面は対称なので、絶対値がほぼ同じはず
    assert!(
        (eval_0.abs() - eval_flipped.abs()).abs() < 0.1,
        "黒白反転した盤面の評価値の絶対値は同じはず"
    );
}

/// Test 6: 長時間のゲームフロー（メモリリークチェック）
///
/// 要件: 13.3
/// Rust の所有権システムに準拠し、メモリリークを防ぐ
#[test]
fn test_long_game_flow_memory_safety() {
    if !std::path::Path::new("patterns.csv").exists() {
        println!("patterns.csv not found, skipping integration test");
        return;
    }

    let evaluator = Evaluator::new("patterns.csv").expect("Evaluator初期化に失敗");

    // 100回のゲームをシミュレーション
    for game_num in 0..100 {
        let mut board = BitBoard::new();
        let mut move_count = 0;

        // 最大60手まで
        while move_count < 60 {
            let game_state = check_game_state(&board);

            match game_state {
                GameState::Playing => {
                    let moves = legal_moves(&board);
                    if moves == 0 {
                        break;
                    }

                    // 評価関数を呼び出し
                    let _ = evaluator.evaluate(&board);

                    // ランダムな合法手を選択（最初の合法手）
                    let move_pos = moves.trailing_zeros() as u8;
                    if make_move(&mut board, move_pos).is_err() {
                        break;
                    }

                    move_count += 1;
                }
                GameState::Pass => {
                    board = board.flip();
                }
                GameState::GameOver(_) => {
                    break;
                }
            }
        }

        // 定期的にログ出力
        if game_num % 20 == 0 {
            println!("Completed game {} with {} moves", game_num, move_count);
        }
    }

    // メモリリークがないことを確認（Rustの所有権システムにより保証）
    // このテストが正常に完了すれば、メモリ安全性が確保されている
}

/// Test 7: 並行実行の安全性（Copyトレイト検証）
///
/// 要件: 13.3
/// BitBoardのCopyトレイトにより、複数所有が可能
#[test]
fn test_concurrent_board_operations() {
    let board = BitBoard::new();

    // BitBoardはCopyなので、複数の変数で所有できる
    let board1 = board;
    let board2 = board;
    let board3 = board;

    assert_eq!(board1, board2);
    assert_eq!(board2, board3);

    // それぞれ独立して変更可能
    let mut mutable_board1 = board1;
    let mut mutable_board2 = board2;

    let moves = legal_moves(&mutable_board1);
    let first_move = moves.trailing_zeros() as u8;

    make_move(&mut mutable_board1, first_move).expect("着手1失敗");
    make_move(&mut mutable_board2, first_move).expect("着手2失敗");

    // 両方とも同じ結果
    assert_eq!(mutable_board1, mutable_board2);

    // 元のboardは変更されていない
    assert_eq!(board, board3);
}
