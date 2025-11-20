//! ステージ管理機能の統合テスト
//!
//! Task 10: ステージ管理の境界値検証

use prismind::board::BitBoard;
use prismind::evaluator::calculate_stage;

#[test]
fn test_stage_0_for_moves_0_to_1() {
    // 0-1手でステージ0を返すテスト
    let board = BitBoard::new();
    assert_eq!(board.move_count(), 0);
    assert_eq!(calculate_stage(0), 0, "手数0でステージ0を返すべき");
    assert_eq!(calculate_stage(1), 0, "手数1でステージ0を返すべき");
}

#[test]
fn test_stage_1_for_moves_2_to_3() {
    // 2-3手でステージ1を返すテスト
    assert_eq!(calculate_stage(2), 1, "手数2でステージ1を返すべき");
    assert_eq!(calculate_stage(3), 1, "手数3でステージ1を返すべき");
}

#[test]
fn test_stage_29_for_moves_58_to_59() {
    // 58-59手でステージ29を返すテスト
    assert_eq!(calculate_stage(58), 29, "手数58でステージ29を返すべき");
    assert_eq!(calculate_stage(59), 29, "手数59でステージ29を返すべき");
}

#[test]
fn test_stage_29_for_move_60() {
    // 手数60でステージ29を返すテスト
    assert_eq!(calculate_stage(60), 29, "手数60でステージ29を返すべき");
}

#[test]
fn test_stage_range_is_0_to_29() {
    // 0-29の範囲内の整数を返すことを保証
    for move_count in 0..=60 {
        let stage = calculate_stage(move_count);
        assert!(
            stage <= 29,
            "ステージは0-29の範囲内であるべき。手数{}でステージ{}",
            move_count,
            stage
        );
    }
}

#[test]
fn test_stage_calculation_formula() {
    // 手数÷2の公式が正しく動作することを確認
    assert_eq!(calculate_stage(0), 0);
    assert_eq!(calculate_stage(4), 2);
    assert_eq!(calculate_stage(5), 2);
    assert_eq!(calculate_stage(10), 5);
    assert_eq!(calculate_stage(11), 5);
    assert_eq!(calculate_stage(20), 10);
    assert_eq!(calculate_stage(30), 15);
    assert_eq!(calculate_stage(40), 20);
    assert_eq!(calculate_stage(50), 25);
    assert_eq!(calculate_stage(56), 28);
    assert_eq!(calculate_stage(57), 28);
}

#[test]
fn test_stage_allows_independent_eval_table_access() {
    // 各ステージごとに独立した評価テーブルへのアクセスを可能にする
    // ステージ番号がインデックスとして使用可能であることを確認
    let stage = calculate_stage(10);
    assert!(
        stage < 30,
        "ステージ番号は評価テーブルの配列インデックスとして使用可能であるべき"
    );

    // 異なる手数でも同じステージになることを確認（評価テーブル再利用）
    assert_eq!(calculate_stage(10), calculate_stage(11));
    assert_ne!(calculate_stage(10), calculate_stage(12));
}
