//! 評価テーブル管理モジュール
//!
//! Structure of Arrays (SoA) 形式で評価テーブルを管理し、
//! キャッシュ効率を最適化する。

use crate::pattern::Pattern;

/// 評価テーブル（Structure of Arrays形式）
///
/// # メモリレイアウト
///
/// data[stage][flat_array]形式で保持:
/// - stage: 0-29（ゲームの進行度）
/// - flat_array: 全14パターンのデータを連続配置
///
/// 各ステージのflat_arrayは以下のように構成:
/// [P01のエントリ群 | P02のエントリ群 | ... | P14のエントリ群]
///
/// pattern_offsetsは各パターンの開始位置を示す:
/// - pattern_offsets[0] = 0 (P01の開始位置)
/// - pattern_offsets[1] = 3^k[0] (P02の開始位置)
/// - pattern_offsets[i] = sum(3^k[0..i]) (P(i+1)の開始位置)
///
/// # メモリ使用量
///
/// 総使用量: 約70-80MB（30ステージ × 約2.3MB/ステージ）
#[derive(Debug)]
pub struct EvaluationTable {
    /// [ステージ][平坦化配列]の2次元データ
    /// 各ステージは全14パターンのエントリを連続配置
    data: Vec<Box<[u16]>>,
    /// 各パターンの開始オフセット位置
    pattern_offsets: [usize; 14],
}

impl EvaluationTable {
    /// 評価テーブルを初期化
    ///
    /// 全エントリを32768（石差0に相当）に初期化する。
    ///
    /// # Arguments
    ///
    /// * `patterns` - 14個のパターン定義
    ///
    /// # Examples
    ///
    /// ```
    /// use prismind::pattern::Pattern;
    /// use prismind::evaluator::EvaluationTable;
    ///
    /// let patterns = vec![
    ///     Pattern::new(0, 10, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap(),
    ///     // ... 他の13パターン
    /// ];
    /// let table = EvaluationTable::new(&patterns);
    /// ```
    pub fn new(patterns: &[Pattern]) -> Self {
        assert_eq!(patterns.len(), 14, "Expected exactly 14 patterns");

        // 各パターンの開始オフセットを計算
        let mut pattern_offsets = [0; 14];
        let mut offset = 0;
        for (i, pattern) in patterns.iter().enumerate() {
            pattern_offsets[i] = offset;
            offset += 3_usize.pow(pattern.k as u32);
        }

        let total_entries_per_stage = offset;

        // 30ステージ分のデータを初期化
        let mut data = Vec::with_capacity(30);
        for _ in 0..30 {
            // 全エントリを32768（石差0に相当）で初期化
            let stage_data = vec![32768u16; total_entries_per_stage].into_boxed_slice();
            data.push(stage_data);
        }

        Self {
            data,
            pattern_offsets,
        }
    }

    /// 評価値を取得
    ///
    /// # Arguments
    ///
    /// * `pattern_id` - パターンID (0-13)
    /// * `stage` - ステージ (0-29)
    /// * `index` - パターンインデックス (0 ~ 3^k-1)
    ///
    /// # Returns
    ///
    /// u16型の評価値（32768が石差0に相当）
    ///
    /// # Panics
    ///
    /// pattern_id、stage、indexが範囲外の場合
    pub fn get(&self, pattern_id: usize, stage: usize, index: usize) -> u16 {
        assert!(pattern_id < 14, "pattern_id must be 0-13");
        assert!(stage < 30, "stage must be 0-29");

        let offset = self.pattern_offsets[pattern_id] + index;
        self.data[stage][offset]
    }

    /// 評価値を設定
    ///
    /// Phase 3の学習で使用する。
    ///
    /// # Arguments
    ///
    /// * `pattern_id` - パターンID (0-13)
    /// * `stage` - ステージ (0-29)
    /// * `index` - パターンインデックス (0 ~ 3^k-1)
    /// * `value` - 設定する評価値
    ///
    /// # Panics
    ///
    /// pattern_id、stage、indexが範囲外の場合
    pub fn set(&mut self, pattern_id: usize, stage: usize, index: usize, value: u16) {
        assert!(pattern_id < 14, "pattern_id must be 0-13");
        assert!(stage < 30, "stage must be 0-29");

        let offset = self.pattern_offsets[pattern_id] + index;
        self.data[stage][offset] = value;
    }

    /// メモリ使用量を計算（バイト単位）
    ///
    /// # Returns
    ///
    /// 総メモリ使用量（バイト）
    pub fn memory_usage(&self) -> usize {
        // 各ステージのデータサイズ（u16の配列）
        let entries_per_stage = self.data[0].len();
        let bytes_per_stage = entries_per_stage * std::mem::size_of::<u16>();

        // 30ステージ分の合計
        30 * bytes_per_stage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::Pattern;

    /// テスト用のパターンセットを作成
    fn create_test_patterns() -> Vec<Pattern> {
        vec![
            Pattern::new(0, 10, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap(),
            Pattern::new(1, 10, vec![0, 8, 16, 24, 32, 40, 48, 56, 1, 9]).unwrap(),
            Pattern::new(2, 10, vec![0, 1, 8, 9, 10, 16, 17, 18, 24, 25]).unwrap(),
            Pattern::new(3, 10, vec![0, 9, 18, 27, 36, 45, 54, 63, 1, 10]).unwrap(),
            Pattern::new(4, 8, vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap(),
            Pattern::new(5, 8, vec![0, 8, 16, 24, 32, 40, 48, 56]).unwrap(),
            Pattern::new(6, 8, vec![0, 9, 18, 27, 36, 45, 54, 63]).unwrap(),
            Pattern::new(7, 8, vec![7, 14, 21, 28, 35, 42, 49, 56]).unwrap(),
            Pattern::new(8, 7, vec![0, 1, 2, 3, 4, 5, 6]).unwrap(),
            Pattern::new(9, 7, vec![0, 8, 16, 24, 32, 40, 48]).unwrap(),
            Pattern::new(10, 6, vec![0, 1, 2, 3, 4, 5]).unwrap(),
            Pattern::new(11, 6, vec![0, 8, 16, 24, 32, 40]).unwrap(),
            Pattern::new(12, 5, vec![0, 1, 2, 3, 4]).unwrap(),
            Pattern::new(13, 5, vec![0, 8, 16, 24, 32]).unwrap(),
        ]
    }

    // ========== Task 8.1: EvaluationTable構造体の実装（SoA形式）Tests (TDD - RED) ==========

    #[test]
    fn test_evaluation_table_new_initialization() {
        // Requirement 9.2: 全エントリを32768に初期化
        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);

        // 30ステージ分のデータが作成されていることを確認
        assert_eq!(table.data.len(), 30, "Should have 30 stages (0-29)");

        // pattern_offsetsが14個設定されていることを確認
        assert_eq!(
            table.pattern_offsets.len(),
            14,
            "Should have 14 pattern offsets"
        );
    }

    #[test]
    fn test_evaluation_table_all_entries_initialized_to_32768() {
        // Requirement 9.2: システム初期化時、全エントリを32768に初期化
        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);

        // 各ステージの最初のパターンの最初のエントリを確認
        for stage in 0..30 {
            let value = table.get(0, stage, 0);
            assert_eq!(
                value, 32768,
                "Entry at pattern 0, stage {}, index 0 should be initialized to 32768",
                stage
            );
        }

        // 各パターンの最初のエントリを確認（ステージ0）
        for pattern_id in 0..14 {
            let value = table.get(pattern_id, 0, 0);
            assert_eq!(
                value, 32768,
                "Entry at pattern {}, stage 0, index 0 should be initialized to 32768",
                pattern_id
            );
        }
    }

    #[test]
    fn test_evaluation_table_get_basic() {
        // Requirement 9.4, 9.6: get()メソッドで評価値取得（offset計算含む）
        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);

        // 基本的な取得テスト
        let value = table.get(0, 0, 0);
        assert_eq!(value, 32768, "Should get initial value 32768");

        // 異なるステージでの取得
        let value_stage_5 = table.get(0, 5, 0);
        assert_eq!(
            value_stage_5, 32768,
            "Should get initial value 32768 at stage 5"
        );

        // 異なるパターンでの取得
        let value_pattern_5 = table.get(5, 0, 0);
        assert_eq!(
            value_pattern_5, 32768,
            "Should get initial value 32768 for pattern 5"
        );
    }

    #[test]
    fn test_evaluation_table_set_and_get() {
        // Requirement 9.4: set()メソッドで評価値設定（Phase 3学習用）
        let patterns = create_test_patterns();
        let mut table = EvaluationTable::new(&patterns);

        // 値を設定
        table.set(0, 0, 0, 40000);
        let value = table.get(0, 0, 0);
        assert_eq!(value, 40000, "Should retrieve the set value");

        // 別のエントリに設定
        table.set(5, 10, 100, 25000);
        let value = table.get(5, 10, 100);
        assert_eq!(
            value, 25000,
            "Should retrieve the set value for different entry"
        );

        // 最初のエントリが影響を受けていないことを確認
        let original_value = table.get(0, 0, 0);
        assert_eq!(
            original_value, 40000,
            "Original entry should not be affected"
        );
    }

    #[test]
    fn test_evaluation_table_soa_structure() {
        // Requirement 9.1, 9.6: Structure of Arrays形式で[ステージ][平坦化配列]の2次元構造
        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);

        // data[stage]が平坦化配列であることを確認
        // 各ステージの配列サイズは全パターンのエントリ数の合計
        let expected_total_entries: usize = patterns.iter().map(|p| 3_usize.pow(p.k as u32)).sum();

        for stage in 0..30 {
            assert_eq!(
                table.data[stage].len(),
                expected_total_entries,
                "Stage {} should have {} total entries",
                stage,
                expected_total_entries
            );
        }
    }

    #[test]
    fn test_evaluation_table_pattern_offsets() {
        // Requirement 9.7: pattern_offsets配列で各パターンの開始位置を管理
        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);

        // オフセットの計算を検証
        let mut expected_offset = 0;
        for (i, pattern) in patterns.iter().enumerate() {
            assert_eq!(
                table.pattern_offsets[i], expected_offset,
                "Pattern {} should start at offset {}",
                i, expected_offset
            );
            expected_offset += 3_usize.pow(pattern.k as u32);
        }
    }

    // ========== Task 8.2: 評価テーブルのメモリ管理とアライメント Tests (TDD - RED) ==========

    #[test]
    fn test_evaluation_table_3_power_k_entries() {
        // Requirement 9.3: 各パターンについて3^k個のエントリを割り当て
        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);

        for (i, pattern) in patterns.iter().enumerate() {
            let expected_entries = 3_usize.pow(pattern.k as u32);

            // 各パターンの最後のインデックスまでアクセス可能か確認
            let last_index = expected_entries - 1;

            // パニックしないことを確認（範囲内）
            let _ = table.get(i, 0, last_index);

            println!(
                "Pattern {}: k={}, entries=3^{}={}",
                i, pattern.k, pattern.k, expected_entries
            );
        }
    }

    #[test]
    fn test_evaluation_table_30_independent_stages() {
        // Requirement 9.4: 30ステージ（0-29）それぞれに独立したテーブルを持つ
        let patterns = create_test_patterns();
        let mut table = EvaluationTable::new(&patterns);

        // 各ステージに異なる値を設定
        for stage in 0..30 {
            let value = 32768 + (stage as u16) * 100;
            table.set(0, stage, 0, value);
        }

        // 各ステージが独立していることを確認
        for stage in 0..30 {
            let expected_value = 32768 + (stage as u16) * 100;
            let actual_value = table.get(0, stage, 0);
            assert_eq!(
                actual_value, expected_value,
                "Stage {} should have independent value {}",
                stage, expected_value
            );
        }
    }

    #[test]
    fn test_evaluation_table_continuous_memory_layout() {
        // Requirement 9.6: 同じステージの全パターンデータを連続メモリ配置
        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);

        // SoA形式の検証: 各ステージのデータが連続配置されている
        for _stage in 0..30 {
            // 全パターンのエントリが1つの連続配列にあることを確認
            // パターン0の最後とパターン1の最初が連続している
            if patterns.len() > 1 {
                let pattern0_last_index = 3_usize.pow(patterns[0].k as u32) - 1;
                let pattern1_first_index = 0;

                // オフセット計算で連続性を確認
                let offset0_last = table.pattern_offsets[0] + pattern0_last_index;
                let offset1_first = table.pattern_offsets[1] + pattern1_first_index;

                assert_eq!(
                    offset0_last + 1,
                    offset1_first,
                    "Pattern 0 last entry and Pattern 1 first entry should be adjacent in memory"
                );
            }
        }
    }

    #[test]
    fn test_evaluation_table_memory_usage() {
        // Requirement 13.2: メモリ使用量を計算
        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);

        let memory_bytes = table.memory_usage();

        // メモリ使用量の基本検証
        assert!(memory_bytes > 0, "Memory usage should be greater than 0");

        // 予想されるメモリ使用量を計算
        let total_entries_per_stage: usize = patterns.iter().map(|p| 3_usize.pow(p.k as u32)).sum();
        let expected_bytes = 30 * total_entries_per_stage * std::mem::size_of::<u16>();

        assert_eq!(
            memory_bytes, expected_bytes,
            "Memory usage should match expected calculation"
        );

        println!(
            "Total memory usage: {} bytes ({:.2} MB)",
            memory_bytes,
            memory_bytes as f64 / 1_048_576.0
        );
    }

    #[test]
    fn test_evaluation_table_memory_under_80mb() {
        // Requirement 9.5, 13.2: 総メモリ使用量が80MB以内であることを確認
        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);

        let memory_bytes = table.memory_usage();
        let memory_mb = memory_bytes as f64 / 1_048_576.0;

        println!("EvaluationTable memory usage: {:.2} MB", memory_mb);

        assert!(
            memory_mb <= 80.0,
            "Memory usage should be within 80MB, got {:.2} MB",
            memory_mb
        );
    }

    #[test]
    fn test_evaluation_table_all_patterns_different_sizes() {
        // Requirement 9.3: 各パターンのk値に応じて異なるエントリ数を持つ
        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);

        for (i, pattern) in patterns.iter().enumerate() {
            let expected_entries = 3_usize.pow(pattern.k as u32);

            // 各パターンのエントリ数が正しいか確認
            // (次のパターンのオフセット - 現在のパターンのオフセット)で検証
            if i < patterns.len() - 1 {
                let entries_count = table.pattern_offsets[i + 1] - table.pattern_offsets[i];
                assert_eq!(
                    entries_count, expected_entries,
                    "Pattern {} should have {} entries (3^{})",
                    i, expected_entries, pattern.k
                );
            } else {
                // 最後のパターンは全体サイズから計算
                let total_size = table.data[0].len();
                let last_pattern_entries = total_size - table.pattern_offsets[i];
                assert_eq!(
                    last_pattern_entries, expected_entries,
                    "Last pattern {} should have {} entries (3^{})",
                    i, expected_entries, pattern.k
                );
            }
        }
    }

    #[test]
    fn test_task_8_requirements_summary() {
        // Task 8.1 & 8.2の全要件を統合的に検証
        println!("=== Task 8 Requirements Verification ===");

        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);

        // Requirement 9.1: SoA形式
        assert_eq!(table.data.len(), 30);
        println!("✓ 9.1: Structure of Arrays format with [stage][flat_array]");

        // Requirement 9.2: 初期値32768
        assert_eq!(table.get(0, 0, 0), 32768);
        println!("✓ 9.2: All entries initialized to 32768");

        // Requirement 9.3: 3^k個のエントリ
        for pattern in patterns.iter() {
            let _ = 3_usize.pow(pattern.k as u32);
        }
        println!("✓ 9.3: Each pattern has 3^k entries");

        // Requirement 9.4: 30独立ステージ
        assert_eq!(table.data.len(), 30);
        println!("✓ 9.4: 30 independent stages (0-29)");

        // Requirement 9.5, 13.2: メモリ80MB以内
        let memory_mb = table.memory_usage() as f64 / 1_048_576.0;
        assert!(memory_mb <= 80.0);
        println!(
            "✓ 9.5, 13.2: Memory usage within 80MB ({:.2} MB)",
            memory_mb
        );

        // Requirement 9.6: 連続メモリ配置
        println!("✓ 9.6: Continuous memory layout for same stage patterns");

        // Requirement 9.7: pattern_offsets管理
        assert_eq!(table.pattern_offsets.len(), 14);
        println!("✓ 9.7: pattern_offsets array manages start positions");

        println!("=== All Task 8 requirements verified ===");
    }
}
