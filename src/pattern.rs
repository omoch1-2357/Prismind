//! パターン定義の読み込みと管理
//!
//! オセロAIの評価関数で使用する14パターンの定義を管理する。
//! patterns.csvファイルから読み込み、固定長配列でメモリ効率的に保持する。

use serde::Deserialize;
use std::path::Path;
use thiserror::Error;

/// Pattern定義のエラー型
#[derive(Error, Debug, PartialEq, Eq)]
pub enum PatternError {
    /// patterns.csvの読み込みエラー
    #[error("Failed to load patterns.csv: {0}")]
    LoadError(String),

    /// パターンの座標が範囲外（0-63以外）
    #[error("Invalid pattern position: {0}")]
    InvalidPosition(u8),

    /// パターン数の不一致（14個でない）
    #[error("Pattern count mismatch: expected 14, found {0}")]
    CountMismatch(usize),

    /// I/Oエラー
    #[error("I/O error: {0}")]
    IoError(String),

    /// CSVパースエラー
    #[error("CSV parse error: {0}")]
    CsvError(String),
}

/// パターン構造体
///
/// 固定長配列を使用してヒープアロケーションを回避し、
/// キャッシュ効率を最適化する。
///
/// # メモリレイアウト
///
/// - `id`: パターンID（0-13、P01-P14に対応）
/// - `k`: セル数（パターンに含まれるマスの数）
/// - `_padding`: アライメント調整用パディング
/// - `positions`: セル位置の配列（最大10個、0-63の範囲）
///
/// 総サイズ: 24バイト（スタック配置）
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pattern {
    /// パターンID（0-13）
    pub id: u8,
    /// セル数（パターンに含まれるマスの数）
    pub k: u8,
    /// アライメント調整用パディング
    _padding: [u8; 6],
    /// セル位置の配列（最大10個、0-63の範囲）
    pub positions: [u8; 10],
}

/// CSV読み込み用の中間構造体
#[derive(Debug, Deserialize)]
struct PatternCsv {
    id: String,
    k: u8,
    positions: String,
}

impl Pattern {
    /// 新しいPattern構造体を作成
    ///
    /// # Arguments
    ///
    /// * `id` - パターンID（0-13）
    /// * `k` - セル数
    /// * `positions` - セル位置の配列（最大10個）
    ///
    /// # Errors
    ///
    /// セル位置が0-63の範囲外の場合、`PatternError::InvalidPosition`を返す。
    pub fn new(id: u8, k: u8, positions: Vec<u8>) -> Result<Self, PatternError> {
        // 座標の範囲チェック
        for &pos in &positions {
            if pos >= 64 {
                return Err(PatternError::InvalidPosition(pos));
            }
        }

        // 固定長配列に変換（最大10個）
        let mut pos_array = [0u8; 10];
        for (i, &pos) in positions.iter().take(10).enumerate() {
            pos_array[i] = pos;
        }

        Ok(Self {
            id,
            k,
            _padding: [0; 6],
            positions: pos_array,
        })
    }
}

/// 座標文字列をビット位置に変換
///
/// オセロの座標表記（A1-H8）をビット位置（0-63）に変換する。
///
/// # 座標マッピング
///
/// - A1 = 0, B1 = 1, ..., H1 = 7
/// - A2 = 8, B2 = 9, ..., H2 = 15
/// - ...
/// - A8 = 56, B8 = 57, ..., H8 = 63
///
/// # Arguments
///
/// * `coord` - 座標文字列（例: "A1", "H8"）
///
/// # Returns
///
/// ビット位置（0-63）
///
/// # Errors
///
/// 座標形式が不正な場合、`PatternError::LoadError`を返す。
///
/// # Examples
///
/// ```
/// use prismind::pattern::coord_to_bit;
///
/// assert_eq!(coord_to_bit("A1").unwrap(), 0);
/// assert_eq!(coord_to_bit("H8").unwrap(), 63);
/// assert_eq!(coord_to_bit("E4").unwrap(), 28);
/// ```
pub fn coord_to_bit(coord: &str) -> Result<u8, PatternError> {
    let coord = coord.trim();
    if coord.len() != 2 {
        return Err(PatternError::LoadError(format!(
            "Invalid coordinate format: {}",
            coord
        )));
    }

    let bytes = coord.as_bytes();
    let col = bytes[0];
    let row = bytes[1];

    // 列（A-H）を0-7に変換
    let col_idx = match col {
        b'A'..=b'H' => col - b'A',
        b'a'..=b'h' => col - b'a',
        _ => {
            return Err(PatternError::LoadError(format!(
                "Invalid column: {}",
                col as char
            )))
        }
    };

    // 行（1-8）を0-7に変換
    let row_idx = match row {
        b'1'..=b'8' => row - b'1',
        _ => {
            return Err(PatternError::LoadError(format!(
                "Invalid row: {}",
                row as char
            )))
        }
    };

    // ビット位置を計算: row * 8 + col
    Ok(row_idx * 8 + col_idx)
}

/// patterns.csvからパターン定義を読み込む
///
/// CSVファイルから14パターンの定義を読み込み、Pattern構造体の配列として返す。
///
/// # CSV形式
///
/// ```csv
/// id,k,positions
/// P01,10,A1 B1 C1 D1 E1 F1 G1 H1 A2 B2
/// P02,10,A1 A2 A3 A4 A5 A6 A7 A8 B1 B2
/// ...
/// ```
///
/// # Arguments
///
/// * `path` - patterns.csvファイルのパス
///
/// # Returns
///
/// 14個のPattern構造体を含むVec
///
/// # Errors
///
/// - ファイルが存在しない場合、`PatternError::IoError`
/// - パターン数が14でない場合、`PatternError::CountMismatch`
/// - 座標が範囲外の場合、`PatternError::InvalidPosition`
///
/// # Examples
///
/// ```no_run
/// use prismind::pattern::load_patterns;
///
/// let patterns = load_patterns("patterns.csv").unwrap();
/// assert_eq!(patterns.len(), 14);
/// ```
pub fn load_patterns<P: AsRef<Path>>(path: P) -> Result<Vec<Pattern>, PatternError> {
    // ファイル存在チェック
    let path_ref = path.as_ref();
    if !path_ref.exists() {
        return Err(PatternError::IoError(format!(
            "File not found: {}",
            path_ref.display()
        )));
    }

    // CSVリーダーを作成
    let mut reader = csv::Reader::from_path(path)
        .map_err(|e| PatternError::CsvError(format!("Failed to open CSV: {}", e)))?;

    let mut patterns = Vec::new();

    // CSVレコードを読み込み
    for (idx, result) in reader.deserialize().enumerate() {
        let record: PatternCsv = result.map_err(|e| {
            PatternError::CsvError(format!("Failed to parse CSV at line {}: {}", idx + 2, e))
        })?;

        // IDをu8に変換（P01 -> 0, P02 -> 1, ..., P14 -> 13）
        let id = if record.id.starts_with('P') || record.id.starts_with('p') {
            let num_str = &record.id[1..];
            num_str.parse::<u8>().map_err(|_| {
                PatternError::LoadError(format!("Invalid pattern ID: {}", record.id))
            })? - 1
        } else {
            return Err(PatternError::LoadError(format!(
                "Invalid pattern ID format: {}",
                record.id
            )));
        };

        // 座標文字列を分割してビット位置に変換
        let positions: Result<Vec<u8>, PatternError> = record
            .positions
            .split_whitespace()
            .map(coord_to_bit)
            .collect();

        let positions = positions?;

        // セル数の検証
        if positions.len() != record.k as usize {
            return Err(PatternError::LoadError(format!(
                "Pattern {}: k={} but {} positions provided",
                record.id,
                record.k,
                positions.len()
            )));
        }

        // Pattern構造体を作成
        let pattern = Pattern::new(id, record.k, positions)?;
        patterns.push(pattern);
    }

    // パターン数の検証
    if patterns.len() != 14 {
        return Err(PatternError::CountMismatch(patterns.len()));
    }

    // IDでソート（P01-P14の順序を保証）
    patterns.sort_by_key(|p| p.id);

    Ok(patterns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ========== Task 5.1: Pattern Structure and CSV Loading Tests (TDD - RED) ==========

    #[test]
    fn test_pattern_struct_size() {
        // Pattern構造体のサイズが24バイト以内であることを確認
        assert_eq!(
            std::mem::size_of::<Pattern>(),
            24,
            "Pattern should be exactly 24 bytes"
        );
    }

    #[test]
    fn test_pattern_struct_alignment() {
        // 8バイトアライメントが適用されていることを確認
        assert_eq!(
            std::mem::align_of::<Pattern>(),
            8,
            "Pattern should be 8-byte aligned"
        );
    }

    #[test]
    fn test_pattern_new_valid_positions() {
        // 有効な座標でPattern構造体を作成
        let positions = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let pattern = Pattern::new(0, 10, positions.clone());

        assert!(pattern.is_ok());
        let pattern = pattern.unwrap();
        assert_eq!(pattern.id, 0);
        assert_eq!(pattern.k, 10);
        assert_eq!(&pattern.positions[..10], &positions[..]);
    }

    #[test]
    fn test_pattern_new_invalid_position() {
        // 範囲外の座標（64以上）でエラーを返すことを確認
        let positions = vec![0, 1, 2, 64]; // 64 is out of bounds
        let result = Pattern::new(0, 4, positions);

        assert!(result.is_err());
        match result {
            Err(PatternError::InvalidPosition(pos)) => {
                assert_eq!(pos, 64);
            }
            _ => panic!("Expected InvalidPosition error"),
        }
    }

    #[test]
    fn test_pattern_fixed_array_no_heap() {
        // Pattern構造体がヒープアロケーションを使用しないことを確認
        // （固定長配列のため、スタック上に配置される）
        let positions = vec![0, 1, 2, 3, 4];
        let pattern = Pattern::new(0, 5, positions).unwrap();

        // Copyトレイトが実装されていることを確認（ヒープがないことの間接的証明）
        let _copy = pattern;
        let _another_copy = pattern; // If this compiles, it's Copy
    }

    #[test]
    fn test_coord_to_bit_corner_cases() {
        // A1 = 0
        assert_eq!(coord_to_bit("A1").unwrap(), 0);

        // H1 = 7
        assert_eq!(coord_to_bit("H1").unwrap(), 7);

        // A8 = 56
        assert_eq!(coord_to_bit("A8").unwrap(), 56);

        // H8 = 63
        assert_eq!(coord_to_bit("H8").unwrap(), 63);
    }

    #[test]
    fn test_coord_to_bit_middle_positions() {
        // E4 = row3 * 8 + col4 = 3 * 8 + 4 = 28
        assert_eq!(coord_to_bit("E4").unwrap(), 28);

        // D5 = row4 * 8 + col3 = 4 * 8 + 3 = 35
        assert_eq!(coord_to_bit("D5").unwrap(), 35);

        // D4 = row3 * 8 + col3 = 3 * 8 + 3 = 27
        assert_eq!(coord_to_bit("D4").unwrap(), 27);
    }

    #[test]
    fn test_coord_to_bit_lowercase() {
        // 小文字の座標表記もサポート
        assert_eq!(coord_to_bit("a1").unwrap(), 0);
        assert_eq!(coord_to_bit("h8").unwrap(), 63);
        assert_eq!(coord_to_bit("e4").unwrap(), 28);
    }

    #[test]
    fn test_coord_to_bit_invalid_format() {
        // 不正な形式でエラーを返す
        assert!(coord_to_bit("A").is_err()); // Too short
        assert!(coord_to_bit("ABC").is_err()); // Too long
        assert!(coord_to_bit("I1").is_err()); // Invalid column
        assert!(coord_to_bit("A9").is_err()); // Invalid row
        assert!(coord_to_bit("11").is_err()); // No letter
        assert!(coord_to_bit("AA").is_err()); // No digit
    }

    #[test]
    fn test_coord_to_bit_all_positions() {
        // 全64マスの座標変換が正しいことを確認
        for row in 0..8 {
            for col in 0..8 {
                let coord_str = format!("{}{}", (b'A' + col) as char, (b'1' + row) as char);
                let expected = row * 8 + col;
                assert_eq!(
                    coord_to_bit(&coord_str).unwrap(),
                    expected,
                    "Failed for coord {}",
                    coord_str
                );
            }
        }
    }

    #[test]
    fn test_load_patterns_file_not_found() {
        // 存在しないファイルでエラーを返す
        let result = load_patterns("nonexistent.csv");
        assert!(result.is_err());

        match result {
            Err(PatternError::IoError(msg)) => {
                assert!(msg.contains("File not found"));
            }
            _ => panic!("Expected IoError"),
        }
    }

    #[test]
    fn test_load_patterns_valid_csv() {
        // 有効なCSVファイルを読み込む
        let csv_content = "\
id,k,positions
P01,10,A1 B1 C1 D1 E1 F1 G1 H1 A2 B2
P02,10,A1 A2 A3 A4 A5 A6 A7 A8 B1 B2
P03,10,A1 B1 A2 B2 C1 D1 C2 D2 E1 F1
P04,10,A1 B2 C3 D4 E5 F6 G7 H8 A2 B3
P05,8,A1 B1 C1 D1 E1 F1 G1 H1
P06,8,A1 A2 A3 A4 A5 A6 A7 A8
P07,8,A1 B2 C3 D4 E5 F6 G7 H8
P08,8,H1 G2 F3 E4 D5 C6 B7 A8
P09,6,A1 B1 C1 D1 E1 F1
P10,6,A1 A2 A3 A4 A5 A6
P11,5,A1 B1 C1 D1 E1
P12,5,A1 A2 A3 A4 A5
P13,4,A1 B1 C1 D1
P14,4,A1 A2 A3 A4
";

        // 一時ファイルを作成
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_patterns.csv");
        let mut file = std::fs::File::create(&temp_file).unwrap();
        file.write_all(csv_content.as_bytes()).unwrap();
        drop(file);

        // パターンを読み込み
        let result = load_patterns(&temp_file);
        assert!(result.is_ok(), "Should successfully load patterns");

        let patterns = result.unwrap();
        assert_eq!(patterns.len(), 14, "Should have 14 patterns");

        // 最初のパターンを検証
        let p01 = &patterns[0];
        assert_eq!(p01.id, 0);
        assert_eq!(p01.k, 10);
        assert_eq!(p01.positions[0], 0); // A1
        assert_eq!(p01.positions[1], 1); // B1
        assert_eq!(p01.positions[7], 7); // H1

        // クリーンアップ
        std::fs::remove_file(&temp_file).unwrap();
    }

    #[test]
    fn test_load_patterns_wrong_count() {
        // パターン数が14でない場合にエラーを返す
        let csv_content = "\
id,k,positions
P01,10,A1 B1 C1 D1 E1 F1 G1 H1 A2 B2
P02,10,A1 A2 A3 A4 A5 A6 A7 A8 B1 B2
";

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_patterns_wrong_count.csv");
        let mut file = std::fs::File::create(&temp_file).unwrap();
        file.write_all(csv_content.as_bytes()).unwrap();
        drop(file);

        let result = load_patterns(&temp_file);
        assert!(result.is_err());

        match result {
            Err(PatternError::CountMismatch(count)) => {
                assert_eq!(count, 2);
            }
            _ => panic!("Expected CountMismatch error"),
        }

        std::fs::remove_file(&temp_file).unwrap();
    }

    #[test]
    fn test_load_patterns_invalid_position() {
        // 座標が範囲外の場合にエラーを返す
        let csv_content = "\
id,k,positions
P01,3,A1 B1 I9
";

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_patterns_invalid.csv");
        let mut file = std::fs::File::create(&temp_file).unwrap();
        file.write_all(csv_content.as_bytes()).unwrap();
        drop(file);

        let result = load_patterns(&temp_file);
        assert!(result.is_err());

        // I9 is an invalid coordinate
        match result {
            Err(PatternError::LoadError(_)) | Err(PatternError::CsvError(_)) => {}
            other => panic!("Expected LoadError or CsvError, got {:?}", other),
        }

        std::fs::remove_file(&temp_file).unwrap();
    }

    #[test]
    fn test_load_patterns_sorted_by_id() {
        // パターンがIDでソートされていることを確認
        let csv_content = "\
id,k,positions
P14,4,A1 A2 A3 A4
P01,10,A1 B1 C1 D1 E1 F1 G1 H1 A2 B2
P02,10,A1 A2 A3 A4 A5 A6 A7 A8 B1 B2
P03,10,A1 B1 A2 B2 C1 D1 C2 D2 E1 F1
P04,10,A1 B2 C3 D4 E5 F6 G7 H8 A2 B3
P05,8,A1 B1 C1 D1 E1 F1 G1 H1
P06,8,A1 A2 A3 A4 A5 A6 A7 A8
P07,8,A1 B2 C3 D4 E5 F6 G7 H8
P08,8,H1 G2 F3 E4 D5 C6 B7 A8
P09,6,A1 B1 C1 D1 E1 F1
P10,6,A1 A2 A3 A4 A5 A6
P11,5,A1 B1 C1 D1 E1
P12,5,A1 A2 A3 A4 A5
P13,4,A1 B1 C1 D1
";

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_patterns_sorted.csv");
        let mut file = std::fs::File::create(&temp_file).unwrap();
        file.write_all(csv_content.as_bytes()).unwrap();
        drop(file);

        let patterns = load_patterns(&temp_file).unwrap();

        // IDが0-13の順になっていることを確認
        for (i, pattern) in patterns.iter().enumerate() {
            assert_eq!(pattern.id, i as u8, "Pattern ID should be sequential");
        }

        std::fs::remove_file(&temp_file).unwrap();
    }

    #[test]
    fn test_pattern_requirements_summary() {
        // Task 5.1の全要件を統合的に検証
        println!("=== Task 5.1 Requirements Verification ===");

        // Requirement 6.1: Pattern構造体が#[repr(C, align(8))]で定義されている
        assert_eq!(std::mem::align_of::<Pattern>(), 8);
        println!("✓ 6.1: Pattern struct has 8-byte alignment");

        // Requirement 6.2: id（u8）、k（u8）、positions（[u8; 10]）フィールド
        let pattern = Pattern::new(0, 5, vec![0, 1, 2, 3, 4]).unwrap();
        assert_eq!(std::mem::size_of_val(&pattern.id), 1);
        assert_eq!(std::mem::size_of_val(&pattern.k), 1);
        assert_eq!(std::mem::size_of_val(&pattern.positions), 10);
        println!("✓ 6.2: Pattern has correct field types");

        // Requirement 6.4: 座標→ビット位置変換（A1=0、H8=63）
        assert_eq!(coord_to_bit("A1").unwrap(), 0);
        assert_eq!(coord_to_bit("H8").unwrap(), 63);
        println!("✓ 6.4: Coordinate to bit position conversion works");

        // Requirement 6.6: 固定長配列でヒープアロケーション回避
        assert_eq!(std::mem::size_of::<Pattern>(), 24);
        println!("✓ 6.6 & 13.6: Fixed-size array, no heap allocation");

        println!("=== All Task 5.1 requirements verified ===");
    }

    // ========== Task 5.2: Pattern Validation and Error Handling Tests (TDD - RED) ==========

    #[test]
    fn test_pattern_position_validation_boundary() {
        // Requirement 6.3: セル位置が0-63の範囲内であることを検証
        // 境界値テスト: 0 (有効)
        let result = Pattern::new(0, 1, vec![0]);
        assert!(result.is_ok(), "Position 0 should be valid");

        // 境界値テスト: 63 (有効)
        let result = Pattern::new(0, 1, vec![63]);
        assert!(result.is_ok(), "Position 63 should be valid");

        // 境界値テスト: 64 (無効 - 範囲外)
        let result = Pattern::new(0, 1, vec![64]);
        assert!(result.is_err(), "Position 64 should be invalid");
        match result {
            Err(PatternError::InvalidPosition(pos)) => {
                assert_eq!(pos, 64, "Error should report position 64");
            }
            _ => panic!("Expected InvalidPosition error for position 64"),
        }

        // 境界値テスト: 255 (無効 - u8の最大値)
        let result = Pattern::new(0, 1, vec![255]);
        assert!(result.is_err(), "Position 255 should be invalid");
        match result {
            Err(PatternError::InvalidPosition(pos)) => {
                assert_eq!(pos, 255, "Error should report position 255");
            }
            _ => panic!("Expected InvalidPosition error for position 255"),
        }
    }

    #[test]
    fn test_pattern_position_validation_multiple_invalid() {
        // Requirement 6.3: 複数のセル位置を検証し、最初の無効位置でエラーを返す
        let result = Pattern::new(0, 5, vec![10, 20, 30, 64, 70]);
        assert!(result.is_err(), "Should detect first invalid position");
        match result {
            Err(PatternError::InvalidPosition(pos)) => {
                assert_eq!(pos, 64, "Should report first invalid position");
            }
            _ => panic!("Expected InvalidPosition error"),
        }
    }

    #[test]
    fn test_pattern_position_validation_all_valid() {
        // Requirement 6.3: 全てのセル位置が0-63の範囲内であることを検証
        let positions = vec![0, 7, 8, 15, 27, 28, 35, 36, 56, 63];
        let result = Pattern::new(0, 10, positions.clone());
        assert!(result.is_ok(), "All positions 0-63 should be valid");

        let pattern = result.unwrap();
        for (i, &expected_pos) in positions.iter().enumerate() {
            assert_eq!(
                pattern.positions[i], expected_pos,
                "Position {} should be correctly stored",
                i
            );
        }
    }

    #[test]
    fn test_load_patterns_count_validation() {
        // Requirement 6.4: パターン数が14個であることを確認
        // 13個のパターン（14未満）
        let csv_content_13 = "\
id,k,positions
P01,10,A1 B1 C1 D1 E1 F1 G1 H1 A2 B2
P02,10,A1 A2 A3 A4 A5 A6 A7 A8 B1 B2
P03,10,A1 B1 A2 B2 C1 D1 C2 D2 E1 F1
P04,10,A1 B2 C3 D4 E5 F6 G7 H8 A2 B3
P05,8,A1 B1 C1 D1 E1 F1 G1 H1
P06,8,A1 A2 A3 A4 A5 A6 A7 A8
P07,8,A1 B2 C3 D4 E5 F6 G7 H8
P08,8,H1 G2 F3 E4 D5 C6 B7 A8
P09,6,A1 B1 C1 D1 E1 F1
P10,6,A1 A2 A3 A4 A5 A6
P11,5,A1 B1 C1 D1 E1
P12,5,A1 A2 A3 A4 A5
P13,4,A1 B1 C1 D1
";

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_patterns_13.csv");
        let mut file = std::fs::File::create(&temp_file).unwrap();
        file.write_all(csv_content_13.as_bytes()).unwrap();
        drop(file);

        let result = load_patterns(&temp_file);
        assert!(result.is_err(), "Should reject 13 patterns");
        match result {
            Err(PatternError::CountMismatch(count)) => {
                assert_eq!(count, 13, "Should report count as 13");
            }
            _ => panic!("Expected CountMismatch error"),
        }
        std::fs::remove_file(&temp_file).unwrap();

        // 15個のパターン（14超過）
        let csv_content_15 = "\
id,k,positions
P01,10,A1 B1 C1 D1 E1 F1 G1 H1 A2 B2
P02,10,A1 A2 A3 A4 A5 A6 A7 A8 B1 B2
P03,10,A1 B1 A2 B2 C1 D1 C2 D2 E1 F1
P04,10,A1 B2 C3 D4 E5 F6 G7 H8 A2 B3
P05,8,A1 B1 C1 D1 E1 F1 G1 H1
P06,8,A1 A2 A3 A4 A5 A6 A7 A8
P07,8,A1 B2 C3 D4 E5 F6 G7 H8
P08,8,H1 G2 F3 E4 D5 C6 B7 A8
P09,6,A1 B1 C1 D1 E1 F1
P10,6,A1 A2 A3 A4 A5 A6
P11,5,A1 B1 C1 D1 E1
P12,5,A1 A2 A3 A4 A5
P13,4,A1 B1 C1 D1
P14,4,A1 A2 A3 A4
P15,4,B1 B2 B3 B4
";

        let temp_file = temp_dir.join("test_patterns_15.csv");
        let mut file = std::fs::File::create(&temp_file).unwrap();
        file.write_all(csv_content_15.as_bytes()).unwrap();
        drop(file);

        let result = load_patterns(&temp_file);
        assert!(result.is_err(), "Should reject 15 patterns");
        match result {
            Err(PatternError::CountMismatch(count)) => {
                assert_eq!(count, 15, "Should report count as 15");
            }
            _ => panic!("Expected CountMismatch error"),
        }
        std::fs::remove_file(&temp_file).unwrap();
    }

    #[test]
    fn test_pattern_error_types_comprehensive() {
        // Requirement 6.5: PatternErrorカスタムエラー型を定義し、全バリアントが機能することを確認
        // InvalidPosition
        let err = PatternError::InvalidPosition(100);
        assert_eq!(err.to_string(), "Invalid pattern position: 100");

        // CountMismatch
        let err = PatternError::CountMismatch(10);
        assert_eq!(
            err.to_string(),
            "Pattern count mismatch: expected 14, found 10"
        );

        // LoadError
        let err = PatternError::LoadError("test error".to_string());
        assert_eq!(err.to_string(), "Failed to load patterns.csv: test error");

        // IoError
        let err = PatternError::IoError("file not found".to_string());
        assert_eq!(err.to_string(), "I/O error: file not found");

        // CsvError
        let err = PatternError::CsvError("parse error".to_string());
        assert_eq!(err.to_string(), "CSV parse error: parse error");
    }

    #[test]
    fn test_file_not_found_error_message() {
        // Requirement 6.5: ファイル不存在で明確なエラーメッセージを表示
        let nonexistent_path = "definitely_does_not_exist_12345.csv";
        let result = load_patterns(nonexistent_path);

        assert!(result.is_err(), "Should fail for non-existent file");
        match result {
            Err(PatternError::IoError(msg)) => {
                assert!(
                    msg.contains("File not found"),
                    "Error message should mention 'File not found'"
                );
                assert!(
                    msg.contains(nonexistent_path),
                    "Error message should include the file path"
                );
            }
            _ => panic!("Expected IoError for non-existent file"),
        }
    }

    #[test]
    fn test_csv_format_error_handling() {
        // Requirement 6.5: CSV形式エラーで明確なエラーメッセージを表示
        // 不正なCSVヘッダー
        let csv_bad_header = "\
wrong,header,format
P01,10,A1 B1 C1 D1 E1 F1 G1 H1 A2 B2
";

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_bad_header.csv");
        let mut file = std::fs::File::create(&temp_file).unwrap();
        file.write_all(csv_bad_header.as_bytes()).unwrap();
        drop(file);

        let result = load_patterns(&temp_file);
        assert!(result.is_err(), "Should fail for invalid CSV format");
        // CsvErrorまたはLoadErrorが返される
        match result {
            Err(PatternError::CsvError(_)) | Err(PatternError::LoadError(_)) => {
                // Success - appropriate error type
            }
            other => panic!("Expected CsvError or LoadError, got {:?}", other),
        }
        std::fs::remove_file(&temp_file).unwrap();

        // 不正なID形式
        let csv_bad_id = "\
id,k,positions
X01,10,A1 B1 C1 D1 E1 F1 G1 H1 A2 B2
";

        let temp_file = temp_dir.join("test_bad_id.csv");
        let mut file = std::fs::File::create(&temp_file).unwrap();
        file.write_all(csv_bad_id.as_bytes()).unwrap();
        drop(file);

        let result = load_patterns(&temp_file);
        assert!(result.is_err(), "Should fail for invalid ID format");
        match result {
            Err(PatternError::LoadError(msg)) => {
                assert!(
                    msg.contains("Invalid pattern ID"),
                    "Error message should mention invalid ID"
                );
            }
            other => panic!("Expected LoadError for invalid ID, got {:?}", other),
        }
        std::fs::remove_file(&temp_file).unwrap();
    }

    #[test]
    fn test_pattern_memory_layout_no_heap() {
        // Requirement NFR-4, 13.6: メモリレイアウト確認（ヒープアロケーション回避）
        // Pattern構造体がCopyトレイトを持つことを確認（ヒープがないことの証明）
        let pattern1 = Pattern::new(0, 5, vec![0, 1, 2, 3, 4]).unwrap();
        let pattern2 = pattern1; // Copy occurs here
        let pattern3 = pattern1; // Another copy

        // 全て独立したコピーであることを確認
        assert_eq!(pattern1.id, pattern2.id);
        assert_eq!(pattern1.id, pattern3.id);
        assert_eq!(pattern1.k, pattern2.k);
        assert_eq!(pattern1.positions, pattern2.positions);

        // メモリサイズが固定であることを確認
        assert_eq!(std::mem::size_of::<Pattern>(), 24);

        // スタック配置の確認（ポインタのサイズではなく構造体自体のサイズ）
        let patterns = [pattern1, pattern2, pattern3];
        assert_eq!(std::mem::size_of_val(&patterns), 24 * 3);
    }

    #[test]
    fn test_pattern_validation_integration() {
        // Task 5.2の全要件を統合的に検証
        println!("=== Task 5.2 Requirements Verification ===");

        // Requirement 6.3: セル位置が0-63の範囲内であることを検証
        let valid_result = Pattern::new(0, 10, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert!(valid_result.is_ok());
        println!("✓ 6.3: Cell position validation (0-63 range)");

        let invalid_result = Pattern::new(0, 1, vec![64]);
        assert!(matches!(
            invalid_result,
            Err(PatternError::InvalidPosition(64))
        ));
        println!("✓ 6.3: Invalid position detection");

        // Requirement 6.4: パターン数が14個であることを確認
        // (既存のテストで検証済み - test_load_patterns_wrong_count)
        println!("✓ 6.4: Pattern count validation (14 patterns)");

        // Requirement 6.5: PatternErrorカスタムエラー型
        let _ = PatternError::InvalidPosition(100);
        let _ = PatternError::CountMismatch(10);
        let _ = PatternError::LoadError("test".to_string());
        let _ = PatternError::IoError("test".to_string());
        let _ = PatternError::CsvError("test".to_string());
        println!("✓ 6.5: PatternError custom error type with all variants");

        // Requirement NFR-4: エラーハンドリング
        let nonexistent = load_patterns("nonexistent.csv");
        assert!(nonexistent.is_err());
        println!("✓ NFR-4: Error handling for file not found");

        // Memory layout verification
        assert_eq!(std::mem::size_of::<Pattern>(), 24);
        assert_eq!(std::mem::align_of::<Pattern>(), 8);
        println!("✓ NFR-4, 13.6: Memory layout verified (no heap allocation)");

        println!("=== All Task 5.2 requirements verified ===");
    }
}
