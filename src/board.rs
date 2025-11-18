//! BitBoard盤面表現とColor型定義

/// 石の色を表す列挙型
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    /// 黒石
    Black = 0,
    /// 白石
    White = 1,
}

impl Color {
    /// 反対の色を返す
    ///
    /// # Examples
    ///
    /// ```
    /// use prismind::board::Color;
    ///
    /// assert_eq!(Color::Black.opposite(), Color::White);
    /// assert_eq!(Color::White.opposite(), Color::Black);
    /// ```
    #[inline]
    pub fn opposite(self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }
}

/// オセロ盤面を表すBitBoard構造体
///
/// 黒石と白石をそれぞれu64のビットマスクで表現する。
/// A1=bit 0, B1=bit 1, ..., H8=bit 63のマッピング。
/// メモリレイアウト: 正確に16バイト（手番情報を上位ビットに埋め込み）
#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BitBoard {
    /// 黒石の配置ビットマスク
    pub black: u64,
    /// 白石の配置ビットマスク（最下位1ビット: 手番, 次の7ビット: move_count）
    /// 実際の白石データは上位56ビットに格納
    white: u64,
}

impl std::fmt::Debug for BitBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitBoard")
            .field("black", &format_args!("{:#018x}", self.black))
            .field("white", &format_args!("{:#018x}", self.white_mask()))
            .field("turn", &self.turn())
            .field("move_count", &self.move_count())
            .finish()
    }
}

const TURN_MASK: u64 = 0x01;
const MOVE_COUNT_MASK: u64 = 0xFE;
const WHITE_MASK: u64 = !0xFF;

impl BitBoard {
    /// 初期盤面を生成
    ///
    /// D4白、E4黒、D5黒、E5白の標準オセロ初期配置を設定する。
    ///
    /// # Examples
    ///
    /// ```
    /// use prismind::board::BitBoard;
    ///
    /// let board = BitBoard::new();
    /// assert_eq!(board.black.count_ones(), 2);
    /// assert_eq!(board.white_mask().count_ones(), 2);
    /// ```
    pub fn new() -> Self {
        // 初期配置: D4白(27)、E4黒(28)、D5黒(35)、E5白(36)
        // A1=0, B1=1, ..., H8=63（行優先）
        let black = (1u64 << 28) | (1u64 << 35);  // E4, D5
        let white_stones = (1u64 << 27) | (1u64 << 36);  // D4, E5

        Self {
            black,
            white: white_stones,  // 手番=Black(0), move_count=0がデフォルトで0
        }
    }

    /// 白石のビットマスクを取得（メタデータを除く）
    #[inline]
    pub fn white_mask(&self) -> u64 {
        self.white & WHITE_MASK
    }

    /// 現在の手番を取得
    #[inline]
    pub fn turn(&self) -> Color {
        if (self.white & TURN_MASK) == 0 {
            Color::Black
        } else {
            Color::White
        }
    }

    /// 手数カウンタを取得
    #[inline]
    pub fn move_count(&self) -> u8 {
        ((self.white & MOVE_COUNT_MASK) >> 1) as u8
    }

    /// 現在の手番のビットマスクを取得
    #[inline]
    pub fn current_player(&self) -> u64 {
        if (self.white & TURN_MASK) == 0 {
            self.black
        } else {
            self.white_mask()
        }
    }

    /// 相手の手番のビットマスクを取得
    #[inline]
    pub fn opponent(&self) -> u64 {
        if (self.white & TURN_MASK) == 0 {
            self.white_mask()
        } else {
            self.black
        }
    }

    /// 盤面を白黒反転
    ///
    /// 黒石と白石を入れ替え、手番も反転する。
    ///
    /// # Examples
    ///
    /// ```
    /// use prismind::board::{BitBoard, Color};
    ///
    /// let board = BitBoard::new();
    /// let flipped = board.flip();
    /// assert_eq!(flipped.turn(), Color::White);
    /// ```
    pub fn flip(&self) -> Self {
        let turn = (self.white & TURN_MASK) ^ 1;
        let move_count = self.white & MOVE_COUNT_MASK;
        Self {
            black: self.white_mask(),
            white: self.black | turn | move_count,
        }
    }
}

impl Default for BitBoard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_opposite() {
        assert_eq!(Color::Black.opposite(), Color::White);
        assert_eq!(Color::White.opposite(), Color::Black);
    }

    #[test]
    fn test_bitboard_size() {
        // BitBoardは16バイト以内
        assert!(std::mem::size_of::<BitBoard>() <= 16);
    }

    #[test]
    fn test_bitboard_default_traits() {
        let board1 = BitBoard::new();
        let board2 = board1; // Copy
        let board3 = board1.clone(); // Clone

        assert_eq!(board1, board2); // PartialEq, Eq
        assert_eq!(board2, board3);

        // Debug
        let debug_str = format!("{:?}", board1);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_initial_board() {
        // 初期盤面の正当性を検証
        let board = BitBoard::new();

        // 初期配置: D4白(27)、E4黒(28)、D5黒(35)、E5白(36)
        // A1=0, B1=1, ..., H1=7, A2=8, ..., D4=27, E4=28
        assert_eq!(board.black.count_ones(), 2, "黒石は2個");
        assert_eq!(board.white_mask().count_ones(), 2, "白石は2個");

        // D4 = row3 * 8 + col3 = 3*8 + 3 = 27
        // E4 = row3 * 8 + col4 = 3*8 + 4 = 28
        // D5 = row4 * 8 + col3 = 4*8 + 3 = 35
        // E5 = row4 * 8 + col4 = 4*8 + 4 = 36
        assert_eq!(board.black & (1 << 28), 1 << 28, "E4に黒石");
        assert_eq!(board.black & (1 << 35), 1 << 35, "D5に黒石");
        assert_eq!(board.white_mask() & (1 << 27), 1 << 27, "D4に白石");
        assert_eq!(board.white_mask() & (1 << 36), 1 << 36, "E5に白石");

        // 初期手番は黒
        assert_eq!(board.turn(), Color::Black);
        assert_eq!(board.move_count(), 0);
    }

    #[test]
    fn test_current_player_and_opponent() {
        let board = BitBoard::new();

        // 黒の手番
        assert_eq!(board.current_player(), board.black);
        assert_eq!(board.opponent(), board.white_mask());

        // 白の手番
        let flipped = board.flip();
        assert_eq!(flipped.current_player(), flipped.white_mask());
        assert_eq!(flipped.opponent(), flipped.black);
    }

    #[test]
    fn test_flip() {
        let board = BitBoard::new();
        let flipped = board.flip();

        // 手番が反転
        assert_eq!(flipped.turn(), Color::White);

        // 黒と白が入れ替わる
        assert_eq!(flipped.black, board.white_mask());
        assert_eq!(flipped.white_mask(), board.black);

        // 手数は保持される
        assert_eq!(flipped.move_count(), board.move_count());
    }
}
