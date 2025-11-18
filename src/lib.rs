//! Prismind - オセロAI基礎実装
//!
//! Phase 1: BitBoard盤面表現とパターン評価システム

pub mod board;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitboard_size() {
        // BitBoardは16バイト以内であること
        assert!(std::mem::size_of::<board::BitBoard>() <= 16);
    }

    #[test]
    fn test_color_opposite() {
        use board::Color;

        // 白の反対は黒
        assert_eq!(Color::White.opposite(), Color::Black);
        // 黒の反対は白
        assert_eq!(Color::Black.opposite(), Color::White);
    }

    #[test]
    fn test_bitboard_traits() {
        use board::{BitBoard, Color};

        // Clone, Copy, Debug, PartialEq, Eqトレイトが実装されていること
        let board1 = BitBoard::new();
        let board2 = board1; // Copy
        let board3 = board1;

        assert_eq!(board1, board2); // PartialEq
        assert_eq!(board2, board3); // Eq

        // 初期手番は黒
        assert_eq!(board1.turn(), Color::Black);

        // Debug
        let debug_str = format!("{:?}", board1);
        assert!(!debug_str.is_empty());
    }
}
