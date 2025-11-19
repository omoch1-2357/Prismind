//! Display function demonstration
//!
//! This example demonstrates the BitBoard display function with and without legal moves.

use prismind::board::{display, legal_moves, make_move, BitBoard};

fn main() {
    println!("=== BitBoard Display Function Demo ===\n");

    // Initial board
    let mut board = BitBoard::new();
    println!("Initial Board:");
    println!("{}", display(&board, false));

    println!("\nInitial Board with Legal Moves:");
    println!("{}", display(&board, true));

    // Make a move
    let moves = legal_moves(&board);
    if moves != 0 {
        let first_move = moves.trailing_zeros() as u8;
        println!("\nMaking move at position {}", first_move);
        make_move(&mut board, first_move).unwrap();

        println!("\nAfter First Move:");
        println!("{}", display(&board, false));

        println!("\nAfter First Move with Legal Moves:");
        println!("{}", display(&board, true));
    }
}
