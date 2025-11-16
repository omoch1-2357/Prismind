# オセロAI技術詳細仕様書

本書は実装レベルの詳細な技術仕様を記載します。

---

## 1. BitBoard実装

### 1.1 盤面表現

```rust
struct Board {
    black: u64,   // 黒石の配置（ビットボード）
    white: u64,   // 白石の配置（ビットボード）
    turn: Color,  // 現在の手番
}

enum Color {
    Black = 0,
    White = 1,
}
```

### 1.2 座標とビット位置

```
A1=0,  B1=1,  C1=2,  ..., H1=7
A2=8,  B2=9,  C2=10, ..., H2=15
...
A8=56, B8=57, C8=58, ..., H8=63

ビットマスク例:
A1 = 1 << 0  = 0x0000000000000001
H8 = 1 << 63 = 0x8000000000000000
```

### 1.3 初期配置

```rust
初期盤面:
  D4(27): 白 = 1 << 27
  E4(28): 黒 = 1 << 28
  D5(35): 黒 = 1 << 35
  E5(36): 白 = 1 << 36

black = (1 << 28) | (1 << 35) = 0x0000001010000000
white = (1 << 27) | (1 << 36) = 0x0000001008000000
```

### 1.4 合法手生成

高速なビット演算アルゴリズムを使用:

```rust
fn legal_moves(board: &Board) -> u64 {
    let player = board.current_bitboard();
    let opponent = board.opponent_bitboard();
    let empty = !(player | opponent);
    
    let mut moves = 0u64;
    
    // 8方向それぞれについて
    for &dir in &DIRECTIONS {
        let mut candidates = shift(opponent, dir) & player;
        while candidates != 0 {
            candidates = shift(candidates, dir);
            moves |= candidates & empty;
            candidates &= opponent;
        }
    }
    
    moves
}

const DIRECTIONS: [i32; 8] = [
    -9, -8, -7,  // 左上、上、右上
    -1,      1,  // 左、右
     7,  8,  9,  // 左下、下、右下
];
```

### 1.5 石を返す処理

```rust
fn make_move(&mut self, pos: u8) {
    let bit = 1u64 << pos;
    let player = self.current_bitboard_mut();
    let opponent = self.opponent_bitboard_mut();
    
    *player |= bit;
    
    // 8方向それぞれで石を返す
    for &dir in &DIRECTIONS {
        let flipped = find_flipped_stones(bit, *player, *opponent, dir);
        *player |= flipped;
        *opponent &= !flipped;
    }
    
    self.turn = self.turn.opposite();
}
```

### 1.6 回転操作

4方向の回転を効率的に実装:

```rust
// 90度回転
fn rotate_90(board: u64) -> u64 {
    // ビット操作による効率的な実装
    // 例: A1(0) -> H1(7), B1(1) -> H2(15), ...
    let mut result = 0u64;
    for row in 0..8 {
        for col in 0..8 {
            let src_bit = row * 8 + col;
            let dst_bit = col * 8 + (7 - row);
            if board & (1 << src_bit) != 0 {
                result |= 1 << dst_bit;
            }
        }
    }
    result
}

// 180度回転（ビット反転）
fn rotate_180(board: u64) -> u64 {
    board.reverse_bits() >> 0  // ビット順序を反転
}

// 270度回転
fn rotate_270(board: u64) -> u64 {
    rotate_90(rotate_180(board))
}
```

---

## 2. パターン抽出

### 2.1 パターン定義の内部表現

```rust
struct Pattern {
    id: usize,              // P01=0, P02=1, ..., P14=13
    k: usize,               // セル数
    positions: Vec<u8>,     // セル位置のリスト（0-63）
}

// patterns.csvから読み込んで初期化
const PATTERNS: [Pattern; 14] = [...];
```

### 2.2 インデックス計算

```rust
fn extract_pattern_index(
    black: u64,
    white: u64,
    pattern: &Pattern,
    swap_colors: bool
) -> usize {
    let mut index = 0;
    let base = 3;
    
    for (i, &pos) in pattern.positions.iter().enumerate() {
        let bit = 1u64 << pos;
        let state = if black & bit != 0 {
            if swap_colors { 2 } else { 1 }  // 黒
        } else if white & bit != 0 {
            if swap_colors { 1 } else { 2 }  // 白
        } else {
            0  // 空
        };
        
        index += state * base.pow(i as u32);
    }
    
    index
}
```

### 2.3 4方向抽出の最適化

```rust
// 事前計算: 各パターンの各回転での位置
struct PrecomputedPatterns {
    // [pattern_id][rotation][cell_index] -> board_position
    positions: [[[u8; 10]; 4]; 14],
}

fn extract_all_pattern_indices(board: &Board) -> [usize; 56] {
    let mut indices = [0; 56];
    let mut idx = 0;
    
    for rotation in 0..4 {
        let (black_rot, white_rot) = board.rotate(rotation);
        let swap = rotation == 1 || rotation == 3;  // 90°または270°
        
        for pattern_id in 0..14 {
            indices[idx] = extract_pattern_index(
                black_rot,
                white_rot,
                &PATTERNS[pattern_id],
                swap
            );
            idx += 1;
        }
    }
    
    indices
}
```

---

## 3. 評価関数実装

### 3.1 データ構造

```rust
struct Evaluator {
    // [pattern_id][stage][index] -> u16
    tables: Vec<Vec<Vec<u16>>>,
}

impl Evaluator {
    fn new() -> Self {
        let mut tables = Vec::new();
        
        for pattern in &PATTERNS {
            let entries_per_stage = 3usize.pow(pattern.k as u32);
            let mut pattern_table = Vec::new();
            
            for _ in 0..30 {  // 30ステージ
                pattern_table.push(vec![32768u16; entries_per_stage]);
            }
            
            tables.push(pattern_table);
        }
        
        Self { tables }
    }
}
```

### 3.2 評価値計算

```rust
impl Evaluator {
    fn evaluate(&self, board: &Board) -> f32 {
        let stage = board.move_count() / 2;  // 0-29
        let indices = extract_all_pattern_indices(board);
        
        let mut sum = 0.0;
        for (i, &index) in indices.iter().enumerate() {
            let pattern_id = i % 14;
            let score_u16 = self.tables[pattern_id][stage][index];
            sum += u16_to_score(score_u16);
        }
        
        // 手番考慮
        if board.turn == Color::White {
            -sum
        } else {
            sum
        }
    }
}

fn u16_to_score(v: u16) -> f32 {
    (v as f32 - 32768.0) / 256.0
}

fn score_to_u16(s: f32) -> u16 {
    ((s * 256.0 + 32768.0).clamp(0.0, 65535.0)) as u16
}
```

---

## 4. MTD(f)探索実装

### 4.1 置換表

```rust
#[derive(Clone, Copy)]
enum Bound {
    Exact,
    Lower,  // α値（下限）
    Upper,  // β値（上限）
}

#[derive(Clone, Copy)]
struct TTEntry {
    hash: u64,
    depth: i8,
    bound: Bound,
    score: i16,
    best_move: u8,  // 0-63, 255=なし
    age: u8,        // 世代管理
}

struct TranspositionTable {
    entries: Vec<Option<TTEntry>>,
    size: usize,
    current_age: u8,
}

impl TranspositionTable {
    fn new(size_mb: usize) -> Self {
        let size = (size_mb * 1024 * 1024) / std::mem::size_of::<Option<TTEntry>>();
        Self {
            entries: vec![None; size],
            size,
            current_age: 0,
        }
    }
    
    fn probe(&self, hash: u64) -> Option<TTEntry> {
        let index = (hash as usize) % self.size;
        self.entries[index]
    }
    
    fn store(&mut self, hash: u64, entry: TTEntry) {
        let index = (hash as usize) % self.size;
        
        // 置換戦略: 常に上書き、または深さ・世代で判断
        if let Some(existing) = self.entries[index] {
            if existing.depth > entry.depth && 
               existing.age == self.current_age {
                return;  // より深い探索結果を優先
            }
        }
        
        self.entries[index] = Some(entry);
    }
}
```

### 4.2 Zobrist Hashing

```rust
struct ZobristTable {
    black: [u64; 64],
    white: [u64; 64],
    turn: u64,
}

impl ZobristTable {
    fn new() -> Self {
        use rand::{SeedableRng, Rng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(123456789);
        
        Self {
            black: [0; 64].map(|_| rng.gen()),
            white: [0; 64].map(|_| rng.gen()),
            turn: rng.gen(),
        }
    }
    
    fn hash(&self, board: &Board) -> u64 {
        let mut hash = 0u64;
        
        for pos in 0..64 {
            let bit = 1u64 << pos;
            if board.black & bit != 0 {
                hash ^= self.black[pos];
            } else if board.white & bit != 0 {
                hash ^= self.white[pos];
            }
        }
        
        if board.turn == Color::White {
            hash ^= self.turn;
        }
        
        hash
    }
}
```

### 4.3 AlphaBeta探索

```rust
fn alpha_beta(
    board: &mut Board,
    depth: i32,
    mut alpha: i32,
    beta: i32,
    evaluator: &Evaluator,
    tt: &mut TranspositionTable,
    zobrist: &ZobristTable,
) -> (i32, Option<u8>) {
    let hash = zobrist.hash(board);
    
    // 置換表プローブ
    if let Some(entry) = tt.probe(hash) {
        if entry.hash == hash && entry.depth >= depth {
            match entry.bound {
                Bound::Exact => return (entry.score as i32, Some(entry.best_move)),
                Bound::Lower => alpha = alpha.max(entry.score as i32),
                Bound::Upper => beta = beta.min(entry.score as i32),
            }
            if alpha >= beta {
                return (entry.score as i32, Some(entry.best_move));
            }
        }
    }
    
    // 深さ0または終端ノード
    if depth <= 0 || board.is_game_over() {
        let score = (evaluator.evaluate(board) * 100.0) as i32;
        return (score, None);
    }
    
    let moves = board.legal_moves();
    if moves == 0 {
        // パス
        board.pass();
        let (score, _) = alpha_beta(board, depth, -beta, -alpha, evaluator, tt, zobrist);
        board.undo_pass();
        return (-score, None);
    }
    
    // Move Ordering
    let ordered_moves = order_moves(moves, board, tt, hash);
    
    let mut best_move = None;
    let mut best_score = i32::MIN;
    
    for mv in ordered_moves {
        let undo_info = board.make_move(mv);
        let (score, _) = alpha_beta(board, depth - 1, -beta, -alpha, evaluator, tt, zobrist);
        let score = -score;
        board.undo_move(undo_info);
        
        if score > best_score {
            best_score = score;
            best_move = Some(mv);
        }
        
        alpha = alpha.max(score);
        if alpha >= beta {
            break;  // βカット
        }
    }
    
    // 置換表に保存
    let bound = if best_score <= alpha {
        Bound::Upper
    } else if best_score >= beta {
        Bound::Lower
    } else {
        Bound::Exact
    };
    
    tt.store(hash, TTEntry {
        hash,
        depth: depth as i8,
        bound,
        score: best_score as i16,
        best_move: best_move.unwrap_or(255),
        age: tt.current_age,
    });
    
    (best_score, best_move)
}
```

### 4.4 MTD(f)実装

```rust
fn mtdf(
    board: &mut Board,
    depth: i32,
    guess: i32,
    evaluator: &Evaluator,
    tt: &mut TranspositionTable,
    zobrist: &ZobristTable,
) -> (i32, Option<u8>) {
    let mut g = guess;
    let mut upper_bound = i32::MAX;
    let mut lower_bound = i32::MIN;
    let mut best_move = None;
    
    while lower_bound < upper_bound {
        let beta = if g == lower_bound { g + 1 } else { g };
        let (score, mv) = alpha_beta(board, depth, beta - 1, beta, evaluator, tt, zobrist);
        
        if let Some(m) = mv {
            best_move = Some(m);
        }
        
        if score < beta {
            upper_bound = score;
        } else {
            lower_bound = score;
        }
        
        g = score;
    }
    
    (g, best_move)
}
```

### 4.5 反復深化

```rust
fn iterative_deepening(
    board: &mut Board,
    time_limit_ms: u64,
    evaluator: &Evaluator,
    tt: &mut TranspositionTable,
    zobrist: &ZobristTable,
) -> (i32, u8) {
    let start = Instant::now();
    let mut best_move = 0u8;
    let mut best_score = 0i32;
    let mut guess = (evaluator.evaluate(board) * 100.0) as i32;
    
    for depth in 1..60 {
        if start.elapsed().as_millis() as u64 >= time_limit_ms {
            break;
        }
        
        let (score, mv) = mtdf(board, depth, guess, evaluator, tt, zobrist);
        
        if let Some(m) = mv {
            best_move = m;
            best_score = score;
            guess = score;
        }
        
        // 時間チェック
        if start.elapsed().as_millis() as u64 >= time_limit_ms * 80 / 100 {
            break;  // 80%使用で終了
        }
    }
    
    (best_score, best_move)
}
```

### 4.6 Move Ordering

```rust
fn order_moves(
    moves: u64,
    board: &Board,
    tt: &TranspositionTable,
    hash: u64,
) -> Vec<u8> {
    let mut move_list = Vec::new();
    
    for pos in 0..64 {
        if moves & (1 << pos) != 0 {
            move_list.push(pos);
        }
    }
    
    // 優先順位付け
    move_list.sort_by_key(|&pos| {
        let mut priority = 0;
        
        // 1. 置換表の最善手
        if let Some(entry) = tt.probe(hash) {
            if entry.best_move == pos {
                return -10000;
            }
        }
        
        // 2. 角
        if is_corner(pos) {
            priority -= 1000;
        }
        
        // 3. 角の隣（X打ち）を避ける
        if is_x_square(pos) {
            priority += 500;
        }
        
        // 4. 辺
        if is_edge(pos) {
            priority -= 100;
        }
        
        priority
    });
    
    move_list
}

fn is_corner(pos: u8) -> bool {
    matches!(pos, 0 | 7 | 56 | 63)
}

fn is_x_square(pos: u8) -> bool {
    matches!(pos, 1 | 8 | 9 | 6 | 14 | 15 | 48 | 49 | 54 | 55 | 57 | 62)
}

fn is_edge(pos: u8) -> bool {
    let row = pos / 8;
    let col = pos % 8;
    row == 0 || row == 7 || col == 0 || col == 7
}
```

---

## 5. TD(λ)-Leaf学習実装

### 5.1 データ構造

```rust
struct GameHistory {
    states: Vec<Board>,
    leaf_values: Vec<f32>,      // 各手の葉評価値
    pattern_indices: Vec<[usize; 56]>,  // 各手の56パターンインデックス
    stages: Vec<usize>,          // 各手のステージ
}

struct EligibilityTrace {
    // (pattern_id, stage, index) -> trace値
    traces: HashMap<(usize, usize, usize), f32>,
}

impl EligibilityTrace {
    fn new() -> Self {
        Self {
            traces: HashMap::new(),
        }
    }
    
    fn update(&mut self, lambda: f32) {
        for (_, trace) in self.traces.iter_mut() {
            *trace *= lambda;
        }
    }
    
    fn increment(&mut self, pattern_id: usize, stage: usize, index: usize) {
        *self.traces.entry((pattern_id, stage, index)).or_insert(0.0) += 1.0;
    }
    
    fn get(&self, pattern_id: usize, stage: usize, index: usize) -> f32 {
        *self.traces.get(&(pattern_id, stage, index)).unwrap_or(&0.0)
    }
}
```

### 5.2 自己対戦

```rust
fn play_self_game(
    evaluator: &Evaluator,
    searcher: &mut Searcher,
    epsilon: f32,
    rng: &mut impl Rng,
) -> GameHistory {
    let mut board = Board::new();
    let mut history = GameHistory {
        states: Vec::new(),
        leaf_values: Vec::new(),
        pattern_indices: Vec::new(),
        stages: Vec::new(),
    };
    
    while !board.is_game_over() && board.move_count() < 60 {
        // ε-greedy
        let (leaf_value, best_move) = if rng.gen::<f32>() < epsilon {
            // ランダム
            let moves = board.legal_moves();
            let mv = select_random_move(moves, rng);
            (evaluator.evaluate(&board), mv)
        } else {
            // 探索
            searcher.search(&board, evaluator)
        };
        
        // 履歴に保存
        history.states.push(board.clone());
        history.leaf_values.push(leaf_value);
        history.pattern_indices.push(extract_all_pattern_indices(&board));
        history.stages.push(board.move_count() / 2);
        
        board.make_move(best_move);
    }
    
    history
}
```

### 5.3 TD更新

```rust
fn td_update(
    history: &GameHistory,
    final_score: f32,
    lambda: f32,
    evaluator: &mut Evaluator,
    adam: &mut AdamOptimizer,
) {
    let n = history.states.len();
    let mut traces = EligibilityTrace::new();
    
    // 逆順に更新
    for t in (0..n).rev() {
        let stage = history.stages[t];
        let pattern_indices = &history.pattern_indices[t];
        
        // Eligibility Traceを更新
        for (i, &index) in pattern_indices.iter().enumerate() {
            let pattern_id = i % 14;
            traces.increment(pattern_id, stage, index);
        }
        
        // TD誤差計算
        let current_value = history.leaf_values[t];
        let target = if t == n - 1 {
            final_score  // 最終手
        } else {
            let next_value = history.leaf_values[t + 1];
            // TD(λ)ターゲット
            lambda * final_score + (1.0 - lambda) * next_value
        };
        
        let td_error = target - current_value;
        
        // 各パターンエントリを更新
        for (i, &index) in pattern_indices.iter().enumerate() {
            let pattern_id = i % 14;
            let trace = traces.get(pattern_id, stage, index);
            let gradient = td_error * trace;
            
            // Adam更新
            let current_score = u16_to_score(evaluator.tables[pattern_id][stage][index]);
            let new_score = adam.update(
                pattern_id, stage, index,
                current_score,
                gradient
            );
            evaluator.tables[pattern_id][stage][index] = score_to_u16(new_score);
        }
        
        // トレース減衰
        traces.update(lambda);
    }
}
```

---

## 6. Adamオプティマイザ実装

```rust
struct AdamOptimizer {
    alpha: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    
    // [pattern_id][stage][index]
    m: Vec<Vec<Vec<f32>>>,  // 1次モーメント
    v: Vec<Vec<Vec<f32>>>,  // 2次モーメント
    t: u64,                  // タイムステップ
}

impl AdamOptimizer {
    fn new(patterns: &[Pattern]) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();
        
        for pattern in patterns {
            let entries_per_stage = 3usize.pow(pattern.k as u32);
            let mut pattern_m = Vec::new();
            let mut pattern_v = Vec::new();
            
            for _ in 0..30 {
                pattern_m.push(vec![0.0f32; entries_per_stage]);
                pattern_v.push(vec![0.0f32; entries_per_stage]);
            }
            
            m.push(pattern_m);
            v.push(pattern_v);
        }
        
        Self {
            alpha: 0.025,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m,
            v,
            t: 0,
        }
    }
    
    fn update(
        &mut self,
        pattern_id: usize,
        stage: usize,
        index: usize,
        current_value: f32,
        gradient: f32,
    ) -> f32 {
        self.t += 1;
        
        // 1次モーメント更新
        let m = &mut self.m[pattern_id][stage][index];
        *m = self.beta1 * *m + (1.0 - self.beta1) * gradient;
        
        // 2次モーメント更新
        let v = &mut self.v[pattern_id][stage][index];
        *v = self.beta2 * *v + (1.0 - self.beta2) * gradient * gradient;
        
        // バイアス補正
        let m_hat = *m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = *v / (1.0 - self.beta2.powi(self.t as i32));
        
        // パラメータ更新
        current_value + self.alpha * m_hat / (v_hat.sqrt() + self.epsilon)
    }
}
```

---

## 7. チェックポイント保存

```rust
use std::io::{Write, Read};
use std::fs::File;

fn save_checkpoint(
    evaluator: &Evaluator,
    adam: &AdamOptimizer,
    game_count: usize,
    path: &str,
) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    
    // ヘッダー
    file.write_all(b"OTHELLO_AI_CHECKPOINT_V1")?;
    file.write_all(&game_count.to_le_bytes())?;
    
    // パターンテーブル
    for pattern in &evaluator.tables {
        for stage in pattern {
            for &value in stage {
                file.write_all(&value.to_le_bytes())?;
            }
        }
    }
    
    // Adam状態
    for pattern_m in &adam.m {
        for stage_m in pattern_m {
            for &m_value in stage_m {
                file.write_all(&m_value.to_le_bytes())?;
            }
        }
    }
    
    for pattern_v in &adam.v {
        for stage_v in pattern_v {
            for &v_value in stage_v {
                file.write_all(&v_value.to_le_bytes())?;
            }
        }
    }
    
    file.write_all(&adam.t.to_le_bytes())?;
    
    Ok(())
}
```

---

## 8. 完全読み実装

```rust
fn solve_exact(
    board: &mut Board,
    alpha: i32,
    beta: i32,
) -> i32 {
    if board.is_game_over() {
        return board.final_score() * 100;
    }
    
    let moves = board.legal_moves();
    if moves == 0 {
        board.pass();
        let score = -solve_exact(board, -beta, -alpha);
        board.undo_pass();
        return score;
    }
    
    let mut best = alpha;
    
    for mv in 0..64 {
        if moves & (1 << mv) == 0 {
            continue;
        }
        
        let undo = board.make_move(mv);
        let score = -solve_exact(board, -beta, -best);
        board.undo_move(undo);
        
        if score > best {
            best = score;
            if best >= beta {
                break;
            }
        }
    }
    
    best
}
```

---

## 9. Python連携（PyO3）

```rust
use pyo3::prelude::*;

#[pyclass]
struct PyEvaluator {
    evaluator: Evaluator,
}

#[pymethods]
impl PyEvaluator {
    #[new]
    fn new() -> Self {
        Self {
            evaluator: Evaluator::new(),
        }
    }
    
    fn evaluate(&self, board: Vec<u8>) -> PyResult<f32> {
        // board: 64要素のVec、0=空、1=黒、2=白
        let board = Board::from_array(&board)?;
        Ok(self.evaluator.evaluate(&board))
    }
    
    fn train_game(&mut self, time_ms: u64, epsilon: f32) -> PyResult<Vec<f32>> {
        // 1局対戦して統計を返す
        // ...
    }
    
    fn save(&self, path: &str) -> PyResult<()> {
        // チェックポイント保存
        Ok(())
    }
    
    fn load(&mut self, path: &str) -> PyResult<()> {
        // チェックポイント読み込み
        Ok(())
    }
}

#[pymodule]
fn othello_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEvaluator>()?;
    Ok(())
}
```

---

## 10. 最適化ポイント

### 10.1 コンパイラ最適化

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
```

### 10.2 並列化（Rayon）

```rust
use rayon::prelude::*;

fn train_parallel(
    num_games: usize,
    num_threads: usize,
) {
    (0..num_games).into_par_iter().for_each(|_| {
        let history = play_self_game(...);
        // ロック付きで更新
    });
}
```

### 10.3 SIMD（将来の最適化）

ARM Neon命令を使用してビット操作を高速化

```rust
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
```

---

本書により、実装レベルの詳細が明確になりました。
