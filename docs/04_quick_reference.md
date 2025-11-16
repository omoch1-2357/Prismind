# オセロAI 実装クイックリファレンス

実装時に素早く参照できる重要なアルゴリズムとデータ構造の概要

---

## 1. 主要データ構造

### Board (BitBoard)
```rust
struct Board {
    black: u64,     // 黒石のビットマスク
    white: u64,     // 白石のビットマスク
    turn: Color,    // 現在の手番
}
```

### Pattern
```rust
struct Pattern {
    id: usize,              // 0-13 (P01-P14)
    k: usize,               // セル数（7-10）
    positions: Vec<u8>,     // セル位置リスト（0-63）
}
```

### Evaluator
```rust
struct Evaluator {
    // [pattern_id][stage][index] -> u16
    tables: Vec<Vec<Vec<u16>>>,
}
```

### TranspositionTable Entry
```rust
struct TTEntry {
    hash: u64,
    depth: i8,
    bound: Bound,      // Exact/Lower/Upper
    score: i16,
    best_move: u8,
}
```

### AdamOptimizer
```rust
struct AdamOptimizer {
    m: Vec<Vec<Vec<f32>>>,  // 1次モーメント
    v: Vec<Vec<Vec<f32>>>,  // 2次モーメント
    t: u64,                  // タイムステップ
}
```

---

## 2. 核心アルゴリズム（疑似コード）

### 評価関数
```
function evaluate(board, stage):
    sum = 0
    
    for rotation in [0°, 90°, 180°, 270°]:
        rotated = rotate_board(board, rotation)
        swap = (rotation == 90° or rotation == 270°)
        
        for pattern in PATTERNS[0..13]:
            index = extract_index(rotated, pattern, swap)
            score = u16_to_score(table[pattern.id][stage][index])
            sum += score
    
    if board.turn == White:
        return -sum
    else:
        return sum
```

### MTD(f)探索
```
function mtdf(board, depth, guess):
    g = guess
    lower = -∞
    upper = +∞
    
    while lower < upper:
        beta = (g == lower) ? g + 1 : g
        g = alpha_beta(board, depth, beta - 1, beta)
        
        if g < beta:
            upper = g
        else:
            lower = g
    
    return g
```

### 反復深化
```
function iterative_deepening(board, time_limit):
    start_time = now()
    guess = evaluate(board)
    best_move = null
    
    for depth = 1 to 60:
        if elapsed_time() > time_limit:
            break
        
        score, move = mtdf(board, depth, guess)
        guess = score
        best_move = move
        
        if elapsed_time() > time_limit * 0.8:
            break
    
    return best_move
```

### TD(λ)-Leaf更新
```
function td_update(history, final_score, lambda):
    traces = new EligibilityTrace()
    
    for t from (history.length - 1) down to 0:
        # Eligibility Trace更新
        for each pattern_instance in history[t]:
            traces.increment(pattern_instance)
        
        # TD誤差計算
        current_value = history[t].leaf_value
        
        if t == last:
            target = final_score
        else:
            next_value = history[t+1].leaf_value
            target = lambda * final_score + (1 - lambda) * next_value
        
        td_error = target - current_value
        
        # 各パターンを更新
        for each pattern_instance in history[t]:
            trace = traces.get(pattern_instance)
            gradient = td_error * trace
            
            # Adam更新
            new_value = adam.update(pattern_instance, gradient)
            table[pattern_instance] = score_to_u16(new_value)
        
        # トレース減衰
        traces.decay(lambda)
```

### Adam更新
```
function adam_update(param_id, current_value, gradient):
    t += 1
    
    # モーメント更新
    m[param_id] = β1 * m[param_id] + (1 - β1) * gradient
    v[param_id] = β2 * v[param_id] + (1 - β2) * gradient²
    
    # バイアス補正
    m_hat = m[param_id] / (1 - β1^t)
    v_hat = v[param_id] / (1 - β2^t)
    
    # パラメータ更新
    new_value = current_value + α * m_hat / (√v_hat + ε)
    
    return new_value
```

### 自己対戦ループ
```
function play_self_game(evaluator, epsilon):
    board = new_initial_board()
    history = []
    
    while not board.is_game_over():
        if random() < epsilon:
            # ε-greedy: ランダム手
            move = random_legal_move(board)
            leaf_value = evaluate(board)
        else:
            # 探索
            move = iterative_deepening(board, time_limit=15ms)
            leaf_value = evaluate(board)  # 探索後の評価
        
        history.append({
            board: board.clone(),
            leaf_value: leaf_value,
            patterns: extract_all_patterns(board),
            stage: board.move_count() / 2
        })
        
        board.make_move(move)
    
    return history, board.final_score()
```

---

## 3. 重要な変換関数

### u16 ⇔ 石差
```rust
// u16 → 石差
fn u16_to_score(v: u16) -> f32 {
    (v as f32 - 32768.0) / 256.0
}

// 石差 → u16
fn score_to_u16(s: f32) -> u16 {
    ((s * 256.0 + 32768.0).clamp(0.0, 65535.0)) as u16
}
```

### パターンインデックス抽出
```rust
// 3進数でインデックス化
// 0=空, 1=黒, 2=白（swap時は逆）
fn extract_index(black: u64, white: u64, pattern: &Pattern, swap: bool) -> usize {
    let mut index = 0;
    
    for (i, &pos) in pattern.positions.iter().enumerate() {
        let state = if black & (1 << pos) != 0 {
            if swap { 2 } else { 1 }
        } else if white & (1 << pos) != 0 {
            if swap { 1 } else { 2 }
        } else {
            0
        };
        
        index += state * 3.pow(i);
    }
    
    index
}
```

### BitBoard回転
```rust
// 90度回転: (row, col) → (col, 7-row)
fn rotate_90(board: u64) -> u64 {
    let mut result = 0;
    for row in 0..8 {
        for col in 0..8 {
            if board & (1 << (row * 8 + col)) != 0 {
                result |= 1 << (col * 8 + (7 - row));
            }
        }
    }
    result
}

// 180度回転: ビット順序反転
fn rotate_180(board: u64) -> u64 {
    board.reverse_bits()
}

// 270度回転
fn rotate_270(board: u64) -> u64 {
    rotate_90(rotate_180(board))
}
```

---

## 4. 主要な制御フロー

### メイン学習ループ
```
initialize evaluator (all entries = 32768)
initialize adam (m=0, v=0, t=0)

for game_num from 0 to 1_000_000:
    # ε-greedy設定
    epsilon = get_epsilon(game_num)
    
    # 自己対戦
    history, final_score = play_self_game(evaluator, epsilon)
    
    # TD更新
    td_update(history, final_score, lambda=0.3, evaluator, adam)
    
    # 統計記録
    if game_num % 100 == 0:
        log_statistics(history, final_score)
    
    # チェックポイント
    if game_num % 100_000 == 0:
        save_checkpoint(evaluator, adam, game_num)
```

### ε-greedy設定
```
function get_epsilon(game_num):
    if game_num < 300_000:
        return 0.15
    elif game_num < 700_000:
        return 0.05
    else:
        return 0.0
```

---

## 5. BitBoard操作の基本

### 合法手判定
```
empty = ~(black | white)
opponent = white (黒番時)

for each direction:
    # 方向に沿って相手の石があるか探索
    candidates = shift(opponent, dir) & black
    
    while candidates != 0:
        candidates = shift(candidates, dir)
        legal_moves |= candidates & empty
        candidates &= opponent
```

### 石を返す
```
for each direction:
    flipped = find_consecutive_opponent_stones(pos, dir)
    if flipped ends with player stone:
        player |= flipped
        opponent &= ~flipped
```

---

## 6. 置換表操作

### プローブ
```
index = hash % table_size
entry = table[index]

if entry.hash == hash and entry.depth >= depth:
    if entry.bound == Exact:
        return entry.score
    elif entry.bound == Lower:
        alpha = max(alpha, entry.score)
    elif entry.bound == Upper:
        beta = min(beta, entry.score)
    
    if alpha >= beta:
        return entry.score
```

### 保存
```
index = hash % table_size

# 置換戦略: 深さと世代で判断
if table[index] is empty or 
   table[index].depth < new_depth or
   table[index].age != current_age:
    table[index] = new_entry
```

---

## 7. Move Ordering実装

```
function order_moves(moves, tt_entry):
    move_list = []
    
    for pos in legal_positions(moves):
        priority = 0
        
        if pos == tt_entry.best_move:
            priority = -10000
        elif is_corner(pos):
            priority = -1000
        elif is_x_square(pos):
            priority = +500
        elif is_edge(pos):
            priority = -100
        
        move_list.append((pos, priority))
    
    sort move_list by priority (ascending)
    return move_list
```

---

## 8. 完全読み（終盤）

```
function solve_exact(board, alpha, beta):
    if game_over:
        return final_score * 100
    
    moves = legal_moves(board)
    
    if no_moves:
        # パス
        return -solve_exact(pass(board), -beta, -alpha)
    
    best = alpha
    
    for move in moves:
        board.make_move(move)
        score = -solve_exact(board, -beta, -best)
        board.undo_move()
        
        best = max(best, score)
        if best >= beta:
            break  # βカット
    
    return best
```

---

## 9. パフォーマンスチェックポイント

### 高速化の要所
1. **BitBoard操作**: ビット演算で一括処理
2. **パターン抽出**: 事前計算テーブル使用
3. **置換表**: 高いヒット率を維持
4. **Move Ordering**: 枝刈りを最大化
5. **並列化**: 複数ゲームを同時実行

### プロファイリング重点箇所
- `legal_moves()`: 最頻呼び出し
- `extract_pattern_index()`: 56回/局面
- `alpha_beta()`: 再帰呼び出し多数
- `evaluate()`: 評価関数呼び出し

---

## 10. デバッグ用チェックリスト

### 盤面表示
```
function print_board(black, white):
    for row in 0..8:
        for col in 0..8:
            pos = row * 8 + col
            if black & (1 << pos):
                print "●"
            elif white & (1 << pos):
                print "○"
            else:
                print "."
```

### パターン抽出確認
```
# 初期盤面で4方向のパターンが正しく抽出されるか
board = initial_board()
for rotation in [0, 90, 180, 270]:
    indices = extract_all_patterns(rotate(board, rotation))
    print indices
```

### 評価値の対称性確認
```
# 同じ盤面を回転しても評価値が同じになるか
board = some_board()
scores = []
for rotation in [0, 90, 180, 270]:
    rotated = rotate(board, rotation)
    scores.append(evaluate(rotated))

assert all scores are equal (considering color swap)
```

---

## 11. 典型的なバグと対処

| バグ | 症状 | 対処 |
|------|------|------|
| インデックス越境 | パニック | パターン位置が0-63の範囲内か確認 |
| 評価値発散 | NaNまたは極端な値 | スコア変換のclamp確認 |
| 置換表衝突 | 不正確な探索結果 | hash衝突検出を追加 |
| 対称性不一致 | 回転で評価値変化 | 白黒反転の実装確認 |
| TD更新エラー | 学習が進まない | trace減衰率とλ値確認 |

---

## 12. 実装順序（推奨）

### Phase 1: 基礎
1. BitBoard実装
2. 合法手生成
3. 石を返す処理
4. 回転操作

### Phase 2: パターン
5. パターン定義読み込み
6. インデックス抽出
7. 評価関数（固定値で）

### Phase 3: 探索
8. AlphaBeta基本実装
9. 置換表
10. Move Ordering
11. MTD(f)移行
12. 反復深化

### Phase 4: 学習
13. Adam実装
14. TD更新
15. Eligibility Trace
16. 自己対戦ループ

### Phase 5: 統合
17. チェックポイント保存/読み込み
18. ログ・統計
19. Python連携
20. 並列化

---

このクイックリファレンスで、実装中に重要なアルゴリズムや構造を素早く確認できます。
