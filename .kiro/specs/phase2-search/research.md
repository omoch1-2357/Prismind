# Research & Design Decisions

---
**Purpose**: Phase 2探索アルゴリズムの設計決定と技術調査結果

**Usage**:
- 探索アルゴリズム（Negamax、AlphaBeta、MTD(f)）の実装戦略を記録
- 置換表、Zobristハッシュ、ムーブオーダリングの設計判断を文書化
- 外部技術文献とベストプラクティスの調査結果を保存
---

## Summary
- **Feature**: `phase2-search`
- **Discovery Scope**: Complex Integration（Phase 1基盤の上に高性能探索システムを構築）
- **Key Findings**:
  - MTD(f)アルゴリズムはゼロ幅探索により AlphaBetaより平均5-15%高効率
  - Zobrist ハッシュ64ビットで衝突率は約2^32回に1回（実用上問題なし）
  - ムーブオーダリング（置換表最善手 + キラーヒューリスティック）で枝刈り効率20-30%向上
  - 残り14空きマスで完全読み開始が業界標準（約16,384局面）

## Research Log

### MTD(f)アルゴリズムの実装特性

**Context**: Phase 2の探索効率最大化のため、AlphaBeta、NegaScout、MTD(f)を比較評価

**Sources Consulted**:
- Aske Plaat (MIT CSAIL): https://people.csail.mit.edu/plaat/mtdf.html
- Wikipedia MTD-f: https://en.wikipedia.org/wiki/MTD-f
- Code Review Stack Exchange: Python MTD-f実装例

**Findings**:
- **ゼロ幅探索**: MTD(f)は幅0のAlphaBeta呼び出しを繰り返し、上限と下限を収束させる
- **初期推測値の重要性**: 良い初期推測（前回の反復深化の結果）で通常2-3パス、最悪5-15パスで収束
- **置換表必須**: MTD(f)は置換表なしでは全ノードを再探索するため非効率
- **実測性能**: トーナメントプログラム（Chinook、Phoenix、Keyano）でNegaScout/PVSを上回る性能

**Implications**:
- Phase 2ではAlphaBeta実装後にMTD(f)へ移行（段階的最適化）
- 反復深化の前回結果をMTD(f)の初期推測値として使用
- 置換表のエントリ品質（深さ、境界タイプ）がMTD(f)の性能を左右

### 置換表とZobristハッシュの設計

**Context**: 128-256MBの置換表で最大400万エントリを効率的に管理

**Sources Consulted**:
- Chessprogramming wiki: Zobrist Hashing
- Robert Hyatt論文: 衝突率実測データ
- Stack Overflow: 衝突検出とテーブルサイズ最適化

**Findings**:
- **64ビットハッシュ**: 業界標準、約2^32（40億）プローブに1回の衝突率
- **衝突検出**: フル64ビットハッシュをエントリに保存し、プローブ時に完全一致確認
- **テーブルサイズ**: 2の累乗より素数を使うべきだが、実用上は2の累乗でビットマスク（高速）が一般的
- **置換戦略**: 深さ優先（深い探索結果を保持）+ 世代管理（古いエントリを効率的に置換）
- **メモリオーバーヘッド**: エントリサイズは16-24バイト（hash 8B、depth 1B、score 2B、move 1B、bound 1B、age 1B + パディング）

**Implications**:
- TTEntryは24バイト構造体（8バイトアライメント）
- `hash % table_size` でインデックス計算（table_sizeは2の累乗）
- 置換時は`depth >= existing.depth || age != existing.age`で上書き
- 128MB = 約5.3M エントリ、256MB = 約10.6M エントリ

### ムーブオーダリング戦略

**Context**: 枝刈り効率を最大化するための合法手の評価順序

**Sources Consulted**:
- Chessprogramming wiki: Move Ordering
- AlphaBeta最適化論文（GeeksforGeeks、Applied AI Course）
- Othello専用ヒューリスティック（Medium記事）

**Findings**:
- **キラーヒューリスティック**: 同じ深さで前回βカットした手を優先（メモリ効率的）
- **MVV-LVA（Most Valuable Victim - Least Valuable Aggressor）**: チェスでは標準だが、Othelloでは角 > 辺 > 内側の位置ヒューリスティックに適応
- **置換表最善手**: 最優先で評価（前回の探索結果を活用）
- **前回イテレーションのPV（Principal Variation）**: 反復深化で次の深さの左端パスとして使用

**Implications**:
- 優先順位: 置換表最善手 > 角を取る手 > 辺の手 > X打ち回避 > その他
- Killerヒューリスティックは深さごとに2手保存（メモリ: 120B程度）
- 実装順序: 静的評価（角、辺）→ 置換表最善手 → Killerヒューリスティック（将来拡張）

### 完全読みの開始タイミング

**Context**: 残り何手から終局までの完全探索を開始するか

**Sources Consulted**:
- Cornell大学Othelloプロジェクト
- comp.ai.gamesディスカッション
- Radagast Othelloソルバー実装

**Findings**:
- **業界標準**: 残り14空きマスが一般的な閾値（約16,384局面）
- **実測限界**: 深さ12-14が実用的な限界、20-26は勝敗判定のみ（詳細評価なし）
- **計算量**: 14空きで平均分岐数5-10、最悪16,384ノード（AlphaBeta + 置換表で削減）
- **時間制約**: 15ms/手の制約下では深さ14で100ms以内が目標

**Implications**:
```rust
let empty_squares = 60 - board.move_count;
if empty_squares <= 14 {
    return complete_search(board, alpha, beta, tt, zobrist);
}
```
- （60-14=46）で完全読みモード切替
- 完全読みは専用関数（評価関数呼び出し不要、最終スコア×100を返す）
- 置換表は通常探索と共有（エンドゲームデータベースは不使用）

### Rust thiserrorによるエラー設計

**Context**: Phase 2探索での型安全なエラーハンドリング

**Sources Consulted**:
- Rust By Example: Custom Error Types
- thiserror vs anyhow比較記事（Nick's Blog、DEV Community）
- GreptimeDB大規模プロジェクト事例

**Findings**:
- **thiserrorの適用**: ライブラリ向け、呼び出し側が異なるエラー種別に応じて処理を分岐可能
- **anyhowの適用**: バイナリ向け、エラーは報告のみで詳細分岐不要
- **Phase 2の方針**: prismindはライブラリ（Phase 3学習システムから利用）のためthiserrorを使用
- **エラー種別**: `IllegalMove`、`OutOfBounds`、`MemoryAllocation`、`TimeoutExceeded`

**Implications**:
- `SearchError` enumをthiserrorで定義
- Phase 1の`GameError`と同様の設計パターンを踏襲
- Result<T, SearchError>を探索関数の戻り値型として使用

## Architecture Pattern Evaluation

### 選択したパターン: Layered Search Architecture

| Layer | Description | Responsibilities |
|-------|-------------|------------------|
| **Search Coordinator** | 探索全体の制御 | 時間管理、反復深化、探索統計収集 |
| **Search Algorithms** | 探索アルゴリズム実装 | Negamax、AlphaBeta、MTD(f) |
| **Transposition Table** | 局面キャッシュ | Zobristハッシュ、エントリ管理、置換戦略 |
| **Move Ordering** | 合法手の優先順位付け | 置換表最善手、角優先、X打ち回避 |
| **Evaluation Integration** | 評価関数呼び出し | Phase 1 Evaluatorとの統合 |

**Strengths**:
- 明確な責務分離（単一責任原則）
- 段階的実装が可能（Negamax → AlphaBeta → MTD(f)）
- テスト容易性（各レイヤーを独立テスト）

**Risks / Limitations**:
- レイヤー間の関数呼び出しオーバーヘッド（Rustのインライン展開で軽減）
- 置換表の並行アクセスは未対応（Phase 2ではシングルスレッド前提）

**Notes**:
- Phase 4並列化に備え、Search構造体は状態を分離（置換表は共有可能に設計）

## Design Decisions

### Decision: AlphaBeta fail-soft実装を採用

**Context**: AlphaBetaの境界値処理方式（fail-hard vs fail-soft）

**Alternatives Considered**:
1. **fail-hard**: alpha-beta範囲外は境界値を返す（実装シンプル）
2. **fail-soft**: alpha-beta範囲外でも正確な評価値を返す（情報量多い）

**Selected Approach**: fail-soft実装

**Rationale**:
- MTD(f)はゼロ幅探索を行うため、正確な境界値情報が収束速度に影響
- 置換表に保存する評価値の精度が向上（次回探索での再利用効率アップ）
- 実装コストはfail-hardと同等（returnする値を変えるのみ）

**Trade-offs**:
- Benefits: MTD(f)の収束速度向上、置換表エントリの品質向上
- Compromises: なし（実装複雑度は変わらず）

**Follow-up**: AlphaBeta実装時にfail-softロジックを確認（テストケースで検証）

### Decision: 置換表サイズを128MBデフォルト、256MB上限とする

**Context**: メモリ使用量とヒット率のトレードオフ

**Alternatives Considered**:
1. 64MB（約2.6M エントリ）: メモリ節約だがヒット率低下
2. 128MB（約5.3M エントリ）: バランス型
3. 256MB（約10.6M エントリ）: ヒット率最大化

**Selected Approach**: 128MBデフォルト、実行時に256MBまで設定可能

**Rationale**:
- Phase 1評価テーブル70MB + 置換表128MB + その他 = 約220MB（OCI 24GB環境で余裕）
- 128MBで中盤以降のヒット率50%以上を達成見込み
- 256MBオプションで学習時（Phase 3）のヒット率最適化

**Trade-offs**:
- Benefits: 実用的なヒット率、メモリ余裕、柔軟な設定
- Compromises: 64MBより大きいが、Phase 3学習の33億回更新に対応可能

**Follow-up**: ベンチマークでヒット率を測定、必要に応じて調整

### Decision: ムーブオーダリングは静的評価のみ（Phase 2スコープ）

**Context**: ムーブオーダリングの実装範囲

**Alternatives Considered**:
1. 静的評価のみ（角 > 辺 > X打ち回避）
2. 静的評価 + 置換表最善手
3. 静的評価 + 置換表 + Killerヒューリスティック + ヒストリーヒューリスティック

**Selected Approach**: 静的評価 + 置換表最善手（オプション2）

**Rationale**:
- 静的評価は実装コスト低く、Othelloの戦略的知識を反映
- 置換表最善手は追加コスト最小で効果大（既存データ活用）
- KillerヒューリスティックはPhase 4並列化時に追加検討

**Trade-offs**:
- Benefits: 実装シンプル、20-30%枝刈り効率向上見込み
- Compromises: Killer/Historyヒューリスティックは将来拡張

**Follow-up**: Phase 2完了後、Killerヒューリスティックの効果を測定

### Decision: 探索統計はSearchResult構造体で返す

**Context**: 探索ノード数、置換表ヒット率、探索時間の記録方法

**Alternatives Considered**:
1. グローバル変数でカウンタ管理（シンプルだが並行処理不可）
2. Search構造体にカウンタフィールド（可変借用の管理が複雑）
3. SearchResult構造体で統計を返す（不変性維持、並行処理対応）

**Selected Approach**: SearchResult構造体で統計を返す（オプション3）

**Rationale**:
- Rustの所有権モデルに適合（不変性維持）
- Phase 4並列化時にスレッド間の統計集計が容易
- Phase 3学習システムが探索統計をロギング可能

**Trade-offs**:
- Benefits: 並行処理対応、型安全、テスト容易
- Compromises: SearchResult構造体のサイズ増加（64バイト程度）

**Follow-up**: SearchResult構造体の詳細設計（必須フィールドとオプションフィールド）

## Risks & Mitigations

### Risk 1: MTD(f)の収束遅延
**Description**: 初期推測値が悪い場合、15パス以上かかり15ms制限を超過
**Mitigation**:
- 反復深化の前回結果を初期推測値とする（通常2-3パスで収束）
- 時間制限80%で次の深さをスキップ（最後に完了した深さの最善手を返す）

### Risk 2: 置換表衝突による探索誤差
**Description**: 64ビットハッシュでも40億プローブに1回衝突、最善手を誤る可能性
**Mitigation**:
- フル64ビットハッシュをエントリに保存し、プローブ時に完全一致確認
- TTEntryにhashフィールドを保存し、entry.hash == computed_hashで衝突を検出。一致しない場合はミス扱い
- 学習時（Phase 3）の統計で衝突率を監視（1e-7以下なら実用上問題なし）

### Risk 3: 完全読みの時間超過
**Description**: 残り14手で100ms目標だが、局面によっては超過可能性
**Mitigation**:
- 完全読みでも置換表を活用（同一局面の再評価回避）
- 時間切れ時はヒューリスティック評価にフォールバック
- ベンチマークで14手完全読みの平均時間を測定

### Risk 4: Phase 1 APIの変更リスク
**Description**: Phase 2実装中にPhase 1の評価関数APIが変更される可能性
**Mitigation**:
- Phase 1は完了済み（185テスト成功）、APIは安定
- Phase 2ではPhase 1の公開APIのみ使用（内部実装に依存しない）
- 統合テストでPhase 1とPhase 2の互換性を継続検証

## References

### Official Documentation
- [Aske Plaat: MTD(f) Algorithm](https://people.csail.mit.edu/plaat/mtdf.html) — MTD(f)の原著者による詳細解説
- [Chessprogramming Wiki: Zobrist Hashing](https://www.chessprogramming.org/Zobrist_Hashing) — Zobristハッシュの実装詳細
- [Chessprogramming Wiki: Move Ordering](https://www.chessprogramming.org/Move_Ordering) — ムーブオーダリングの各種ヒューリスティック

### Research Papers
- Robert Hyatt & Anthony Cozzie: "Hash Collisions in Game Tree Search" — 64ビットハッシュの衝突率実測データ
- "Othello is solved" (arXiv 2310.19387v3) — 最新のOthello完全解析（2024年）

### Implementation References
- [GitHub: GalaX1us/Othello-Agents](https://github.com/GalaX1us/Othello-Agents) — Python実装（Negamax、MTD-f、MCTS）
- [Radagast Othello Solver](http://www.radagast.se/othello/endgame.c) — C言語による完全読み実装例

### Rust Error Handling
- [Rust By Example: Custom Error Types](https://doc.rust-lang.org/rust-by-example/error/multiple_error_types/define_error_type.html)
- [thiserror crate documentation](https://docs.rs/thiserror/) — カスタムエラー型のマクロ
