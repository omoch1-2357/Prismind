//! x86-64専用最適化モジュール
//!
//! このモジュールはx86-64アーキテクチャ専用の最適化を集約する:
//! - SSE/AVX SIMD命令による並列処理
//! - BMI2 PEXT命令によるパターン抽出の高速化
//! - プリフェッチによるキャッシュ最適化
//!
//! # プラットフォームサポート
//!
//! このモジュールの機能はx86-64 (x86_64) アーキテクチャでのみ利用可能。
//! 他のプラットフォームでは、これらの最適化は無効化され、スカラー実装が使用される。

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::sync::OnceLock;

// ============================================================================
// PEXT-based Pattern Extraction (BMI2)
// ============================================================================

/// k=10用の3進数ルックアップテーブル（1024 x 1024 = 1MB）
/// `TERNARY_LUT_K10\[black_bits\]\[white_bits\]` = 3進数インデックス
///
/// 事前計算により、ループなしで3進数インデックスを取得可能。
/// メモリ使用量は大きいが、キャッシュに乗れば非常に高速。
#[cfg(target_arch = "x86_64")]
static TERNARY_LUT_K10: once_cell::sync::Lazy<Box<[[u32; 1024]; 1024]>> =
    once_cell::sync::Lazy::new(|| {
        // Heap-allocate via Vec to avoid large stack frames during initialization
        let mut lut: Box<[[u32; 1024]; 1024]> = vec![[0u32; 1024]; 1024]
            .into_boxed_slice()
            .try_into()
            .expect("length must match");
        for black_bits in 0..1024usize {
            for white_bits in 0..1024usize {
                let mut index = 0u32;
                let mut power_of_3 = 1u32;
                for bit_pos in 0..10 {
                    let is_black = ((black_bits >> bit_pos) & 1) as u32;
                    let is_white = ((white_bits >> bit_pos) & 1) as u32;
                    // 0=empty, 1=black, 2=white
                    let cell_value = is_black + is_white * 2;
                    index += cell_value * power_of_3;
                    power_of_3 *= 3;
                }
                lut[black_bits][white_bits] = index;
            }
        }
        lut
    });

/// k=8用の3進数ルックアップテーブル（256 x 256 = 64KB）
#[cfg(target_arch = "x86_64")]
static TERNARY_LUT_K8: once_cell::sync::Lazy<Box<[[u16; 256]; 256]>> =
    once_cell::sync::Lazy::new(|| {
        let mut lut: Box<[[u16; 256]; 256]> = vec![[0u16; 256]; 256]
            .into_boxed_slice()
            .try_into()
            .expect("length must match");
        for black_bits in 0..256usize {
            for white_bits in 0..256usize {
                let mut index = 0u16;
                let mut power_of_3 = 1u16;
                for bit_pos in 0..8 {
                    let is_black = ((black_bits >> bit_pos) & 1) as u16;
                    let is_white = ((white_bits >> bit_pos) & 1) as u16;
                    let cell_value = is_black + is_white * 2;
                    index += cell_value * power_of_3;
                    power_of_3 *= 3;
                }
                lut[black_bits][white_bits] = index;
            }
        }
        lut
    });

/// k=7用の3進数ルックアップテーブル（128 x 128 = 16KB）
#[cfg(target_arch = "x86_64")]
static TERNARY_LUT_K7: once_cell::sync::Lazy<Box<[[u16; 128]; 128]>> =
    once_cell::sync::Lazy::new(|| {
        let mut lut: Box<[[u16; 128]; 128]> = vec![[0u16; 128]; 128]
            .into_boxed_slice()
            .try_into()
            .expect("length must match");
        for black_bits in 0..128usize {
            for white_bits in 0..128usize {
                let mut index = 0u16;
                let mut power_of_3 = 1u16;
                for bit_pos in 0..7 {
                    let is_black = ((black_bits >> bit_pos) & 1) as u16;
                    let is_white = ((white_bits >> bit_pos) & 1) as u16;
                    let cell_value = is_black + is_white * 2;
                    index += cell_value * power_of_3;
                    power_of_3 *= 3;
                }
                lut[black_bits][white_bits] = index;
            }
        }
        lut
    });

/// k=6用の3進数ルックアップテーブル（64 x 64 = 4KB）
#[cfg(target_arch = "x86_64")]
static TERNARY_LUT_K6: once_cell::sync::Lazy<Box<[[u16; 64]; 64]>> =
    once_cell::sync::Lazy::new(|| {
        let mut lut: Box<[[u16; 64]; 64]> = vec![[0u16; 64]; 64]
            .into_boxed_slice()
            .try_into()
            .expect("length must match");
        for black_bits in 0..64usize {
            for white_bits in 0..64usize {
                let mut index = 0u16;
                let mut power_of_3 = 1u16;
                for bit_pos in 0..6 {
                    let is_black = ((black_bits >> bit_pos) & 1) as u16;
                    let is_white = ((white_bits >> bit_pos) & 1) as u16;
                    let cell_value = is_black + is_white * 2;
                    index += cell_value * power_of_3;
                    power_of_3 *= 3;
                }
                lut[black_bits][white_bits] = index;
            }
        }
        lut
    });

/// k=5用の3進数ルックアップテーブル（32 x 32 = 1KB）
#[cfg(target_arch = "x86_64")]
static TERNARY_LUT_K5: once_cell::sync::Lazy<Box<[[u8; 32]; 32]>> =
    once_cell::sync::Lazy::new(|| {
        let mut lut: Box<[[u8; 32]; 32]> = vec![[0u8; 32]; 32]
            .into_boxed_slice()
            .try_into()
            .expect("length must match");
        for black_bits in 0..32usize {
            for white_bits in 0..32usize {
                // Use u16 to avoid overflow when multiplying by 3 after the last iteration.
                let mut index: u16 = 0;
                let mut power_of_3: u16 = 1;
                for bit_pos in 0..5 {
                    let is_black = ((black_bits >> bit_pos) & 1) as u16;
                    let is_white = ((white_bits >> bit_pos) & 1) as u16;
                    let cell_value = is_black + is_white * 2;
                    index += cell_value * power_of_3;
                    power_of_3 *= 3;
                }
                lut[black_bits][white_bits] = index as u8;
            }
        }
        lut
    });

/// PEXTの出力ビットをpext_to_array_mapに従って並べ替え
///
/// PEXTはビット位置順（LSB→MSB）でビットを出力するが、
/// 3進数計算には配列順序が必要。この関数でビットを正しい順序に並べ替える。
#[inline(always)]
fn permute_bits_k10(pext_bits: usize, perm: &[u8; 10]) -> usize {
    let mut r = 0usize;
    r |= (pext_bits & 1) << (perm[0] as usize);
    r |= ((pext_bits >> 1) & 1) << (perm[1] as usize);
    r |= ((pext_bits >> 2) & 1) << (perm[2] as usize);
    r |= ((pext_bits >> 3) & 1) << (perm[3] as usize);
    r |= ((pext_bits >> 4) & 1) << (perm[4] as usize);
    r |= ((pext_bits >> 5) & 1) << (perm[5] as usize);
    r |= ((pext_bits >> 6) & 1) << (perm[6] as usize);
    r |= ((pext_bits >> 7) & 1) << (perm[7] as usize);
    r |= ((pext_bits >> 8) & 1) << (perm[8] as usize);
    r |= ((pext_bits >> 9) & 1) << (perm[9] as usize);
    r
}

#[inline(always)]
fn permute_bits_k8(pext_bits: usize, perm: &[u8; 10]) -> usize {
    let mut r = 0usize;
    r |= (pext_bits & 1) << (perm[0] as usize);
    r |= ((pext_bits >> 1) & 1) << (perm[1] as usize);
    r |= ((pext_bits >> 2) & 1) << (perm[2] as usize);
    r |= ((pext_bits >> 3) & 1) << (perm[3] as usize);
    r |= ((pext_bits >> 4) & 1) << (perm[4] as usize);
    r |= ((pext_bits >> 5) & 1) << (perm[5] as usize);
    r |= ((pext_bits >> 6) & 1) << (perm[6] as usize);
    r |= ((pext_bits >> 7) & 1) << (perm[7] as usize);
    r
}

#[inline(always)]
fn permute_bits_k7(pext_bits: usize, perm: &[u8; 10]) -> usize {
    let mut r = 0usize;
    r |= (pext_bits & 1) << (perm[0] as usize);
    r |= ((pext_bits >> 1) & 1) << (perm[1] as usize);
    r |= ((pext_bits >> 2) & 1) << (perm[2] as usize);
    r |= ((pext_bits >> 3) & 1) << (perm[3] as usize);
    r |= ((pext_bits >> 4) & 1) << (perm[4] as usize);
    r |= ((pext_bits >> 5) & 1) << (perm[5] as usize);
    r |= ((pext_bits >> 6) & 1) << (perm[6] as usize);
    r
}

#[inline(always)]
fn permute_bits_k6(pext_bits: usize, perm: &[u8; 10]) -> usize {
    let mut r = 0usize;
    r |= (pext_bits & 1) << (perm[0] as usize);
    r |= ((pext_bits >> 1) & 1) << (perm[1] as usize);
    r |= ((pext_bits >> 2) & 1) << (perm[2] as usize);
    r |= ((pext_bits >> 3) & 1) << (perm[3] as usize);
    r |= ((pext_bits >> 4) & 1) << (perm[4] as usize);
    r |= ((pext_bits >> 5) & 1) << (perm[5] as usize);
    r
}

#[inline(always)]
fn permute_bits_k5(pext_bits: usize, perm: &[u8; 10]) -> usize {
    let mut r = 0usize;
    r |= (pext_bits & 1) << (perm[0] as usize);
    r |= ((pext_bits >> 1) & 1) << (perm[1] as usize);
    r |= ((pext_bits >> 2) & 1) << (perm[2] as usize);
    r |= ((pext_bits >> 3) & 1) << (perm[3] as usize);
    r |= ((pext_bits >> 4) & 1) << (perm[4] as usize);
    r
}

#[derive(Debug)]
struct PermuteCaches {
    k10: [[u16; 1024]; 16],
    k8: [[u16; 256]; 16],
    k7: [[u16; 128]; 8],
    k6: [[u16; 64]; 8],
    k5: [[u16; 32]; 8],
}

static PERMUTE_CACHES: OnceLock<PermuteCaches> = OnceLock::new();

#[derive(Debug)]
struct FusedCaches {
    // rotation(0-3) x pattern local index
    k10: [Box<[u16]>; 16],
    k8: [Box<[u16]>; 16],
    k7: [Box<[u16]>; 8],
    k6: [Box<[u16]>; 8],
    k5: [Box<[u16]>; 8],
}

static FUSED_CACHES: OnceLock<FusedCaches> = OnceLock::new();

fn build_permute_caches(patterns: &[crate::pattern::Pattern]) -> PermuteCaches {
    debug_assert_eq!(patterns.len(), 14);

    let mut caches = PermuteCaches {
        k10: [[0u16; 1024]; 16],
        k8: [[0u16; 256]; 16],
        k7: [[0u16; 128]; 8],
        k6: [[0u16; 64]; 8],
        k5: [[0u16; 32]; 8],
    };

    // k=10 patterns: indices 0-3
    for (pattern_idx, pattern) in patterns.iter().enumerate().take(4) {
        for rotation in 0..4 {
            let perm = &pattern.pext_to_array_map[rotation];
            let table = &mut caches.k10[rotation * 4 + pattern_idx];
            for (val, entry) in table.iter_mut().enumerate() {
                *entry = permute_bits_k10(val, perm) as u16;
            }
        }
    }

    // k=8 patterns: indices 4-7
    for (pattern_idx, pattern) in patterns.iter().enumerate().skip(4).take(4) {
        let local_idx = pattern_idx - 4;
        for rotation in 0..4 {
            let perm = &pattern.pext_to_array_map[rotation];
            let table = &mut caches.k8[rotation * 4 + local_idx];
            for (val, entry) in table.iter_mut().enumerate() {
                *entry = permute_bits_k8(val, perm) as u16;
            }
        }
    }

    // k=7 patterns: indices 8-9
    for (pattern_idx, pattern) in patterns.iter().enumerate().skip(8).take(2) {
        let local_idx = pattern_idx - 8;
        for rotation in 0..4 {
            let perm = &pattern.pext_to_array_map[rotation];
            let table = &mut caches.k7[rotation * 2 + local_idx];
            for (val, entry) in table.iter_mut().enumerate() {
                *entry = permute_bits_k7(val, perm) as u16;
            }
        }
    }

    // k=6 patterns: indices 10-11
    for (pattern_idx, pattern) in patterns.iter().enumerate().skip(10).take(2) {
        let local_idx = pattern_idx - 10;
        for rotation in 0..4 {
            let perm = &pattern.pext_to_array_map[rotation];
            let table = &mut caches.k6[rotation * 2 + local_idx];
            for (val, entry) in table.iter_mut().enumerate() {
                *entry = permute_bits_k6(val, perm) as u16;
            }
        }
    }

    // k=5 patterns: indices 12-13
    for (pattern_idx, pattern) in patterns.iter().enumerate().skip(12).take(2) {
        let local_idx = pattern_idx - 12;
        for rotation in 0..4 {
            let perm = &pattern.pext_to_array_map[rotation];
            let table = &mut caches.k5[rotation * 2 + local_idx];
            for (val, entry) in table.iter_mut().enumerate() {
                *entry = permute_bits_k5(val, perm) as u16;
            }
        }
    }

    caches
}

#[inline(always)]
fn get_permute_caches(patterns: &[crate::pattern::Pattern]) -> &PermuteCaches {
    PERMUTE_CACHES.get_or_init(|| build_permute_caches(patterns))
}

#[inline]
fn build_fused_table_k10(perm: &[u16; 1024], swap_colors: bool) -> Box<[u16]> {
    const SIZE: usize = 1024;
    let mut lut = vec![0u16; SIZE * SIZE].into_boxed_slice();

    for (black_raw, &black_perm) in perm.iter().enumerate() {
        let black_bits = black_perm as usize;
        let row_base = black_raw * SIZE;
        for (white_raw, &white_perm) in perm.iter().enumerate() {
            let white_bits = white_perm as usize;
            let val = if swap_colors {
                TERNARY_LUT_K10[white_bits][black_bits]
            } else {
                TERNARY_LUT_K10[black_bits][white_bits]
            } as u16;
            // SAFETY: bounds checked by construction
            unsafe {
                *lut.get_unchecked_mut(row_base + white_raw) = val;
            }
        }
    }

    lut
}

#[inline]
fn build_fused_table_k8(perm: &[u16; 256], swap_colors: bool) -> Box<[u16]> {
    const SIZE: usize = 256;
    let mut lut = vec![0u16; SIZE * SIZE].into_boxed_slice();

    for (black_raw, &black_perm) in perm.iter().enumerate() {
        let black_bits = black_perm as usize;
        let row_base = black_raw * SIZE;
        for (white_raw, &white_perm) in perm.iter().enumerate() {
            let white_bits = white_perm as usize;
            let val = if swap_colors {
                TERNARY_LUT_K8[white_bits][black_bits]
            } else {
                TERNARY_LUT_K8[black_bits][white_bits]
            };
            unsafe {
                *lut.get_unchecked_mut(row_base + white_raw) = val;
            }
        }
    }

    lut
}

#[inline]
fn build_fused_table_k7(perm: &[u16; 128], swap_colors: bool) -> Box<[u16]> {
    const SIZE: usize = 128;
    let mut lut = vec![0u16; SIZE * SIZE].into_boxed_slice();

    for (black_raw, &black_perm) in perm.iter().enumerate() {
        let black_bits = black_perm as usize;
        let row_base = black_raw * SIZE;
        for (white_raw, &white_perm) in perm.iter().enumerate() {
            let white_bits = white_perm as usize;
            let val = if swap_colors {
                TERNARY_LUT_K7[white_bits][black_bits]
            } else {
                TERNARY_LUT_K7[black_bits][white_bits]
            };
            unsafe {
                *lut.get_unchecked_mut(row_base + white_raw) = val;
            }
        }
    }

    lut
}

#[inline]
fn build_fused_table_k6(perm: &[u16; 64], swap_colors: bool) -> Box<[u16]> {
    const SIZE: usize = 64;
    let mut lut = vec![0u16; SIZE * SIZE].into_boxed_slice();

    for (black_raw, &black_perm) in perm.iter().enumerate() {
        let black_bits = black_perm as usize;
        let row_base = black_raw * SIZE;
        for (white_raw, &white_perm) in perm.iter().enumerate() {
            let white_bits = white_perm as usize;
            let val = if swap_colors {
                TERNARY_LUT_K6[white_bits][black_bits]
            } else {
                TERNARY_LUT_K6[black_bits][white_bits]
            };
            unsafe {
                *lut.get_unchecked_mut(row_base + white_raw) = val;
            }
        }
    }

    lut
}

#[inline]
fn build_fused_table_k5(perm: &[u16; 32], swap_colors: bool) -> Box<[u16]> {
    const SIZE: usize = 32;
    let mut lut = vec![0u16; SIZE * SIZE].into_boxed_slice();

    for (black_raw, &black_perm) in perm.iter().enumerate() {
        let black_bits = black_perm as usize;
        let row_base = black_raw * SIZE;
        for (white_raw, &white_perm) in perm.iter().enumerate() {
            let white_bits = white_perm as usize;
            let val = if swap_colors {
                TERNARY_LUT_K5[white_bits][black_bits]
            } else {
                TERNARY_LUT_K5[black_bits][white_bits]
            } as u16;
            unsafe {
                *lut.get_unchecked_mut(row_base + white_raw) = val;
            }
        }
    }

    lut
}

fn build_fused_caches(patterns: &[crate::pattern::Pattern]) -> FusedCaches {
    debug_assert_eq!(patterns.len(), 14);
    let perm = get_permute_caches(patterns);

    let mut k10: [Box<[u16]>; 16] = std::array::from_fn(|_| Vec::new().into_boxed_slice());
    let mut k8: [Box<[u16]>; 16] = std::array::from_fn(|_| Vec::new().into_boxed_slice());
    let mut k7: [Box<[u16]>; 8] = std::array::from_fn(|_| Vec::new().into_boxed_slice());
    let mut k6: [Box<[u16]>; 8] = std::array::from_fn(|_| Vec::new().into_boxed_slice());
    let mut k5: [Box<[u16]>; 8] = std::array::from_fn(|_| Vec::new().into_boxed_slice());

    for rotation in 0..4 {
        let swap_colors = rotation & 1 == 1;

        // k10 patterns indices 0..4
        for pattern_idx in 0..4 {
            let lut_idx = rotation * 4 + pattern_idx;
            k10[lut_idx] = build_fused_table_k10(&perm.k10[lut_idx], swap_colors);
        }

        // k8 patterns indices 4..8 (local 0..4)
        for pattern_idx in 0..4 {
            let lut_idx = rotation * 4 + pattern_idx;
            k8[lut_idx] = build_fused_table_k8(&perm.k8[lut_idx], swap_colors);
        }

        // k7 patterns indices 8..10 (local 0..2)
        for pattern_idx in 0..2 {
            let lut_idx = rotation * 2 + pattern_idx;
            k7[lut_idx] = build_fused_table_k7(&perm.k7[lut_idx], swap_colors);
        }

        // k6 patterns indices 10..12 (local 0..2)
        for pattern_idx in 0..2 {
            let lut_idx = rotation * 2 + pattern_idx;
            k6[lut_idx] = build_fused_table_k6(&perm.k6[lut_idx], swap_colors);
        }

        // k5 patterns indices 12..14 (local 0..2)
        for pattern_idx in 0..2 {
            let lut_idx = rotation * 2 + pattern_idx;
            k5[lut_idx] = build_fused_table_k5(&perm.k5[lut_idx], swap_colors);
        }
    }

    FusedCaches {
        k10,
        k8,
        k7,
        k6,
        k5,
    }
}

#[inline(always)]
fn get_fused_caches(patterns: &[crate::pattern::Pattern]) -> &FusedCaches {
    FUSED_CACHES.get_or_init(|| build_fused_caches(patterns))
}

/// PEXT命令を使用して3進数インデックスを抽出（k=10用）
///
/// BMI2のPEXT命令でビット抽出を一括処理し、LUTで3進数変換。
/// Intel Haswell以降で3サイクルのレイテンシ。
///
/// # Safety
///
/// BMI2命令を使用するため、対応CPUでのみ実行可能。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
pub unsafe fn extract_index_pext_k10(
    black: u64,
    white: u64,
    mask: u64,
    swap_colors: bool,
    lut: &[u16; 1024],
) -> usize {
    let black_pext = _pext_u64(black, mask) as usize;
    let white_pext = _pext_u64(white, mask) as usize;

    // PEXTビット順→配列順に並べ替え（キャッシュ済みLUT使用）
    let black_bits = lut[black_pext] as usize;
    let white_bits = lut[white_pext] as usize;

    if swap_colors {
        TERNARY_LUT_K10[white_bits][black_bits] as usize
    } else {
        TERNARY_LUT_K10[black_bits][white_bits] as usize
    }
}

#[inline(always)]
unsafe fn extract_index_pext_k10_fused(black: u64, white: u64, mask: u64, fused: &[u16]) -> usize {
    const SIZE: usize = 1024;
    let black_pext = unsafe { _pext_u64(black, mask) as usize };
    let white_pext = unsafe { _pext_u64(white, mask) as usize };
    // SAFETY: fused table is pre-sized to SIZE * SIZE
    unsafe { *fused.get_unchecked(black_pext * SIZE + white_pext) as usize }
}

/// PEXT命令を使用して3進数インデックスを抽出（k=8用）
///
/// # Safety
///
/// BMI2命令を使用するため、対応CPUでのみ実行可能。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
pub unsafe fn extract_index_pext_k8(
    black: u64,
    white: u64,
    mask: u64,
    swap_colors: bool,
    lut: &[u16; 256],
) -> usize {
    let black_pext = _pext_u64(black, mask) as usize;
    let white_pext = _pext_u64(white, mask) as usize;

    let black_bits = lut[black_pext] as usize;
    let white_bits = lut[white_pext] as usize;

    if swap_colors {
        TERNARY_LUT_K8[white_bits][black_bits] as usize
    } else {
        TERNARY_LUT_K8[black_bits][white_bits] as usize
    }
}

#[inline(always)]
unsafe fn extract_index_pext_k8_fused(black: u64, white: u64, mask: u64, fused: &[u16]) -> usize {
    const SIZE: usize = 256;
    let black_pext = unsafe { _pext_u64(black, mask) as usize };
    let white_pext = unsafe { _pext_u64(white, mask) as usize };
    unsafe { *fused.get_unchecked(black_pext * SIZE + white_pext) as usize }
}

/// PEXT命令を使用して3進数インデックスを抽出（k=7用）
///
/// # Safety
///
/// BMI2命令を使用するため、対応CPUでのみ実行可能。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
pub unsafe fn extract_index_pext_k7(
    black: u64,
    white: u64,
    mask: u64,
    swap_colors: bool,
    lut: &[u16; 128],
) -> usize {
    let black_pext = _pext_u64(black, mask) as usize;
    let white_pext = _pext_u64(white, mask) as usize;

    let black_bits = lut[black_pext] as usize;
    let white_bits = lut[white_pext] as usize;

    if swap_colors {
        TERNARY_LUT_K7[white_bits][black_bits] as usize
    } else {
        TERNARY_LUT_K7[black_bits][white_bits] as usize
    }
}

#[inline(always)]
unsafe fn extract_index_pext_k7_fused(black: u64, white: u64, mask: u64, fused: &[u16]) -> usize {
    const SIZE: usize = 128;
    let black_pext = unsafe { _pext_u64(black, mask) as usize };
    let white_pext = unsafe { _pext_u64(white, mask) as usize };
    unsafe { *fused.get_unchecked(black_pext * SIZE + white_pext) as usize }
}

/// PEXT命令を使用して3進数インデックスを抽出（k=6用）
///
/// # Safety
///
/// BMI2命令を使用するため、対応CPUでのみ実行可能。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
pub unsafe fn extract_index_pext_k6(
    black: u64,
    white: u64,
    mask: u64,
    swap_colors: bool,
    lut: &[u16; 64],
) -> usize {
    let black_pext = _pext_u64(black, mask) as usize;
    let white_pext = _pext_u64(white, mask) as usize;

    let black_bits = lut[black_pext] as usize;
    let white_bits = lut[white_pext] as usize;

    if swap_colors {
        TERNARY_LUT_K6[white_bits][black_bits] as usize
    } else {
        TERNARY_LUT_K6[black_bits][white_bits] as usize
    }
}

#[inline(always)]
unsafe fn extract_index_pext_k6_fused(black: u64, white: u64, mask: u64, fused: &[u16]) -> usize {
    const SIZE: usize = 64;
    let black_pext = unsafe { _pext_u64(black, mask) as usize };
    let white_pext = unsafe { _pext_u64(white, mask) as usize };
    unsafe { *fused.get_unchecked(black_pext * SIZE + white_pext) as usize }
}

/// PEXT命令を使用して3進数インデックスを抽出（k=5用）
///
/// # Safety
///
/// BMI2命令を使用するため、対応CPUでのみ実行可能。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
pub unsafe fn extract_index_pext_k5(
    black: u64,
    white: u64,
    mask: u64,
    swap_colors: bool,
    lut: &[u16; 32],
) -> usize {
    let black_pext = _pext_u64(black, mask) as usize;
    let white_pext = _pext_u64(white, mask) as usize;

    let black_bits = lut[black_pext] as usize;
    let white_bits = lut[white_pext] as usize;

    if swap_colors {
        TERNARY_LUT_K5[white_bits][black_bits] as usize
    } else {
        TERNARY_LUT_K5[black_bits][white_bits] as usize
    }
}

#[inline(always)]
unsafe fn extract_index_pext_k5_fused(black: u64, white: u64, mask: u64, fused: &[u16]) -> usize {
    const SIZE: usize = 32;
    let black_pext = unsafe { _pext_u64(black, mask) as usize };
    let white_pext = unsafe { _pext_u64(white, mask) as usize };
    unsafe { *fused.get_unchecked(black_pext * SIZE + white_pext) as usize }
}

/// BMI2が利用可能かどうかをランタイムで判定
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_bmi2() -> bool {
    is_x86_feature_detected!("bmi2")
}

/// 全パターンインデックスをPEXT命令で抽出（安全なラッパー）
///
/// BMI2が利用可能な場合のみ呼び出すこと。
/// 内部でunsafe関数を呼び出す。
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn extract_all_patterns_pext_safe(
    black: u64,
    white: u64,
    patterns: &[crate::pattern::Pattern],
    out: &mut [usize; 56],
) {
    debug_assert!(has_bmi2(), "BMI2 must be available");
    // Safety: has_bmi2()で事前チェック済み
    unsafe {
        extract_all_patterns_pext(black, white, patterns, out);
    }
}

/// 全パターンインデックスをPEXT命令で抽出（BMI2対応CPU用）
///
/// 14パターン × 4回転 = 56個のインデックスを高速抽出。
/// BMI2非対応環境では呼び出し禁止。
///
/// # Safety
///
/// BMI2命令を使用するため、`has_bmi2()`がtrueの場合のみ呼び出し可能。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
pub unsafe fn extract_all_patterns_pext(
    black: u64,
    white: u64,
    patterns: &[crate::pattern::Pattern],
    out: &mut [usize; 56],
) {
    debug_assert_eq!(patterns.len(), 14);

    let fused = get_fused_caches(patterns);

    // パターンサイズ: P01-P04=10, P05-P08=8, P09-P10=7, P11-P12=6, P13-P14=5
    for rotation in 0..4 {
        let base_idx = rotation * 14;

        // P01-P04 (k=10)
        for pattern_idx in 0..4 {
            let pattern = &patterns[pattern_idx];
            let mask = pattern.rotated_masks[rotation];
            let lut = &fused.k10[rotation * 4 + pattern_idx];
            out[base_idx + pattern_idx] =
                unsafe { extract_index_pext_k10_fused(black, white, mask, lut) };
        }

        // P05-P08 (k=8)
        for pattern_idx in 4..8 {
            let pattern = &patterns[pattern_idx];
            let mask = pattern.rotated_masks[rotation];
            let lut = &fused.k8[rotation * 4 + (pattern_idx - 4)];
            out[base_idx + pattern_idx] =
                unsafe { extract_index_pext_k8_fused(black, white, mask, lut) };
        }

        // P09-P10 (k=7)
        for pattern_idx in 8..10 {
            let pattern = &patterns[pattern_idx];
            let mask = pattern.rotated_masks[rotation];
            let lut = &fused.k7[rotation * 2 + (pattern_idx - 8)];
            out[base_idx + pattern_idx] =
                unsafe { extract_index_pext_k7_fused(black, white, mask, lut) };
        }

        // P11-P12 (k=6)
        for pattern_idx in 10..12 {
            let pattern = &patterns[pattern_idx];
            let mask = pattern.rotated_masks[rotation];
            let lut = &fused.k6[rotation * 2 + (pattern_idx - 10)];
            out[base_idx + pattern_idx] =
                unsafe { extract_index_pext_k6_fused(black, white, mask, lut) };
        }

        // P13-P14 (k=5)
        for pattern_idx in 12..14 {
            let pattern = &patterns[pattern_idx];
            let mask = pattern.rotated_masks[rotation];
            let lut = &fused.k5[rotation * 2 + (pattern_idx - 12)];
            out[base_idx + pattern_idx] =
                unsafe { extract_index_pext_k5_fused(black, white, mask, lut) };
        }
    }
}

// ============================================================================
// SIMD Score Conversion
// ============================================================================

/// SSE4.1を使用してu16型の評価値8個をf32型の石差に一括変換
///
/// # 変換式
///
/// 各要素について `(value - 32768.0) / 256.0` を並列実行
///
/// # Arguments
///
/// * `values` - u16型の評価値8個の配列
///
/// # Returns
///
/// f32型の石差8個の配列
///
/// # Platform Support
///
/// この関数はx86-64アーキテクチャでSSE4.1がサポートされている場合のみ利用可能。
///
/// # Safety
///
/// この関数はSSE4.1命令を使用するため、対応CPUでのみ実行可能。
/// `is_x86_feature_detected!("sse4.1")` で事前確認すること。
///
/// # Examples
///
/// ```ignore
/// #[cfg(target_arch = "x86_64")]
/// use prismind::x86_64::u16_to_score_simd_sse;
///
/// #[cfg(target_arch = "x86_64")]
/// {
///     if is_x86_feature_detected!("sse4.1") {
///         let values: [u16; 8] = [0, 10000, 20000, 32768, 40000, 50000, 60000, 65535];
///         let scores = unsafe { u16_to_score_simd_sse(&values) };
///         assert_eq!(scores[0], -128.0);
///         assert_eq!(scores[3], 0.0);
///     }
/// }
/// ```
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
pub unsafe fn u16_to_score_simd_sse(values: &[u16; 8]) -> [f32; 8] {
    unsafe {
        // 8個のu16を2つの128ビットレジスタにロード（各4個ずつ）
        // SSE4.1のpmovzxwd命令でu16→u32拡張

        // 下位4個: values[0..4]
        let v_low_u16 = _mm_loadl_epi64(values.as_ptr() as *const __m128i);
        let v_low_u32 = _mm_cvtepu16_epi32(v_low_u16);
        let v_low_f32 = _mm_cvtepi32_ps(v_low_u32);

        // 上位4個: values[4..8]
        let v_high_u16 = _mm_loadl_epi64(values.as_ptr().add(4) as *const __m128i);
        let v_high_u32 = _mm_cvtepu16_epi32(v_high_u16);
        let v_high_f32 = _mm_cvtepi32_ps(v_high_u32);

        // (value - 32768.0) / 256.0 の計算
        let offset = _mm_set1_ps(32768.0);
        let scale = _mm_set1_ps(1.0 / 256.0);

        let result_low = _mm_mul_ps(_mm_sub_ps(v_low_f32, offset), scale);
        let result_high = _mm_mul_ps(_mm_sub_ps(v_high_f32, offset), scale);

        // 結果を配列に格納
        let mut out = [0.0f32; 8];
        _mm_storeu_ps(out.as_mut_ptr(), result_low);
        _mm_storeu_ps(out.as_mut_ptr().add(4), result_high);
        out
    }
}

/// AVX2を使用してu16型の評価値8個をf32型の石差に一括変換
///
/// # 変換式
///
/// 各要素について `(value - 32768.0) / 256.0` を並列実行
///
/// # Arguments
///
/// * `values` - u16型の評価値8個の配列
///
/// # Returns
///
/// f32型の石差8個の配列
///
/// # Platform Support
///
/// この関数はx86-64アーキテクチャでAVX2がサポートされている場合のみ利用可能。
///
/// # Safety
///
/// この関数はAVX2命令を使用するため、対応CPUでのみ実行可能。
/// `is_x86_feature_detected!("avx2")` で事前確認すること。
///
/// # Examples
///
/// ```ignore
/// #[cfg(target_arch = "x86_64")]
/// use prismind::x86_64::u16_to_score_simd_avx2;
///
/// #[cfg(target_arch = "x86_64")]
/// {
///     if is_x86_feature_detected!("avx2") {
///         let values: [u16; 8] = [0, 10000, 20000, 32768, 40000, 50000, 60000, 65535];
///         let scores = unsafe { u16_to_score_simd_avx2(&values) };
///         assert_eq!(scores[0], -128.0);
///         assert_eq!(scores[3], 0.0);
///     }
/// }
/// ```
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn u16_to_score_simd_avx2(values: &[u16; 8]) -> [f32; 8] {
    unsafe {
        // 8個のu16を128ビットレジスタにロード
        let v_u16 = _mm_loadu_si128(values.as_ptr() as *const __m128i);

        // AVX2のvpmovzxwd命令でu16→u32拡張（8個一括）
        let v_u32 = _mm256_cvtepu16_epi32(v_u16);

        // u32→f32変換
        let v_f32 = _mm256_cvtepi32_ps(v_u32);

        // (value - 32768.0) / 256.0 の計算
        let offset = _mm256_set1_ps(32768.0);
        let scale = _mm256_set1_ps(1.0 / 256.0);

        let result = _mm256_mul_ps(_mm256_sub_ps(v_f32, offset), scale);

        // 結果を配列に格納
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), result);
        out
    }
}

/// x86-64用の安全なSIMD変換ラッパー
///
/// ランタイムでCPU機能を検出し、利用可能な最速のSIMD実装を選択する:
/// 1. AVX2が利用可能 → `u16_to_score_simd_avx2`
/// 2. SSE4.1が利用可能 → `u16_to_score_simd_sse`
/// 3. どちらも利用不可 → スカラー実装にフォールバック
///
/// # Arguments
///
/// * `values` - u16型の評価値8個の配列
///
/// # Returns
///
/// f32型の石差8個の配列
///
/// # Platform Support
///
/// この関数はx86-64アーキテクチャでのみ利用可能。
///
/// # Examples
///
/// ```ignore
/// #[cfg(target_arch = "x86_64")]
/// use prismind::x86_64::u16_to_score_simd_x86_64;
///
/// #[cfg(target_arch = "x86_64")]
/// {
///     let values: [u16; 8] = [0, 10000, 20000, 32768, 40000, 50000, 60000, 65535];
///     let scores = u16_to_score_simd_x86_64(&values);
///     assert_eq!(scores[0], -128.0);
///     assert_eq!(scores[3], 0.0);
/// }
/// ```
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn u16_to_score_simd_x86_64(values: &[u16; 8]) -> [f32; 8] {
    // ランタイムでCPU機能を検出
    if is_x86_feature_detected!("avx2") {
        unsafe { u16_to_score_simd_avx2(values) }
    } else if is_x86_feature_detected!("sse4.1") {
        unsafe { u16_to_score_simd_sse(values) }
    } else {
        // フォールバック: スカラー実装
        let mut scores = [0.0f32; 8];
        for i in 0..8 {
            scores[i] = (values[i] as f32 - 32768.0) / 256.0;
        }
        scores
    }
}

/// x86-64プリフェッチヒント
///
/// 次にアクセスする予定のメモリアドレスをCPUに事前通知し、
/// キャッシュミスを削減する。
///
/// # Arguments
///
/// * `ptr` - プリフェッチするメモリアドレス
///
/// # Safety
///
/// ポインタは有効なメモリアドレスを指している必要がある。
/// ただし、プリフェッチはヒントであり、無効なアドレスでも
/// プログラムはクラッシュしない（無視されるだけ）。
///
/// # Platform Support
///
/// x86-64専用。SSEがサポートされている場合に使用。
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn prefetch_x86_64<T>(ptr: *const T) {
    // SSEのprefetch命令（_MM_HINT_T0 = 最も近いキャッシュレベルに読み込み）
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_u16_to_score_simd_x86_64_basic() {
        let values: [u16; 8] = [0, 10000, 20000, 32768, 40000, 50000, 60000, 65535];
        let scores = u16_to_score_simd_x86_64(&values);

        // SIMD版とスカラー版が同じ結果を返すことを確認
        for i in 0..8 {
            let expected = (values[i] as f32 - 32768.0) / 256.0;
            assert!(
                (scores[i] - expected).abs() < 0.0001,
                "SIMD conversion at index {} failed: expected {}, got {}",
                i,
                expected,
                scores[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_u16_to_score_simd_x86_64_boundary_values() {
        let values: [u16; 8] = [0, 0, 32768, 32768, 65535, 65535, 1000, 60000];
        let scores = u16_to_score_simd_x86_64(&values);

        // 境界値の検証
        assert!(
            (scores[0] - (-128.0)).abs() < 0.0001,
            "SIMD: u16 0 should be -128.0"
        );
        assert!(
            (scores[2] - 0.0).abs() < 0.0001,
            "SIMD: u16 32768 should be 0.0"
        );
        assert!(
            (scores[4] - 127.996_09).abs() < 0.0001,
            "SIMD: u16 65535 should be ~127.996"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_u16_to_score_simd_sse_if_available() {
        if is_x86_feature_detected!("sse4.1") {
            let values: [u16; 8] = [0, 10000, 20000, 32768, 40000, 50000, 60000, 65535];
            let scores = unsafe { u16_to_score_simd_sse(&values) };

            for i in 0..8 {
                let expected = (values[i] as f32 - 32768.0) / 256.0;
                assert!(
                    (scores[i] - expected).abs() < 0.0001,
                    "SSE conversion at index {} failed: expected {}, got {}",
                    i,
                    expected,
                    scores[i]
                );
            }
        } else {
            println!("SSE4.1 not available, skipping test");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_u16_to_score_simd_avx2_if_available() {
        if is_x86_feature_detected!("avx2") {
            let values: [u16; 8] = [0, 10000, 20000, 32768, 40000, 50000, 60000, 65535];
            let scores = unsafe { u16_to_score_simd_avx2(&values) };

            for i in 0..8 {
                let expected = (values[i] as f32 - 32768.0) / 256.0;
                assert!(
                    (scores[i] - expected).abs() < 0.0001,
                    "AVX2 conversion at index {} failed: expected {}, got {}",
                    i,
                    expected,
                    scores[i]
                );
            }
        } else {
            println!("AVX2 not available, skipping test");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_prefetch_x86_64_does_not_crash() {
        // プリフェッチが有効なポインタで動作することを確認
        let data = [1u16, 2, 3, 4, 5];
        unsafe {
            prefetch_x86_64(data.as_ptr());
        }
        // クラッシュしなければOK
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_detect_cpu_features() {
        println!("SSE4.1 available: {}", is_x86_feature_detected!("sse4.1"));
        println!("AVX2 available: {}", is_x86_feature_detected!("avx2"));
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn test_x86_64_module_not_available_on_other_platforms() {
        // x86-64以外のプラットフォームでは、このモジュールの機能は利用不可
        // このテストはコンパイルが通ることを確認するだけ
        println!("x86_64 module is not available on this platform");
    }
}
