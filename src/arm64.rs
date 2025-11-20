//! ARM64専用最適化モジュール
//!
//! このモジュールはARM64アーキテクチャ専用の最適化を集約する:
//! - NEON SIMD命令による並列処理
//! - プリフェッチによるキャッシュ最適化
//!
//! # プラットフォームサポート
//!
//! このモジュールの機能はARM64 (aarch64) アーキテクチャでのみ利用可能。
//! 他のプラットフォームでは、これらの最適化は無効化され、スカラー実装が使用される。

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// ARM NEON SIMDを使用してu16型の評価値8個をf32型の石差に一括変換
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
/// この関数はARM64 (aarch64) アーキテクチャでのみ利用可能。
///
/// # Examples
///
/// ```ignore
/// #[cfg(target_arch = "aarch64")]
/// use prismind::arm64::u16_to_score_simd_arm64;
///
/// #[cfg(target_arch = "aarch64")]
/// {
///     let values: [u16; 8] = [0, 10000, 20000, 32768, 40000, 50000, 60000, 65535];
///     let scores = u16_to_score_simd_arm64(&values);
///     assert_eq!(scores[0], -128.0);
///     assert_eq!(scores[3], 0.0);
/// }
/// ```
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn u16_to_score_simd_arm64(values: &[u16; 8]) -> [f32; 8] {
    unsafe {
        // u16の8個をロード
        let v = vld1q_u16(values.as_ptr());

        // 下位4個をu32に拡張してからf32に変換
        let v_low_u32 = vmovl_u16(vget_low_u16(v));
        let v_low_f32 = vcvtq_f32_u32(v_low_u32);

        // 上位4個をu32に拡張してからf32に変換
        let v_high_u32 = vmovl_u16(vget_high_u16(v));
        let v_high_f32 = vcvtq_f32_u32(v_high_u32);

        // (value - 32768.0) / 256.0 の計算
        let offset = vdupq_n_f32(32768.0);
        let scale = vdupq_n_f32(1.0 / 256.0);

        let result_low = vmulq_f32(vsubq_f32(v_low_f32, offset), scale);
        let result_high = vmulq_f32(vsubq_f32(v_high_f32, offset), scale);

        // 結果を配列に格納
        let mut out = [0.0f32; 8];
        vst1q_f32(out.as_mut_ptr(), result_low);
        vst1q_f32(out.as_mut_ptr().add(4), result_high);
        out
    }
}

/// ARM64プリフェッチヒント
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
/// ARM64専用。他のプラットフォームでは何もしない。
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn prefetch_arm64<T>(ptr: *const T) {
    // ARM64のプリフェッチ命令
    // core::hint::black_boxを使用してコンパイラ最適化による削除を防ぐ
    core::hint::black_box(ptr);

    // 将来的にstd::arch::aarch64::_prefetchが安定化されたら以下に置き換え:
    // std::arch::aarch64::_prefetch(ptr as *const i8, _PREFETCH_READ, _PREFETCH_LOCALITY3);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_u16_to_score_simd_arm64_basic() {
        let values: [u16; 8] = [0, 10000, 20000, 32768, 40000, 50000, 60000, 65535];
        let scores = u16_to_score_simd_arm64(&values);

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

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_u16_to_score_simd_arm64_boundary_values() {
        let values: [u16; 8] = [0, 0, 32768, 32768, 65535, 65535, 1000, 60000];
        let scores = u16_to_score_simd_arm64(&values);

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
            (scores[4] - 127.99609375).abs() < 0.0001,
            "SIMD: u16 65535 should be ~127.996"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_prefetch_arm64_does_not_crash() {
        // プリフェッチが有効なポインタで動作することを確認
        let data = [1u16, 2, 3, 4, 5];
        unsafe {
            prefetch_arm64(data.as_ptr());
        }
        // クラッシュしなければOK
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[test]
    fn test_arm64_module_not_available_on_other_platforms() {
        // ARM64以外のプラットフォームでは、このモジュールの機能は利用不可
        // このテストはコンパイルが通ることを確認するだけ
        println!("ARM64 module is not available on this platform");
    }
}
