//! x86-64専用最適化モジュール
//!
//! このモジュールはx86-64アーキテクチャ専用の最適化を集約する:
//! - SSE/AVX SIMD命令による並列処理
//! - プリフェッチによるキャッシュ最適化
//!
//! # プラットフォームサポート
//!
//! このモジュールの機能はx86-64 (x86_64) アーキテクチャでのみ利用可能。
//! 他のプラットフォームでは、これらの最適化は無効化され、スカラー実装が使用される。

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
