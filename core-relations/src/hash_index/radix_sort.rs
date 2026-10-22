//! Stable value radix sorting with adaptive digit width.
//!
//! Input pairs arrive in RowId order. Stable value passes retain that order
//! within each key, as required by index subsets.

use super::{RowId, Value};
use crate::numeric_id::NumericId;

type Pair = (Value, RowId);

const WIDE_RADIX_MIN_ROWS: usize = 8192;

pub(super) fn sort(data: &mut [Pair], scratch: &mut [Pair]) {
    if data.len() < 64 {
        // Preserve the existing tiny-input tuple comparison path.
        data.sort_unstable();
        return;
    }
    let metadata = scan_metadata(data);
    if metadata.sorted {
        return;
    }
    // Histograms use u32 counters. Real projected row sets fit RowId's u32
    // representation; keep the slice API safe for oversized hypothetical input.
    if u32::try_from(data.len()).is_err() {
        data.sort_by_key(|pair| pair.0);
    } else if digit_bits(data.len(), metadata.varying) == 11 {
        radix_digits::<2048>(data, scratch, metadata, 11);
    } else {
        radix_digits::<256>(data, scratch, metadata, 8);
    }
}

fn digit_bits(len: usize, varying: u32) -> u32 {
    if len < WIDE_RADIX_MIN_ROWS {
        return 8;
    }
    let byte_passes = u32::from(varying & 0x0000_00ff != 0)
        + u32::from(varying & 0x0000_ff00 != 0)
        + u32::from(varying & 0x00ff_0000 != 0)
        + u32::from(varying & 0xff00_0000 != 0);
    let wide_passes = u32::from(varying & 0x0000_07ff != 0)
        + u32::from(varying & 0x003f_f800 != 0)
        + u32::from(varying & 0xffc0_0000 != 0);
    // Use the wider histogram only when it eliminates a full pass. Digits
    // constant across the input are already skipped by either implementation.
    if wide_passes < byte_passes { 11 } else { 8 }
}

#[derive(Clone, Copy)]
struct SortMetadata {
    varying: u32,
    sorted: bool,
}

fn scan_metadata(data: &[Pair]) -> SortMetadata {
    let first = data[0].0.rep();
    let mut prev = first;
    let mut metadata = SortMetadata {
        varying: 0,
        sorted: true,
    };
    for &(value, _) in data {
        let value = value.rep();
        metadata.varying |= first ^ value;
        metadata.sorted &= prev <= value;
        prev = value;
    }
    metadata
}

fn radix_digits<const BUCKETS: usize>(
    data: &mut [Pair],
    scratch: &mut [Pair],
    metadata: SortMetadata,
    digit_bits: u32,
) {
    debug_assert_eq!(BUCKETS, 1usize << digit_bits);
    let len = data.len();
    let mut src = data;
    let mut dst = &mut scratch[..len];
    let mask = (BUCKETS - 1) as u32;
    let mut passes = 0;
    let mut shift = 0;
    while shift < 32 {
        if (metadata.varying >> shift) & mask != 0 {
            // Byte digits use a 1KiB histogram; 11-bit digits use 8KiB
            // and need at most three radix passes.
            let mut count = [0u32; BUCKETS];
            for pair in src.iter() {
                count[((pair.0.rep() >> shift) & mask) as usize] += 1;
            }
            let mut sum = 0;
            for count in &mut count {
                let old = *count;
                *count = sum;
                sum += old;
            }
            for &pair in src.iter() {
                let bucket = ((pair.0.rep() >> shift) & mask) as usize;
                dst[count[bucket] as usize] = pair;
                count[bucket] += 1;
            }
            std::mem::swap(&mut src, &mut dst);
            passes += 1;
        }
        shift += digit_bits;
    }
    if passes % 2 == 1 {
        dst.copy_from_slice(src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(len: usize, kind: usize) -> Vec<Pair> {
        (0..len)
            .map(|row| {
                let mixed = (row as u32).wrapping_mul(0x9e3779b9).rotate_left(11);
                let value = match kind {
                    0 => mixed,
                    1 => 0x1200_0000 | (mixed & 255),
                    2 => 0x1234_0000 | (mixed & 0xff00),
                    3 => mixed & 0xffff,
                    4 => mixed & 0xffffff,
                    5 => (row / 7) as u32,
                    6 => 42,
                    _ => 255 - (row % 256) as u32,
                };
                (Value::new(value), RowId::from_usize(row))
            })
            .collect()
    }

    #[test]
    fn adaptive_radix_matches_reference_across_sizes_and_distributions() {
        for len in [
            0, 1, 63, 64, 65, 255, 256, 257, 1025, 8191, 8192, 8193, 16385,
        ] {
            for kind in 0..8 {
                let mut actual = input(len, kind);
                let mut expected = actual.clone();
                expected.sort_by_key(|pair| pair.0);
                let mut scratch = actual.clone();
                sort(&mut actual, &mut scratch);
                assert_eq!(actual, expected, "len={len}, kind={kind}");
            }
        }
    }

    #[test]
    fn adaptive_radix_preserves_equal_key_order_and_copy_parity() {
        let keys = [0, 1, 2047, 2048, (1 << 22) - 1, 1 << 22, u32::MAX];
        for len in [65, 8191, 8192, 8193] {
            for kind in 0..4 {
                let mut actual: Vec<_> = (0..len)
                    .map(|row| {
                        let mixed = (row as u32).wrapping_mul(0x9e3779b9).rotate_left(11);
                        let value = match kind {
                            0 => keys[row * 13 % keys.len()],
                            1 => 0x8000_0000 | (mixed & 0x7ff),
                            2 => mixed & 0x003f_ffff,
                            _ => (mixed & 0x3ff) | ((mixed & 0x3ff) << 22),
                        };
                        // Descending RowIds distinguish stable value sorting from
                        // tuple comparison sorting, including equal high digits.
                        (Value::new(value), RowId::from_usize(len - row))
                    })
                    .collect();
                let mut expected = actual.clone();
                expected.sort_by_key(|pair| pair.0);
                let mut scratch = actual.clone();
                sort(&mut actual, &mut scratch);
                assert_eq!(actual, expected, "len={len}, kind={kind}");
            }
        }
    }

    #[test]
    fn adaptive_radix_sorted_inputs_do_not_need_ping_pong_storage() {
        for len in [64, 8191, 8192, 16385] {
            for kind in [5, 6] {
                let mut actual = input(len, kind);
                let expected = actual.clone();
                sort(&mut actual, &mut []);
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn adaptive_radix_requires_fewer_effective_wide_passes() {
        assert_eq!(digit_bits(8191, u32::MAX), 8);
        assert_eq!(digit_bits(8192, u32::MAX), 11); // 4 -> 3
        assert_eq!(digit_bits(8193, 0x003f_ffff), 11); // 3 -> 2
        assert_eq!(digit_bits(8192, 0x0000_ffff), 8); // 2 == 2
        assert_eq!(digit_bits(8192, 0x00ff_ffff), 8); // 3 == 3
        assert_eq!(digit_bits(8192, (1 << 10) | (1 << 11)), 8); // 1 -> 2
        assert_eq!(digit_bits(8192, (1 << 21) | (1 << 22)), 8); // 1 -> 2
        assert_eq!(digit_bits(8192, 0), 8);
    }

    #[test]
    fn adaptive_radix_equal_and_reversed_pass_cases_remain_stable() {
        for len in [8191, 8192, 8193] {
            for varying in [
                0xffff,
                0xffffff,
                (1 << 10) | (1 << 11),
                (1 << 21) | (1 << 22),
                0x003f_ffff,
                u32::MAX,
            ] {
                let mut actual: Vec<_> = (0..len)
                    .map(|row| {
                        let mixed = (row as u32).wrapping_mul(0x9e3779b9).rotate_left(11);
                        (Value::new(mixed & varying), RowId::from_usize(len - row))
                    })
                    .collect();
                let mut expected = actual.clone();
                expected.sort_by_key(|pair| pair.0);
                let mut scratch = actual.clone();
                sort(&mut actual, &mut scratch);
                assert_eq!(actual, expected, "len={len}, varying={varying:#x}");
            }
        }
    }
}
