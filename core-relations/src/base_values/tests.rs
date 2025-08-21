use super::{BaseValuePrinter, BaseValues};
use rand::{Rng, SeedableRng};

#[test]
fn base_printing() {
    let mut bases = BaseValues::default();
    bases.register_type::<i64>();
    let ty = bases.get_ty::<i64>();
    let val = bases.get(24i64);
    assert_eq!(
        format!(
            "{:?}",
            BaseValuePrinter {
                base: &bases,
                ty,
                val
            }
        ),
        "24"
    );
}

#[test]
fn roundtrip_small_integers() {
    let mut bases = BaseValues::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Test u8
    bases.register_type::<u8>();
    for val in [0u8, 1, 127, 255] {
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<u8>(boxed);
        assert_eq!(val, unboxed);
    }
    // Random u8 samples
    for _ in 0..100 {
        let val: u8 = rng.gen();
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<u8>(boxed);
        assert_eq!(val, unboxed);
    }

    // Test u16
    bases.register_type::<u16>();
    for val in [0u16, 1, 255, 256, 65535] {
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<u16>(boxed);
        assert_eq!(val, unboxed);
    }
    // Random u16 samples
    for _ in 0..100 {
        let val: u16 = rng.gen();
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<u16>(boxed);
        assert_eq!(val, unboxed);
    }

    // Test u32
    bases.register_type::<u32>();
    for val in [0u32, 1, 255, 65536, 2147483647, 4294967295] {
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<u32>(boxed);
        assert_eq!(val, unboxed);
    }
    // Random u32 samples
    for _ in 0..100 {
        let val: u32 = rng.gen();
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<u32>(boxed);
        assert_eq!(val, unboxed);
    }

    // Test i8
    bases.register_type::<i8>();
    for val in [-128i8, -1, 0, 1, 127] {
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<i8>(boxed);
        assert_eq!(val, unboxed);
    }
    // Random i8 samples
    for _ in 0..100 {
        let val: i8 = rng.gen();
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<i8>(boxed);
        assert_eq!(val, unboxed);
    }

    // Test i16
    bases.register_type::<i16>();
    for val in [-32768i16, -1, 0, 1, 32767] {
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<i16>(boxed);
        assert_eq!(val, unboxed);
    }
    // Random i16 samples
    for _ in 0..100 {
        let val: i16 = rng.gen();
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<i16>(boxed);
        assert_eq!(val, unboxed);
    }

    // Test i32
    bases.register_type::<i32>();
    for val in [-2147483648i32, -1, 0, 1, 2147483647] {
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<i32>(boxed);
        assert_eq!(val, unboxed);
    }
    // Random i32 samples
    for _ in 0..100 {
        let val: i32 = rng.gen();
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<i32>(boxed);
        assert_eq!(val, unboxed);
    }
}

#[test]
fn roundtrip_bool() {
    let mut bases = BaseValues::default();
    bases.register_type::<bool>();

    for val in [true, false] {
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<bool>(boxed);
        assert_eq!(val, unboxed);
    }

    // Random bool samples
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for _ in 0..100 {
        let val: bool = rng.gen();
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<bool>(boxed);
        assert_eq!(val, unboxed);
    }
}

#[test]
fn roundtrip_unit() {
    let mut bases = BaseValues::default();
    bases.register_type::<()>();

    let val = ();
    let boxed = bases.get(val);
    bases.unwrap::<()>(boxed);
}

#[test]
fn roundtrip_medium_integers_unboxable() {
    let mut bases = BaseValues::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Test u64 values that fit in 31 bits (unboxable)
    bases.register_type::<u64>();
    for val in [0u64, 1, 1000, 2147483647] {
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<u64>(boxed);
        assert_eq!(val, unboxed);
    }
    for _ in 0..100 {
        let val: u64 = rng.gen_range(0..=2147483647);
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<u64>(boxed);
        assert_eq!(val, unboxed);
    }

    bases.register_type::<i64>();
    for val in [0, 1, 1000, 2147483647] {
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<i64>(boxed);
        assert_eq!(val, unboxed);
    }
    for _ in 0..100 {
        let val: i64 = rng.gen_range(0..=2147483647);
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<i64>(boxed);
        assert_eq!(val, unboxed);
    }

    bases.register_type::<usize>();
    for val in [0usize, 1, 1000, 2147483647] {
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<usize>(boxed);
        assert_eq!(val, unboxed);
    }
    for _ in 0..100 {
        let val: usize = rng.gen_range(0..=2147483647);
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<usize>(boxed);
        assert_eq!(val, unboxed);
    }

    bases.register_type::<isize>();
    for val in [0, 1, 1000, 2147483647] {
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<isize>(boxed);
        assert_eq!(val, unboxed);
    }
    // Random isize samples (31-bit range)
    for _ in 0..100 {
        let val: isize = rng.gen_range(0..=2147483647);
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<isize>(boxed);
        assert_eq!(val, unboxed);
    }
}

#[test]
fn roundtrip_medium_integers_interned() {
    let mut bases = BaseValues::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Test u64 values that don't fit in 31 bits (need interning)
    bases.register_type::<u64>();
    for val in [2147483648u64, 4294967296, u64::MAX] {
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<u64>(boxed);
        assert_eq!(val, unboxed);
    }
    // Random u64 samples (large values)
    for _ in 0..100 {
        let val: u64 = rng.gen_range(2147483648..=u64::MAX);
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<u64>(boxed);
        assert_eq!(val, unboxed);
    }

    // Test i64 values that don't fit in 31 bits (need interning)
    bases.register_type::<i64>();
    for val in [-2147483649i64, i64::MIN, i64::MAX] {
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<i64>(boxed);
        assert_eq!(val, unboxed);
    }
    // Random i64 samples (values outside 31-bit range)
    for _ in 0..50 {
        let val: i64 = rng.gen_range(i64::MIN..0);
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<i64>(boxed);
        assert_eq!(val, unboxed);
    }
    for _ in 0..50 {
        let val: i64 = rng.gen_range(2147483648..=i64::MAX);
        let boxed = bases.get(val);
        let unboxed = bases.unwrap::<i64>(boxed);
        assert_eq!(val, unboxed);
    }

    // Test large usize values (need interning on 64-bit systems)
    bases.register_type::<usize>();
    if std::mem::size_of::<usize>() == 8 {
        for val in [2147483648usize, usize::MAX] {
            let boxed = bases.get(val);
            let unboxed = bases.unwrap::<usize>(boxed);
            assert_eq!(val, unboxed);
        }
        // Random usize samples (large values)
        for _ in 0..100 {
            let val: usize = rng.gen_range(2147483648..=usize::MAX);
            let boxed = bases.get(val);
            let unboxed = bases.unwrap::<usize>(boxed);
            assert_eq!(val, unboxed);
        }
    }

    // Test large isize values (need interning on 64-bit systems)
    bases.register_type::<isize>();
    if std::mem::size_of::<isize>() == 8 {
        for val in [-2147483649isize, isize::MIN, isize::MAX] {
            let boxed = bases.get(val);
            let unboxed = bases.unwrap::<isize>(boxed);
            assert_eq!(val, unboxed);
        }
        // Random isize samples (values outside 31-bit range)
        for _ in 0..50 {
            let val: isize = rng.gen_range(isize::MIN..0);
            let boxed = bases.get(val);
            let unboxed = bases.unwrap::<isize>(boxed);
            assert_eq!(val, unboxed);
        }
        for _ in 0..50 {
            let val: isize = rng.gen_range(2147483648..=isize::MAX);
            let boxed = bases.get(val);
            let unboxed = bases.unwrap::<isize>(boxed);
            assert_eq!(val, unboxed);
        }
    }
}
