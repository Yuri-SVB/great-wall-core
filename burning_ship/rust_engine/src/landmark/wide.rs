//! An unsigned 256-bit integer: just what the metric and the ladder need to
//! multiply 128-bit values exactly.

/// Unsigned 256-bit integer. Field order makes the derived ordering numeric.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct U256 {
    pub hi: u128,
    pub lo: u128,
}

const HALF_BITS: u32 = 64;
const HALF_MASK: u128 = (1u128 << HALF_BITS) - 1;
const BITS: u32 = 256;

impl U256 {
    pub const ZERO: U256 = U256 { hi: 0, lo: 0 };

    pub fn from_u128(v: u128) -> Self {
        U256 { hi: 0, lo: v }
    }

    /// The exact product of two 128-bit values.
    pub fn mul(a: u128, b: u128) -> Self {
        let (a1, a0) = (a >> HALF_BITS, a & HALF_MASK);
        let (b1, b0) = (b >> HALF_BITS, b & HALF_MASK);
        let p00 = a0 * b0;
        let p01 = a0 * b1;
        let p10 = a1 * b0;
        let p11 = a1 * b1;
        let mid = (p00 >> HALF_BITS) + (p01 & HALF_MASK) + (p10 & HALF_MASK);
        U256 {
            hi: p11 + (p01 >> HALF_BITS) + (p10 >> HALF_BITS) + (mid >> HALF_BITS),
            lo: (p00 & HALF_MASK) | (mid << HALF_BITS),
        }
    }

    /// `self + rhs`. Panics on overflow (callers stay far below 2²⁵⁶).
    pub fn add(self, rhs: U256) -> Self {
        let (lo, carry) = self.lo.overflowing_add(rhs.lo);
        let hi = self
            .hi
            .checked_add(rhs.hi)
            .and_then(|h| h.checked_add(carry as u128))
            .expect("U256 add overflow");
        U256 { hi, lo }
    }

    pub fn shl(self, k: u32) -> Self {
        match k {
            0 => self,
            k if k >= BITS => U256::ZERO,
            k if k >= 128 => U256 {
                hi: self.lo << (k - 128),
                lo: 0,
            },
            k => U256 {
                hi: (self.hi << k) | (self.lo >> (128 - k)),
                lo: self.lo << k,
            },
        }
    }

    pub fn shr(self, k: u32) -> Self {
        match k {
            0 => self,
            k if k >= BITS => U256::ZERO,
            k if k >= 128 => U256 {
                hi: 0,
                lo: self.hi >> (k - 128),
            },
            k => U256 {
                hi: self.hi >> k,
                lo: (self.lo >> k) | (self.hi << (128 - k)),
            },
        }
    }

    /// Number of significant bits (0 for zero).
    pub fn bits(self) -> u32 {
        if self.hi != 0 {
            BITS - self.hi.leading_zeros()
        } else {
            128 - self.lo.leading_zeros()
        }
    }
}

/// `⌊√v⌋`. A floating-point estimate only seeds Newton's iteration, which
/// starts above the root and ends on its exact floor whatever the seed.
pub fn isqrt(v: u128) -> u128 {
    if v == 0 {
        return 0;
    }
    let g = (v as f64).sqrt() as u128;
    // Above the root: the estimate is within 2^-50 of it.
    let mut r = g + (g >> SEED_MARGIN_BITS) + 1;
    while r.checked_mul(r).map_or(true, |sq| sq > v) {
        r = (r + v / r) / 2;
    }
    while (r + 1).checked_mul(r + 1).is_some_and(|sq| sq <= v) {
        r += 1;
    }
    r
}

/// The Newton seed is raised by `2^-SEED_MARGIN_BITS` of itself, well past
/// the error of a double-precision square root.
const SEED_MARGIN_BITS: u32 = 40;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_matches_u128_when_it_fits() {
        for (a, b) in [(0u128, 5u128), (3, 7), (u64::MAX as u128, u64::MAX as u128)] {
            assert_eq!(U256::mul(a, b), U256::from_u128(a * b));
        }
    }

    #[test]
    fn mul_carries_into_the_high_half() {
        let m = U256::mul(u128::MAX, u128::MAX); // (2^128 - 1)^2
        assert_eq!(
            m,
            U256 {
                hi: u128::MAX - 1,
                lo: 1
            }
        );
        assert_eq!(U256::mul(1u128 << 127, 4), U256 { hi: 2, lo: 0 });
    }

    #[test]
    fn shifts_and_bits() {
        let one = U256::from_u128(1);
        assert_eq!(one.shl(200).bits(), 201);
        assert_eq!(one.shl(200).shr(200), one);
        assert_eq!(one.shl(130).shr(3), one.shl(127));
        assert_eq!(U256::ZERO.bits(), 0);
    }

    #[test]
    fn isqrt_is_the_floor() {
        let max = u64::MAX as u128;
        for v in [
            0u128,
            1,
            2,
            3,
            4,
            15,
            16,
            17,
            1 << 100,
            (1 << 100) - 1,
            max * max,
            max * max - 1,
            u128::MAX,
        ] {
            let r = isqrt(v);
            assert!(
                r * r <= v && (r + 1).checked_mul(r + 1).map_or(true, |s| s > v),
                "isqrt({v}) = {r}"
            );
        }
    }
}
