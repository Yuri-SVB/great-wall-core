//! The zoom distance between two points of zoom space `(b, s)`: a barycentre
//! `b` and a size `s`. It is the hyperbolic metric of van Wijk & Nuij,
//!
//! ```text
//! d = (1/ρ) · arcosh(1 + q),   q = (ρ²·|b₁ − b₂|² + (s₁ − s₂)²) / (2·s₁·s₂)
//! ```
//!
//! computed in integers only. With `ρ = n/m`, `q = A / B` exactly, where
//! `A = n²·|Δb|² + m²·(Δs)²` and `B = 2·m²·s₁·s₂`, and
//!
//! ```text
//! arcosh(1 + q) = 2·ln(√(q/2) + √(q/2 + 1)) = 2·ln(√A + √(A + 2B)) − ln(2B)
//! ```
//!
//! with integer square roots and an integer base-2 logarithm. The module
//! knows nothing about features.

use super::params::Ratio;
use super::wide::{isqrt, U256};
use crate::fixed::Fixed;

/// A point of zoom space: barycentre `(re, im)` and size `s` (both Fixed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    pub re: Fixed,
    pub im: Fixed,
    pub size: Fixed,
}

/// ln 2 in Q64 (⌊ln 2 · 2⁶⁴⌋).
const LN_2_Q64: u128 = 0xB172_17F7_D1CF_79AB;
/// Fractional bits of [LN_2_Q64].
const LN_2_FRAC_BITS: u32 = 64;
/// Bits of the logarithm's working mantissa, which holds `[1, 2)`.
const MANTISSA_BITS: u32 = 64;
/// `A` and `A + 2B` are scaled by one power of four to at most this many bits:
/// their square roots then carry about half as many, and fit 128-bit
/// arithmetic.
const SCALED_BITS: u32 = 126;

/// `⌊log₂ x⌋` with `frac_bits` fractional bits, for `x > 0`.
fn log2(x: U256, frac_bits: u32) -> u128 {
    let bits = x.bits();
    debug_assert!(bits > 0, "log2 of zero");
    let int = (bits - 1) as u128;
    let top = if bits > MANTISSA_BITS {
        x.shr(bits - MANTISSA_BITS)
    } else {
        x.shl(MANTISSA_BITS - bits)
    };
    // m / 2^(MANTISSA_BITS - 1) is the mantissa in [1, 2).
    let mut m = top.lo;
    let mut frac = 0u128;
    for _ in 0..frac_bits {
        m = (m * m) >> (MANTISSA_BITS - 1);
        frac <<= 1;
        if m >> MANTISSA_BITS != 0 {
            frac |= 1;
            m >>= 1;
        }
    }
    (int << frac_bits) | frac
}

/// The zoom distance between `a` and `b` with `ρ = rho`, in units of
/// `2^-frac_bits`, rounded down. Both sizes must be positive.
pub fn distance(a: Point, b: Point, rho: Ratio, frac_bits: u32) -> u128 {
    debug_assert!(a.size.0 > 0 && b.size.0 > 0, "sizes must be positive");
    let (n, m) = (rho.num as u128, rho.den as u128);
    let dre = (a.re.0 as i128 - b.re.0 as i128).unsigned_abs();
    let dim = (a.im.0 as i128 - b.im.0 as i128).unsigned_abs();
    let ds = (a.size.0 as i128 - b.size.0 as i128).unsigned_abs();
    let (sa, sb) = (a.size.0 as u128, b.size.0 as u128);

    let a_ = U256::mul(n * dre, n * dre)
        .add(U256::mul(n * dim, n * dim))
        .add(U256::mul(m * ds, m * ds));
    let b2 = U256::mul(m * m * sa, sb).shl(2); // 2B
    let sum = a_.add(b2);
    // √A + √(A + 2B) = 2^k·(√(A·4^-k) + √((A + 2B)·4^-k)).
    let bits = sum.bits() as i64;
    let k = if bits > SCALED_BITS as i64 {
        (bits - SCALED_BITS as i64 + 1) / 2
    } else {
        -((SCALED_BITS as i64 - bits) / 2)
    };
    let scale = |v: U256| {
        if k >= 0 {
            v.shr((2 * k) as u32).lo
        } else {
            v.shl((-2 * k) as u32).lo
        }
    };
    let roots = U256::from_u128(isqrt(scale(a_)) + isqrt(scale(sum)));
    let twice_l1 = 2 * log2(roots, frac_bits) as i128 + ((2 * k as i128) << frac_bits);
    let log_ratio = (twice_l1 - log2(b2, frac_bits) as i128).max(0) as u128;
    ((log_ratio * LN_2_Q64) >> LN_2_FRAC_BITS) * m / n
}

#[cfg(test)]
mod tests {
    use super::*;

    const F: u32 = 32;
    const RHO1: Ratio = Ratio::new(1, 1);

    fn pt(re: f64, im: f64, s: f64) -> Point {
        Point {
            re: Fixed::from_f64(re),
            im: Fixed::from_f64(im),
            size: Fixed::from_f64(s),
        }
    }

    fn reference(a: Point, b: Point, rho: f64) -> f64 {
        let (ar, ai, as_) = (a.re.to_f64(), a.im.to_f64(), a.size.to_f64());
        let (br, bi, bs) = (b.re.to_f64(), b.im.to_f64(), b.size.to_f64());
        let d2 = (ar - br).powi(2) + (ai - bi).powi(2);
        let q = (rho * rho * d2 + (as_ - bs).powi(2)) / (2.0 * as_ * bs);
        (1.0 + q).acosh() / rho
    }

    fn approx(a: Point, b: Point, rho: Ratio) -> f64 {
        distance(a, b, rho, F) as f64 / (1u64 << F) as f64
    }

    #[test]
    fn log2_of_powers_and_others() {
        for k in [0u32, 1, 60, 127, 200] {
            let v = U256::from_u128(1).shl(k);
            assert_eq!(log2(v, F), (k as u128) << F);
        }
        let three = log2(U256::from_u128(3), F) as f64 / (1u64 << F) as f64;
        assert!((three - 3f64.log2()).abs() < 1e-9);
    }

    #[test]
    fn identical_points_are_at_zero() {
        let a = pt(-1.2, 0.3, 0.01);
        assert_eq!(distance(a, a, RHO1, F), 0);
    }

    #[test]
    fn zooming_by_k_costs_ln_k() {
        let a = pt(0.5, -0.25, 0.001);
        let b = pt(0.5, -0.25, 0.002);
        let d = approx(a, b, RHO1);
        assert!((d - 2f64.ln()).abs() < 1e-8, "d = {d}");
    }

    #[test]
    fn matches_the_float_formula() {
        let cases = [
            (pt(-1.5, 0.2, 0.3), pt(-1.4, 0.1, 0.02)),
            (pt(-2.0, -1.5, 1.8), pt(0.7, 1.2, 1e-6)),
            (pt(0.1, 0.1, 1e-5), pt(0.1001, 0.1, 1e-5)),
            (pt(-1.75, 0.0, 3.7), pt(1.4, 1.4, 3.7)),
        ];
        for rho in [Ratio::new(1, 1), Ratio::new(7, 5), Ratio::new(1, 3)] {
            for (a, b) in cases {
                let want = reference(a, b, rho.num as f64 / rho.den as f64);
                let got = approx(a, b, rho);
                assert!(
                    (got - want).abs() < 1e-7 * want.max(1.0),
                    "{a:?} {b:?}: {got} vs {want}"
                );
                assert_eq!(distance(a, b, rho, F), distance(b, a, rho, F), "symmetric");
            }
        }
    }

    #[test]
    fn invariant_under_scaling() {
        let (a, b) = (pt(-1.0, 0.5, 0.25), pt(-0.5, 0.25, 0.0625));
        let (a2, b2) = (pt(-0.5, 0.25, 0.125), pt(-0.25, 0.125, 0.03125));
        let (d, d2) = (approx(a, b, RHO1), approx(a2, b2, RHO1));
        assert!((d - d2).abs() < 1e-9, "{d} vs {d2}");
    }
}
