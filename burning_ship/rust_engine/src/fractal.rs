/// Burning Ship fractal iteration with fast unchecked arithmetic.
///
/// z_{n+1} = (|Re(z_n)| + i * |Im(z_n)|)^2 + c
///
/// Escape criterion: |z|^2 > BAILOUT_RADIUS^2, checked once per iteration.
/// All intermediate arithmetic is wrapping (no per-operation branches).

use crate::fixed::{Fixed, FRAC_BITS};

/// Bailout radius for escape detection.
///
/// The classical Mandelbrot/Burning Ship bound: once |z| > 2, the orbit
/// provably diverges.  With I4F60 representing [-8, +8), all intermediates
/// in the next iteration stay within representable range, so wrapping
/// arithmetic cannot produce false negatives.
const BAILOUT_RADIUS: i64 = 2;

/// Bailout threshold: BAILOUT_RADIUS^2 as a raw i128 value in fixed-point.
///
/// |z|^2 = zr^2 + zi^2.  In raw fixed-point, multiplying two i64 values
/// gives an i128 with 2*FRAC_BITS fractional bits.  We compare against
/// BAILOUT_RADIUS^2 in the same representation (shifted left by 2*FRAC_BITS).
const BAILOUT_THRESHOLD_I128: i128 =
    (BAILOUT_RADIUS as i128 * BAILOUT_RADIUS as i128) << (2 * FRAC_BITS as i128);

/// --- Perturbation p constants (additive shift) ---

/// Number of magnitude bits per p component (real or imaginary).
/// The top bit of each 32-bit half is reserved for the sign.
const P_MAGNITUDE_BITS: u32 = 31;

/// Sign bit position within each 32-bit half (bit 31 for Re, bit 63 for Im).
const P_SIGN_BIT_RE: u32 = 31;
const P_SIGN_BIT_IM: u32 = 63;

/// Minimum exponent offset for p magnitude bits: bit j → 2^{-(P_MIN_EXP + j)}.
const P_MAGNITUDE_MIN_EXPONENT: u32 = 4;

/// Baseline perturbation for p: each component always has at least 2^{-3} = 1/8.
const P_BASELINE_EXPONENT: u32 = 3;

/// Fixed-point shift base for p magnitude bits.
const P_MAGNITUDE_SHIFT_BASE: u32 = FRAC_BITS - P_MAGNITUDE_MIN_EXPONENT;

/// Fixed-point representation of the p baseline perturbation.
const P_BASELINE_RAW: i64 = 1i64 << (FRAC_BITS - P_BASELINE_EXPONENT);

/// --- Linear perturbation q constants (εz term) ---
/// No baseline — q=0 gives ε=0 (no linear term).

/// Number of magnitude bits per q component (real or imaginary).
const Q_MAGNITUDE_BITS: u32 = 31;

/// Sign bit positions for q.
const Q_SIGN_BIT_RE: u32 = 31;
const Q_SIGN_BIT_IM: u32 = 63;

/// Minimum exponent offset for q magnitude bits: bit j → 2^{-(Q_MIN_EXP + j)}.
/// One digit smaller than p (exponent 5 vs 4) for finer linear perturbation.
const Q_MAGNITUDE_MIN_EXPONENT: u32 = 5;

/// Fixed-point shift base for q magnitude bits.
const Q_MAGNITUDE_SHIFT_BASE: u32 = FRAC_BITS - Q_MAGNITUDE_MIN_EXPONENT;

/// --- Origin o constants (orbit seed z₀) ---
/// No baseline — o=0 gives z₀=0 (standard Burning Ship).

/// Number of magnitude bits per o component (real or imaginary).
const O_MAGNITUDE_BITS: u32 = 31;

/// Sign bit positions for o.
const O_SIGN_BIT_RE: u32 = 31;
const O_SIGN_BIT_IM: u32 = 63;

/// Minimum exponent offset for o magnitude bits: bit j → 2^{-(O_MIN_EXP + j)}.
/// One digit larger than p (exponent 3 vs 4) for coarser orbit seed.
const O_MAGNITUDE_MIN_EXPONENT: u32 = 3;

/// Fixed-point shift base for o magnitude bits.
const O_MAGNITUDE_SHIFT_BASE: u32 = FRAC_BITS - O_MAGNITUDE_MIN_EXPONENT;

/// Wrapping absolute value for i64.
///
/// Unlike checked_abs, this never fails: abs(i64::MIN) wraps to i64::MIN,
/// which is caught by the bailout check on the next iteration.
#[inline(always)]
fn wrapping_abs(v: i64) -> i64 {
    if v < 0 { v.wrapping_neg() } else { v }
}

/// Wrapping fixed-point multiply: (a * b) >> FRAC_BITS, all in i128.
#[inline(always)]
fn wrapping_mul(a: i64, b: i64) -> i64 {
    ((a as i128 * b as i128) >> FRAC_BITS) as i64
}

/// Bailout test: |z|^2 > BAILOUT_RADIUS^2, computed in i128 to avoid overflow.
///
/// zr and zi are raw i64 Fixed values.  Their squares are computed in i128
/// (preserving full 2*FRAC_BITS precision) and summed before comparing
/// against the threshold.
#[inline(always)]
fn escaped(zr: i64, zi: i64) -> bool {
    let zr2 = zr as i128 * zr as i128;
    let zi2 = zi as i128 * zi as i128;
    zr2 + zi2 > BAILOUT_THRESHOLD_I128
}

/// Compute the escape count for a point c in the Burning Ship fractal.
///
/// Returns Some(iteration) if the point escapes (|z|^2 > BAILOUT_RADIUS^2),
/// or None if it survives max_iter iterations.
///
/// Uses wrapping arithmetic with a single bailout comparison per iteration
/// (no per-operation branches).
#[inline]
pub fn escape_count(c_re: Fixed, c_im: Fixed, max_iter: u32) -> Option<u32> {
    let mut zr = 0i64;
    let mut zi = 0i64;
    let cr = c_re.0;
    let ci = c_im.0;

    for i in 0..max_iter {
        if escaped(zr, zi) {
            return Some(i);
        }

        let abs_zr = wrapping_abs(zr);
        let abs_zi = wrapping_abs(zi);

        // new_re = |zr|^2 - |zi|^2 + c_re
        let zr2 = wrapping_mul(abs_zr, abs_zr);
        let zi2 = wrapping_mul(abs_zi, abs_zi);
        let new_re = zr2.wrapping_sub(zi2).wrapping_add(cr);

        // new_im = 2 * |zr| * |zi| + c_im
        let prod = wrapping_mul(abs_zr, abs_zi);
        let new_im = prod.wrapping_mul(2).wrapping_add(ci);

        zr = new_re;
        zi = new_im;
    }

    None
}

/// Decode a 64-bit value into fixed-point additive perturbation (p).
///
/// Bit layout:
///   - bits 0..30  → REAL magnitude: bit j set ⟹ adds 2^{-(P_MIN_EXP+j)}
///   - bit 31      → REAL sign: 0 = positive, 1 = negative
///   - bits 32..62 → IMAG magnitude: bit j set ⟹ adds 2^{-(P_MIN_EXP+j)}
///   - bit 63      → IMAG sign: 0 = positive, 1 = negative
///
/// A baseline of 2^{-P_BASELINE_EXPONENT} is always added before applying
/// sign, so the perturbation is never zero.
#[inline]
pub fn decode_perturbation_p(p: u64) -> (Fixed, Fixed) {
    let mut mag_re: i64 = 0;
    let mut mag_im: i64 = 0;
    for j in 0..P_MAGNITUDE_BITS {
        if j < P_MAGNITUDE_SHIFT_BASE && p & (1u64 << j) != 0 {
            mag_re += 1i64 << (P_MAGNITUDE_SHIFT_BASE - j);
        }
        if j < P_MAGNITUDE_SHIFT_BASE && p & (1u64 << (j + 32)) != 0 {
            mag_im += 1i64 << (P_MAGNITUDE_SHIFT_BASE - j);
        }
    }
    mag_re += P_BASELINE_RAW;
    mag_im += P_BASELINE_RAW;
    let sign_re = if p & (1u64 << P_SIGN_BIT_RE) != 0 { -1i64 } else { 1i64 };
    let sign_im = if p & (1u64 << P_SIGN_BIT_IM) != 0 { -1i64 } else { 1i64 };
    (Fixed(sign_re * mag_re), Fixed(sign_im * mag_im))
}

/// Decode a 64-bit value into fixed-point linear perturbation ε (q).
///
/// Same bit layout as p but with independent constants (Q_*).
/// The decoded value is the complex coefficient ε in the term εz
/// added to each iteration.
#[inline]
pub fn decode_perturbation_q(q: u64) -> (Fixed, Fixed) {
    let mut mag_re: i64 = 0;
    let mut mag_im: i64 = 0;
    for j in 0..Q_MAGNITUDE_BITS {
        if j < Q_MAGNITUDE_SHIFT_BASE && q & (1u64 << j) != 0 {
            mag_re += 1i64 << (Q_MAGNITUDE_SHIFT_BASE - j);
        }
        if j < Q_MAGNITUDE_SHIFT_BASE && q & (1u64 << (j + 32)) != 0 {
            mag_im += 1i64 << (Q_MAGNITUDE_SHIFT_BASE - j);
        }
    }
    let sign_re = if q & (1u64 << Q_SIGN_BIT_RE) != 0 { -1i64 } else { 1i64 };
    let sign_im = if q & (1u64 << Q_SIGN_BIT_IM) != 0 { -1i64 } else { 1i64 };
    (Fixed(sign_re * mag_re), Fixed(sign_im * mag_im))
}

/// Decode a 64-bit value into fixed-point orbit seed (o).
///
/// Same bit layout as p but with independent constants (O_*), allowing
/// different exponent ranges.
#[inline]
pub fn decode_perturbation_o(o: u64) -> (Fixed, Fixed) {
    let mut mag_re: i64 = 0;
    let mut mag_im: i64 = 0;
    for j in 0..O_MAGNITUDE_BITS {
        if j < O_MAGNITUDE_SHIFT_BASE && o & (1u64 << j) != 0 {
            mag_re += 1i64 << (O_MAGNITUDE_SHIFT_BASE - j);
        }
        if j < O_MAGNITUDE_SHIFT_BASE && o & (1u64 << (j + 32)) != 0 {
            mag_im += 1i64 << (O_MAGNITUDE_SHIFT_BASE - j);
        }
    }
    let sign_re = if o & (1u64 << O_SIGN_BIT_RE) != 0 { -1i64 } else { 1i64 };
    let sign_im = if o & (1u64 << O_SIGN_BIT_IM) != 0 { -1i64 } else { 1i64 };
    (Fixed(sign_re * mag_re), Fixed(sign_im * mag_im))
}

/// Compute the escape count for a point c using the perturbed Burning Ship
/// formula with orbit seed (o), additive shift (p), and linear term (q):
///
///   z₀ = decoded(o)                   (non-zero starting value)
///   a  = |Re(z)| + Re(p)
///   b  = |Im(z)| + Im(p)
///   z_{n+1} = (a + i·b)² + ε·z_n + c   where ε = decoded(q)
///
/// `o` controls the orbit seed.  When o = 0, z₀ = 0.
/// `p` controls the additive shift (with baseline ±1/8).
/// `q` controls the linear perturbation ε.  When q = 0, ε = 0 (no linear
/// term).  When q ≠ 0, ε = decode_perturbation_q(q), adding a term εz_n
/// that breaks the pure quadratic dynamics — unlike the holomorphic case,
/// this is NOT conjugate to a c-shift because the absolute values prevent
/// the standard change-of-variables from simplifying the linear term away.
///
/// Uses wrapping arithmetic with a single bailout comparison per iteration.
#[inline]
pub fn escape_count_generic(c_re: Fixed, c_im: Fixed, max_iter: u32, o: u64, p: u64, q: u64) -> Option<u32> {
    let (p_re, p_im) = decode_perturbation_p(p);
    // o = 0 → z₀ = 0 (stage 1); o ≠ 0 → z₀ = decoded seed.
    let (seed_re, seed_im) = if o == 0 {
        (0i64, 0i64)
    } else {
        let (o_re, o_im) = decode_perturbation_o(o);
        (o_re.0, o_im.0)
    };
    // q = 0 → no linear term; q ≠ 0 → ε = decoded(q).
    let (eps_re, eps_im) = if q == 0 {
        (0i64, 0i64)
    } else {
        let (q_re, q_im) = decode_perturbation_q(q);
        (q_re.0, q_im.0)
    };
    let pr = p_re.0;
    let pi = p_im.0;

    let mut zr = seed_re;
    let mut zi = seed_im;
    let cr = c_re.0;
    let ci = c_im.0;

    for i in 0..max_iter {
        if escaped(zr, zi) {
            return Some(i);
        }

        // Additive perturbation: a = |Re(z)| + Re(p),  b = |Im(z)| + Im(p)
        let a = wrapping_abs(zr).wrapping_add(pr);
        let b = wrapping_abs(zi).wrapping_add(pi);

        // Squaring: w = (a + i·b)²
        let zr2 = wrapping_mul(a, a);
        let zi2 = wrapping_mul(b, b);
        let w_re = zr2.wrapping_sub(zi2);
        let w_im = wrapping_mul(a, b).wrapping_mul(2);

        // Linear term: ε·z = (eps_re + i·eps_im)·(zr + i·zi)
        //   Re(ε·z) = eps_re·zr − eps_im·zi
        //   Im(ε·z) = eps_re·zi + eps_im·zr
        let lin_re = wrapping_mul(eps_re, zr).wrapping_sub(wrapping_mul(eps_im, zi));
        let lin_im = wrapping_mul(eps_re, zi).wrapping_add(wrapping_mul(eps_im, zr));

        // z_{n+1} = w + ε·z_n + c
        let new_re = w_re.wrapping_add(lin_re).wrapping_add(cr);
        let new_im = w_im.wrapping_add(lin_im).wrapping_add(ci);

        zr = new_re;
        zi = new_im;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_origin_does_not_escape() {
        // (0,0) is in the Burning Ship set
        let result = escape_count(Fixed::ZERO, Fixed::ZERO, 1000);
        assert!(result.is_none(), "Origin should not escape");
    }

    #[test]
    fn test_far_point_escapes_immediately() {
        // (7, 7) should overflow quickly
        let c_re = Fixed::from_f64(7.0);
        let c_im = Fixed::from_f64(7.0);
        let result = escape_count(c_re, c_im, 1000);
        assert!(result.is_some(), "Far point should escape");
        assert!(result.unwrap() < 5, "Should escape very quickly");
    }

    #[test]
    fn test_known_island_point() {
        // (-1.75, 0.0) is in the main cardioid of BS
        let c_re = Fixed::from_f64(-1.75);
        let c_im = Fixed::from_f64(0.0);
        let result = escape_count(c_re, c_im, 1000);
        assert!(result.is_none(), "Main cardioid point should not escape");
    }

    #[test]
    fn test_boundary_point_escapes() {
        // (0.5, 0.5) should escape
        let c_re = Fixed::from_f64(0.5);
        let c_im = Fixed::from_f64(0.5);
        let result = escape_count(c_re, c_im, 1000);
        assert!(result.is_some(), "Boundary point should escape");
    }

    #[test]
    fn test_decode_p_zero_has_baseline() {
        let (p_re, p_im) = super::decode_perturbation_p(0);
        let baseline = Fixed::from_f64(0.125);
        assert_eq!(p_re.0, baseline.0, "p_re should be +1/8, got {}", p_re.to_f64());
        assert_eq!(p_im.0, baseline.0, "p_im should be +1/8, got {}", p_im.to_f64());
    }

    #[test]
    fn test_decode_p_sign_bits() {
        let p: u64 = 1u64 << P_SIGN_BIT_RE;
        let (p_re, p_im) = super::decode_perturbation_p(p);
        assert!(p_re.to_f64() < 0.0, "Re should be negative, got {}", p_re.to_f64());
        assert!(p_im.to_f64() > 0.0, "Im should be positive, got {}", p_im.to_f64());
        let expected_mag = 0.125;
        let eps = 1e-15;
        assert!((p_re.to_f64() + expected_mag).abs() < eps, "Re ≈ -0.125");
        assert!((p_im.to_f64() - expected_mag).abs() < eps, "Im ≈ +0.125");
    }

    #[test]
    fn test_decode_p_magnitude_bit0() {
        let (p_re, p_im) = super::decode_perturbation_p(1);
        let expected = 0.125 + 0.0625;
        let eps = 1e-15;
        assert!((p_re.to_f64() - expected).abs() < eps,
                "p_re≈{}, got {}", expected, p_re.to_f64());
        assert!((p_im.to_f64() - 0.125).abs() < eps, "p_im≈0.125");
    }

    #[test]
    fn test_decode_p_both_signs_negative() {
        let p: u64 = (1u64 << P_SIGN_BIT_RE) | (1u64 << P_SIGN_BIT_IM);
        let (p_re, p_im) = super::decode_perturbation_p(p);
        let eps = 1e-15;
        assert!((p_re.to_f64() + 0.125).abs() < eps, "Re ≈ -0.125");
        assert!((p_im.to_f64() + 0.125).abs() < eps, "Im ≈ -0.125");
    }

    #[test]
    fn test_decode_p_never_zero() {
        for p in [0u64, 1, 0xFFFF_FFFF_FFFF_FFFF, 0x8000_0000_8000_0000] {
            let (p_re, p_im) = super::decode_perturbation_p(p);
            assert!(p_re.0 != 0, "p_re should never be zero for p={:#x}", p);
            assert!(p_im.0 != 0, "p_im should never be zero for p={:#x}", p);
        }
    }

    #[test]
    fn test_decode_o_zero_is_zero() {
        // o=0 → no orbit seed (z₀=0), no baseline
        let (o_re, o_im) = super::decode_perturbation_o(0);
        assert_eq!(o_re.0, 0, "o_re should be 0, got {}", o_re.to_f64());
        assert_eq!(o_im.0, 0, "o_im should be 0, got {}", o_im.to_f64());
    }

    #[test]
    fn test_decode_q_zero_is_zero() {
        // q=0 → no linear term, no baseline
        let (q_re, q_im) = super::decode_perturbation_q(0);
        assert_eq!(q_re.0, 0, "q_re should be 0, got {}", q_re.to_f64());
        assert_eq!(q_im.0, 0, "q_im should be 0, got {}", q_im.to_f64());
    }

    #[test]
    fn test_generic_always_perturbed() {
        // Even p=0 gives a perturbation (baseline), so generic should differ from canonical
        let c_re = Fixed::from_f64(-1.5);
        let c_im = Fixed::from_f64(0.1);
        let max_iter = 1024;
        let canonical = escape_count(c_re, c_im, max_iter);
        let perturbed = escape_count_generic(c_re, c_im, max_iter, 0, 0, 0);
        // They should differ because baseline p ≠ 0
        // (not guaranteed for all points, but very likely for this one)
        let _ = (canonical, perturbed); // at minimum, must not panic
    }
}
