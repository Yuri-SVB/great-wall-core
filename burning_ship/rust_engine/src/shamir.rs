//! Shamir over GF(2^32) — the deep-stage share layer of the orbit protocol.
//!
//! Protocol `0.4.0` "orbit" (see `great-wall-docs/great-wall-core/DESIGN.md`
//! *Orbit Protocol* and `great-wall-docs/next-steps/coercion-resistant-orbit-protocol.md`)
//! turns a deep stage into a **board of `s_i` fractals**, each carrying a 32-bit
//! point `p_i_j`.  A stage's contribution to the orbit is **not** the raw
//! concatenation of those points and **not** the Shamir constant term `f(0)` — it
//! is the **full degree-`(t_i-1)` interpolating polynomial** `Sh_i`, i.e. all
//! `t_i` coefficients, worth exactly `t_i * 32` bits.
//!
//! Feeding the *whole* polynomial is load-bearing:
//!
//! * it makes every orbit link commit `>= t_i * 32 >= 64` bits, so **no single
//!   point `p_i_j` is ever checkable alone** (the block-orbit `k >= 2` condition,
//!   realised as Shamir threshold `t_i >= 2`), and
//! * it is **subset-invariant**: any `t_i` of the `s_i` points that lie on the
//!   stage polynomial reconstruct the *identical* coefficient vector, so the
//!   per-stage master secret `K_i = H(o_i || Sh_i)` is stable no matter which
//!   `t_i` shares the holder recalls.  This module's tests pin that property.
//!
//! A regression to the constant term (`f(0)`) would silently re-introduce a
//! single-point oracle — hence this module never exposes `f(0)` as "the secret".
//!
//! ## Abscissa convention
//!
//! The `t_i` **primary** (threshold-required) shares sit at **positive**
//! abscissae `1, 2, 3, ...` ([`primary_abscissa`]); opt-in
//! `(s_i - t_i)` **forgetting-resistance** shares sit at **"negative"** abscissae
//! (high bit set, [`resistance_abscissa`]).  Abscissa `0` is reserved (it would
//! expose the constant term) and is never a share point.  Extra
//! forgetting-resistance shares strengthen hard recovery against forgetting; they
//! do **not** change `Sh_i`'s entropy, which stays `t_i * 32` bits.
//!
//! ## Field (frozen determinism pin — bump `PROTOCOL_VERSION` on any change)
//!
//! GF(2^32) with reduction polynomial `x^32 + x^7 + x^3 + x^2 + 1`
//! ([`REDUCTION`] = `0x8D`), verified irreducible over GF(2).  Elements are
//! `u32`; the additive identity is `0`, the multiplicative identity is `1`, and
//! `add == sub == XOR` (characteristic 2).

/// Low bits of the GF(2^32) reduction polynomial `x^32 + x^7 + x^3 + x^2 + 1`.
///
/// When a multiplication carries out of bit 31 (an `x^32` term), it is folded
/// back in by XOR-ing this constant (`x^7 + x^3 + x^2 + 1 = 0b1000_1101`).  The
/// polynomial is irreducible over GF(2), so every non-zero element is invertible.
/// **Frozen:** changing it changes every `Sh_i`, hence every `K_i` — a
/// protocol-version bump.
pub const REDUCTION: u32 = 0x8D;

/// GF(2^32) multiplication (carry-less multiply, reduce by [`REDUCTION`]).
pub fn gf_mul(mut a: u32, mut b: u32) -> u32 {
    let mut acc: u32 = 0;
    while b != 0 {
        if b & 1 == 1 {
            acc ^= a;
        }
        let carry = a & 0x8000_0000; // x^31 term about to become x^32
        a <<= 1;
        if carry != 0 {
            a ^= REDUCTION;
        }
        b >>= 1;
    }
    acc
}

/// GF(2^32) exponentiation by square-and-multiply.
pub fn gf_pow(mut a: u32, mut e: u32) -> u32 {
    let mut acc: u32 = 1;
    while e != 0 {
        if e & 1 == 1 {
            acc = gf_mul(acc, a);
        }
        a = gf_mul(a, a);
        e >>= 1;
    }
    acc
}

/// GF(2^32) multiplicative inverse via Fermat: `a^(2^32 - 2)`.
///
/// The multiplicative group has order `2^32 - 1`, so `a^(-1) = a^(2^32 - 2)` for
/// every non-zero `a`.  Panics on `a == 0` (`0` has no inverse — and `0` is never
/// a valid abscissa, so it must not reach here from denominators).
pub fn gf_inv(a: u32) -> u32 {
    assert!(a != 0, "GF(2^32): zero has no multiplicative inverse");
    gf_pow(a, 0xFFFF_FFFE) // 2^32 - 2
}

/// Multiply polynomial `poly` (ascending coefficients) by the linear factor
/// `(x + c)`.  In GF(2^k), `x - c == x + c`.  Returns a vector one longer.
fn mul_by_linear(poly: &[u32], c: u32) -> Vec<u32> {
    let n = poly.len();
    let mut out = vec![0u32; n + 1];
    for (k, &pk) in poly.iter().enumerate() {
        out[k + 1] ^= pk; // x * poly[k]
        out[k] ^= gf_mul(c, pk); // c * poly[k]
    }
    out
}

/// Lagrange-interpolate the unique degree-`(t-1)` polynomial through `t` distinct
/// points and return its **full** coefficient vector `Sh` (ascending powers:
/// `coeffs[k]` is the coefficient of `x^k`), of length exactly `t`.
///
/// This is `Sh_i` for a deep stage: `xs` are the (distinct, non-zero) abscissae
/// and `ys` are the stage's 32-bit points `p_i_j`.  `coeffs.len() * 32` is the
/// stage's entropy in bits.  Panics if `xs`/`ys` differ in length, are empty, or
/// contain a repeated abscissa (which would make a denominator zero).
pub fn interpolate(xs: &[u32], ys: &[u32]) -> Vec<u32> {
    let t = xs.len();
    assert_eq!(t, ys.len(), "xs and ys must have equal length");
    assert!(t >= 1, "need at least one point");

    let mut acc = vec![0u32; t]; // coefficients c_0 .. c_{t-1}
    for j in 0..t {
        // numerator poly = prod_{m != j} (x + xs[m]); denom = prod_{m != j} (xs[j] + xs[m])
        let mut num = vec![1u32];
        let mut denom = 1u32;
        for m in 0..t {
            if m == j {
                continue;
            }
            let diff = xs[j] ^ xs[m];
            assert!(diff != 0, "duplicate abscissa: xs[{j}] == xs[{m}]");
            num = mul_by_linear(&num, xs[m]);
            denom = gf_mul(denom, diff);
        }
        let scale = gf_mul(ys[j], gf_inv(denom));
        for (k, &nk) in num.iter().enumerate() {
            acc[k] ^= gf_mul(scale, nk);
        }
    }
    acc
}

/// Evaluate a polynomial (ascending coefficients) at `x` by Horner's method.
pub fn eval(coeffs: &[u32], x: u32) -> u32 {
    let mut acc = 0u32;
    for &c in coeffs.iter().rev() {
        acc = gf_mul(acc, x) ^ c;
    }
    acc
}

/// Generate the share values `f(x)` at each abscissa in `abscissae` from a
/// polynomial's coefficients — used to compute the opt-in forgetting-resistance
/// points from a stage polynomial fixed by its primary points.
pub fn generate_shares(coeffs: &[u32], abscissae: &[u32]) -> Vec<u32> {
    abscissae.iter().map(|&x| eval(coeffs, x)).collect()
}

/// The `k`-th **primary** (threshold-required) abscissa: `k + 1` (positive;
/// abscissa `0` is reserved).  `k` is 0-based.
pub fn primary_abscissa(k: u32) -> u32 {
    k + 1
}

/// The `k`-th **forgetting-resistance** abscissa: `0x8000_0000 | (k + 1)` — the
/// "negative" half of the abscissa space, disjoint from the primary range.
pub fn resistance_abscissa(k: u32) -> u32 {
    0x8000_0000 | (k + 1)
}

/// Entropy of `Sh_i` in bits for threshold `t`: `t * 32`.
pub fn sh_bit_len(t: usize) -> usize {
    t * 32
}

/// Serialize a coefficient vector `Sh` to big-endian bytes (`4 * len` bytes),
/// the canonical wire form fed to the orbit's cheap hash `H` when computing
/// `o_{i+1} = H*(H(o_i || Sh_i))` and `K_i = H(o_i || Sh_i)`.
pub fn sh_to_bytes(coeffs: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(coeffs.len() * 4);
    for &c in coeffs {
        out.extend_from_slice(&c.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dependency-free deterministic PRNG (SplitMix64) so tests are reproducible
    /// without pulling in a `rand` crate.
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) & 0xFFFF_FFFF) as u32
        }
        fn nonzero_u32(&mut self) -> u32 {
            loop {
                let v = self.next_u32();
                if v != 0 {
                    return v;
                }
            }
        }
    }

    #[test]
    fn field_identities() {
        let mut r = SplitMix64(0x1234_5678_9ABC_DEF0);
        for _ in 0..10_000 {
            let a = r.next_u32();
            assert_eq!(gf_mul(a, 0), 0, "a * 0 == 0");
            assert_eq!(gf_mul(a, 1), a, "a * 1 == a");
            assert_eq!(a ^ a, 0, "a + a == 0 (char 2)");
        }
    }

    #[test]
    fn field_inverse_over_random_and_structured() {
        // If REDUCTION were reducible, some non-zero element would be a
        // zero-divisor and a * inv(a) == 1 would fail — so this doubles as an
        // irreducibility check across a large sample.
        let structured = [1u32, 2, 3, 0x8D, 0x8000_0000, 0xFFFF_FFFF, 0x0001_0001];
        for &a in &structured {
            assert_eq!(gf_mul(a, gf_inv(a)), 1, "structured a=0x{a:08X}");
        }
        let mut r = SplitMix64(0xDEAD_BEEF_CAFE_F00D);
        for _ in 0..50_000 {
            let a = r.nonzero_u32();
            assert_eq!(gf_mul(a, gf_inv(a)), 1, "random a=0x{a:08X}");
        }
    }

    #[test]
    fn field_associative_and_distributive() {
        let mut r = SplitMix64(0x0F0F_0F0F_0F0F_0F0F);
        for _ in 0..20_000 {
            let (a, b, c) = (r.next_u32(), r.next_u32(), r.next_u32());
            assert_eq!(gf_mul(gf_mul(a, b), c), gf_mul(a, gf_mul(b, c)), "assoc");
            assert_eq!(gf_mul(a, b ^ c), gf_mul(a, b) ^ gf_mul(a, c), "distrib");
            assert_eq!(gf_mul(a, b), gf_mul(b, a), "commut");
        }
    }

    #[test]
    fn interpolate_then_eval_roundtrips() {
        let mut r = SplitMix64(0xA5A5_5A5A_A5A5_5A5A);
        for t in 2..=8usize {
            let coeffs: Vec<u32> = (0..t).map(|_| r.next_u32()).collect();
            let xs: Vec<u32> = (0..t as u32).map(primary_abscissa).collect();
            let ys: Vec<u32> = xs.iter().map(|&x| eval(&coeffs, x)).collect();
            let got = interpolate(&xs, &ys);
            assert_eq!(got, coeffs, "recovered coeffs (t={t})");
            for (&x, &y) in xs.iter().zip(&ys) {
                assert_eq!(eval(&got, x), y, "eval matches share (t={t})");
            }
        }
    }

    #[test]
    fn subset_invariance_makes_k_i_stable() {
        // The load-bearing property: any t of s points that lie on the same
        // degree-(t-1) polynomial reconstruct the IDENTICAL Sh coefficient
        // vector, so K_i = H(o_i || Sh_i) does not depend on which shares the
        // holder recalls.
        let mut r = SplitMix64(0xC0FF_EE00_1234_5678);
        for t in 2..=5usize {
            let s = t + 3; // t primary + 3 forgetting-resistance shares
            let coeffs: Vec<u32> = (0..t).map(|_| r.nonzero_u32()).collect();

            // s distinct abscissae: t primary (positive) + (s - t) resistance (negative).
            let mut xs: Vec<u32> = (0..t as u32).map(primary_abscissa).collect();
            xs.extend((0..(s - t) as u32).map(resistance_abscissa));
            let ys: Vec<u32> = xs.iter().map(|&x| eval(&coeffs, x)).collect();

            // Reconstruct from several different t-subsets of the s shares.
            let subsets: [[usize; 5]; 4] = [
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5],
                [0, 2, 4, 5, 6],
                [2, 3, 4, 5, 6],
            ];
            for sub in &subsets {
                let idx = &sub[..t];
                let sx: Vec<u32> = idx.iter().map(|&i| xs[i]).collect();
                let sy: Vec<u32> = idx.iter().map(|&i| ys[i]).collect();
                let sh = interpolate(&sx, &sy);
                assert_eq!(sh, coeffs, "subset {idx:?} must recover the same Sh (t={t})");
                assert_eq!(sh_to_bytes(&sh), sh_to_bytes(&coeffs), "identical wire bytes");
            }
        }
    }

    #[test]
    fn sh_width_is_t_times_32_bits() {
        for t in 2..=8usize {
            let xs: Vec<u32> = (0..t as u32).map(primary_abscissa).collect();
            let ys: Vec<u32> = vec![0x1111_1111; t];
            let sh = interpolate(&xs, &ys);
            assert_eq!(sh.len(), t, "Sh has one coefficient per threshold share");
            assert_eq!(sh_bit_len(sh.len()), t * 32);
            assert_eq!(sh_to_bytes(&sh).len(), t * 4);
        }
    }

    #[test]
    fn abscissae_are_nonzero_and_disjoint() {
        for k in 0..1000u32 {
            let p = primary_abscissa(k);
            let n = resistance_abscissa(k);
            assert_ne!(p, 0, "primary abscissa must never be 0");
            assert_ne!(n, 0, "resistance abscissa must never be 0");
            assert_eq!(p & 0x8000_0000, 0, "primary is 'positive' (high bit clear)");
            assert_eq!(n & 0x8000_0000, 0x8000_0000, "resistance is 'negative' (high bit set)");
        }
    }
}
