//! The orbit — `0.4.0` derivation chain: root, fractals, advance, master secret.
//!
//! Protocol `0.4.0` "orbit" (see `great-wall-docs/great-wall-core/DESIGN.md`
//! *Orbit Protocol* and
//! `great-wall-docs/next-steps/coercion-resistant-orbit-protocol.md`) replaces the
//! `0.3.0` single-fractal chain with an **orbit** rooted at the public Namtso salt
//! `σ`.  With the cheap hash `H` (SHA-256) and the memory-hard hash `H*`
//! (Argon2d, [`crate::argon2_hash`]):
//!
//! ```text
//! o_0       = H(σ)                          # orbit_root
//! theta_i_j = H(o_i ‖ j)                    # theta_digest  → (o,p,q) attribution downstream
//! K_i       = H(o_i ‖ Sh_i)   (i > 0)       # commitment / master_secret; K_i == c_i
//! o_{i+1}   = H*( K_i )                      # advance: wipe {o_i, Sh_i} before the long H*
//! ```
//!
//! `K_i` (the per-stage master secret) **coincides with the orbit-advance
//! commitment `c_i`**, so `o_{i+1} = H*(K_i)`.  One value, two roles:
//! *materialised* as the master secret at the boundary a setup truncates at, and
//! *consumed-then-wiped* while advancing through the stage — [`orbit_step`]
//! enforces that ordering by taking **owned [`Zeroizing`] inputs** and dropping
//! (zeroizing) them before the long `H*`, minimising the seizability window.
//!
//! There is no `K_0`: `o_0 = H(σ)` is public and stage-0 points are seizable, so a
//! stage-0 "master secret" carries no coercion resistance.  Master secrets start
//! at `K_1` (the entry-level Setup 1).
//!
//! `H` is a **frozen determinism pin** (SHA-256) — changing it changes the whole
//! orbit and every `K_i`, a protocol-version bump.

use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::argon2_hash::{argon2_single, Profile};

/// Width of an orbit point / digest, in bytes (SHA-256).
pub const ORBIT_POINT_LEN: usize = 32;

/// The cheap, non-entropy-collapsing hash `H` (SHA-256) over the concatenation of
/// `parts`, hashed incrementally so no combined plaintext buffer is ever formed.
pub fn h(parts: &[&[u8]]) -> [u8; ORBIT_POINT_LEN] {
    let mut hasher = Sha256::new();
    for p in parts {
        hasher.update(p);
    }
    hasher.finalize().into()
}

/// `o_0 := H(σ)` — the orbit root from the public Namtso salt `σ`.
pub fn orbit_root(sigma: &[u8]) -> [u8; ORBIT_POINT_LEN] {
    h(&[sigma])
}

/// `theta_i_j := H(o_i ‖ j)` — the 32-byte parameter digest of fractal `j` of the
/// stage at orbit point `o_i`.  The board index `j` is encoded big-endian (4
/// bytes).  The `(o, p, q)` byte-attribution over this digest is unchanged from
/// the prototype and applied downstream (DESIGN.md *Per-Fractal Parameter
/// Derivation*).
pub fn theta_digest(o_i: &[u8; ORBIT_POINT_LEN], j: u32) -> [u8; ORBIT_POINT_LEN] {
    h(&[o_i, &j.to_be_bytes()])
}

/// `K_i := H(o_i ‖ Sh_i)` — the per-stage master secret (`i > 0`), which also is
/// the orbit-advance commitment `c_i`.  `sh` is the serialized Shamir polynomial
/// (`crate::shamir::sh_to_bytes`), `r_i · 32` bits.
pub fn commitment(o_i: &[u8; ORBIT_POINT_LEN], sh: &[u8]) -> [u8; ORBIT_POINT_LEN] {
    h(&[o_i, sh])
}

/// Alias for [`commitment`] read in its master-secret role: `K_i = H(o_i ‖ Sh_i)`.
///
/// Cheap `H` deliberately — a memory-hard step here would prolong the window in
/// which `o_i` and the stage points are live; resistance lives in the stage's
/// `≥ 96`-bit entropy instead (DESIGN.md *Master Secret*).
pub fn master_secret(o_i: &[u8; ORBIT_POINT_LEN], sh: &[u8]) -> [u8; ORBIT_POINT_LEN] {
    commitment(o_i, sh)
}

/// Generic orbit advance: run `steps` sequential applications of the memory-hard
/// `hstar` starting from the commitment `c` (`o_{i+1} = H*(c)`, iterated).
/// `steps` is the durable derivation-step count `D` (≥ 1).  Generic over `hstar`
/// so the orchestration is testable without a real (multi-GiB) Argon2.
pub fn advance_with<F>(c: &[u8; ORBIT_POINT_LEN], steps: u32, hstar: F) -> [u8; ORBIT_POINT_LEN]
where
    F: Fn(&[u8]) -> [u8; ORBIT_POINT_LEN],
{
    assert!(steps >= 1, "derivation-step count D must be >= 1");
    let mut digest = hstar(c);
    for _ in 1..steps {
        digest = hstar(&digest);
    }
    digest
}

/// Concrete orbit advance backed by Argon2d ([`argon2_single`]) at the given
/// profile — `D` sequential passes, each feeding the previous digest forward.
pub fn advance_argon2(
    c: &[u8; ORBIT_POINT_LEN],
    steps: u32,
    profile: Profile,
) -> [u8; ORBIT_POINT_LEN] {
    advance_with(c, steps, |x| argon2_single(x, profile))
}

/// One orbit step with the wipe enforced by ownership: compute the master secret
/// `K_i = H(o_i ‖ Sh_i)` (instant), **drop (zeroize) the raw `{o_i, Sh_i}`**, then
/// run the long `H*` on `K_i` alone to get `o_{i+1}`.  Returns `(K_i, o_{i+1})`.
///
/// Generic over `hstar` for testing; see [`orbit_step`] for the Argon2 wrapper.
pub fn orbit_step_with<F>(
    o_i: Zeroizing<[u8; ORBIT_POINT_LEN]>,
    sh: Zeroizing<Vec<u8>>,
    steps: u32,
    hstar: F,
) -> ([u8; ORBIT_POINT_LEN], [u8; ORBIT_POINT_LEN])
where
    F: Fn(&[u8]) -> [u8; ORBIT_POINT_LEN],
{
    let k_i = commitment(&o_i, &sh); // c_i = K_i (instant)
    drop(o_i); // zeroized on drop — raw orbit point gone before the long H*
    drop(sh); // zeroized on drop — raw Shamir polynomial gone before the long H*
    let o_next = advance_with(&k_i, steps, hstar);
    (k_i, o_next)
}

/// [`orbit_step_with`] backed by Argon2d at `profile`.  The heavy step; not
/// exercised in unit tests (multi-GiB Argon2) — its orchestration is covered via
/// [`orbit_step_with`] with a cheap `hstar`.
pub fn orbit_step(
    o_i: Zeroizing<[u8; ORBIT_POINT_LEN]>,
    sh: Zeroizing<Vec<u8>>,
    steps: u32,
    profile: Profile,
) -> ([u8; ORBIT_POINT_LEN], [u8; ORBIT_POINT_LEN]) {
    orbit_step_with(o_i, sh, steps, |x| argon2_single(x, profile))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroize::Zeroize;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn h_matches_known_sha256_vector() {
        // Non-circular sanity: SHA-256("abc").
        assert_eq!(
            hex(&h(&[b"abc"])),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Multi-part hashing equals hashing the joined bytes (order-preserving).
        assert_eq!(h(&[b"ab", b"c"]), h(&[b"abc"]));
        // Distinct logical inputs differ.
        assert_ne!(h(&[b"abc"]), h(&[b"abd"]));
    }

    #[test]
    fn root_is_sha256_of_sigma_and_deterministic() {
        let sigma = [0xABu8; 128];
        assert_eq!(orbit_root(&sigma), orbit_root(&sigma));
        assert_eq!(orbit_root(&sigma), h(&[&sigma]));
        assert_ne!(orbit_root(&sigma), orbit_root(&[0xACu8; 128]));
    }

    #[test]
    fn theta_depends_on_orbit_point_and_board_index() {
        let o = [7u8; ORBIT_POINT_LEN];
        assert_eq!(theta_digest(&o, 0), h(&[&o, &0u32.to_be_bytes()]));
        assert_eq!(theta_digest(&o, 3), theta_digest(&o, 3));
        assert_ne!(theta_digest(&o, 0), theta_digest(&o, 1), "distinct boards differ");
        let o2 = [8u8; ORBIT_POINT_LEN];
        assert_ne!(theta_digest(&o, 0), theta_digest(&o2, 0), "distinct stages differ");
    }

    #[test]
    fn k_i_is_commitment_over_orbit_point_and_sh() {
        let o = [1u8; ORBIT_POINT_LEN];
        let sh = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]; // two GF(2^32) coeffs
        assert_eq!(master_secret(&o, &sh), commitment(&o, &sh));
        assert_eq!(master_secret(&o, &sh), h(&[&o, &sh]));
        // Different Sh (a different recalled stage) → different K_i.
        let sh2 = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x89];
        assert_ne!(master_secret(&o, &sh), master_secret(&o, &sh2));
    }

    #[test]
    fn advance_composes_hstar_step_times() {
        // Cheap hstar so we exercise the orchestration, not Argon2.
        let hstar = |x: &[u8]| h(&[x, b"HSTAR"]);
        let c = [0x42u8; ORBIT_POINT_LEN];

        assert_eq!(advance_with(&c, 1, hstar), hstar(&c));
        let expect3 = hstar(&hstar(&hstar(&c)));
        assert_eq!(advance_with(&c, 3, hstar), expect3);
        // Deterministic and step-sensitive.
        assert_eq!(advance_with(&c, 5, hstar), advance_with(&c, 5, hstar));
        assert_ne!(advance_with(&c, 4, hstar), advance_with(&c, 5, hstar));
    }

    #[test]
    fn orbit_step_computes_k_i_then_advances_from_it() {
        let hstar = |x: &[u8]| h(&[x, b"HSTAR"]);
        let o = [9u8; ORBIT_POINT_LEN];
        let sh_bytes = vec![0xDEu8, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];

        let expected_k = commitment(&o, &sh_bytes);
        let expected_next = advance_with(&expected_k, 3, hstar);

        let (k_i, o_next) = orbit_step_with(
            Zeroizing::new(o),
            Zeroizing::new(sh_bytes.clone()),
            3,
            hstar,
        );
        assert_eq!(k_i, expected_k, "K_i is the commitment over (o_i, Sh_i)");
        assert_eq!(o_next, expected_next, "o_{{i+1}} = H*(K_i), iterated");
    }

    #[test]
    fn zeroizing_wipes_the_buffer_we_rely_on() {
        // The wipe mechanism orbit_step depends on: zeroize clears the bytes.
        let mut secret = [0xFFu8; ORBIT_POINT_LEN];
        secret.zeroize();
        assert!(secret.iter().all(|&b| b == 0), "zeroize() must clear the buffer");

        let mut v = vec![0xABu8; 24];
        v.zeroize();
        assert!(v.iter().all(|&b| b == 0));
    }
}
