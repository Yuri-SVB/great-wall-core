//! The orbit — `0.5.0` derivation chain: root, fractals, advance, master secret.
//!
//! Protocol `0.5.0` "orbit" (see `great-wall-docs/great-wall-core/DESIGN.md`
//! *Orbit Protocol* and
//! `great-wall-docs/next-steps/coercion-resistant-orbit-protocol.md`) replaces the
//! `0.3.0` single-fractal chain with an **orbit** rooted at the public Namtso salt
//! `σ`.  With the cheap hash `H` (SHA-512) and the memory-hard hash `H*`
//! (Argon2d, [`crate::argon2_hash`]):
//!
//! ```text
//! o_0       = H(σ)                          # orbit_root
//! theta_i_j = H(o_i ‖ j)                    # theta_digest  → (o,p,q) attribution downstream
//! u_i       = H(o_i ‖ Sh_i)                 # commitment — consumed by H*, then wiped
//! o_{i+1}   = H*( u_i )                     # advance: wipe {o_i, Sh_i} before the long H*
//! K_i       = TH(TAG_MASTER_SECRET, o_i ‖ Sh_i)   # stage master secret — internal
//! K_i^L     = TH(TAG_EXPORT_LABEL,  K_i ‖ L)      # the only exported key
//! ```
//!
//! **`u_i` and `K_i` are domain-separated.**  They were byte-identical through
//! `0.4.0`, which meant a leaked master secret yielded `o_{i+1} = H*(K_i)` and
//! with it every stage above — `N·D` memory-hard passes, well under the threat
//! model's prohibitive floor.  Recovering `u_i` from `K_i` now requires inverting
//! `H`.  See `HQ/next-steps/domain-separate-u-and-k.md`.
//!
//! [`orbit_step`] takes **owned [`Zeroizing`] inputs** and drops (zeroizes) them
//! before the long `H*`, minimising the window in which `{o_i, Sh_i}` are
//! seizable.  `K_i` is derived in that same instant and retained, so the export
//! path never needs `o_i` or `Sh_i` again.
//!
//! There is no `K_0`: `o_0 = H(σ)` is public and stage-0 points are seizable, so a
//! stage-0 "master secret" carries no coercion resistance.  Master secrets start
//! at `K_1` (the entry-level Setup 1).
//!
//! `H` is a **frozen determinism pin** (SHA-512) — changing it, or either domain
//! tag, changes the whole orbit and every `K_i`: a protocol-version bump.
//!
//! ## Entropy ceiling
//!
//! `H` is non-entropy-collapsing *for the practical purposes of protocol design*.
//! A fixed-width state cannot preserve unbounded entropy: setup tiers place
//! `64 + 96N` bits over `N` deep stages, which exceeds 512 at `N = 5` (level 6),
//! and past that the orbit's keyspace saturates at the state width.  This is
//! deliberate and harmless — the ceiling sits at `2^512` preimage / `2^256`
//! collision classically (`2^256` / `~2^170` under Grover and BHT), against a
//! model that already treats `2^96` cheap operations as prohibitive.  Stages
//! beyond saturation still buy memory-hard duration, tacit-recall structure and
//! Shamir thresholds; they simply stop buying keyspace.

use sha2::{Digest, Sha512};
use zeroize::Zeroizing;

use crate::argon2_hash::{orbit_argon2_single, Profile};

/// Width of an orbit point / digest, in bytes (SHA-512).
pub const ORBIT_POINT_LEN: usize = 64;

/// Domain tag for the stage master secret `K_i`.
pub const TAG_MASTER_SECRET: &[u8] = b"GreatWall/TGPO/master-secret/v1";

/// Domain tag for the exported, label-salted key `K_i^L`.
pub const TAG_EXPORT_LABEL: &[u8] = b"GreatWall/TGPO/export-label/v1";

/// The cheap hash `H` (SHA-512) over the concatenation of `parts`, hashed
/// incrementally so no combined plaintext buffer is ever formed.
///
/// Non-entropy-collapsing for the practical purposes of protocol design — see
/// the module's *Entropy ceiling* note for the bound and why it is unreachable.
pub fn h(parts: &[&[u8]]) -> [u8; ORBIT_POINT_LEN] {
    let mut hasher = Sha512::new();
    for p in parts {
        hasher.update(p);
    }
    hasher.finalize().into()
}

/// BIP-340-style tagged hash: `H( H(tag) ‖ H(tag) ‖ parts… )`.
///
/// The tag contributes a fixed 128-byte prefix whatever its length, so domain
/// separation is structural rather than a padding rule that can be tidied away.
/// `2 × 64 = 128` bytes is exactly one SHA-512 block, so the prefix midstate is
/// precomputable and the tag costs nothing per call.
pub fn tagged_h(tag: &[u8], parts: &[&[u8]]) -> [u8; ORBIT_POINT_LEN] {
    let t = h(&[tag]);
    let mut hasher = Sha512::new();
    hasher.update(t);
    hasher.update(t);
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

/// `u_i := H(o_i ‖ Sh_i)` — the orbit-advance commitment, consumed by `H*` and
/// then wiped.  `sh` is the serialized Shamir polynomial
/// (`crate::shamir::sh_to_bytes`), `r_i · 32` bits.
///
/// This is **not** the master secret. Through `0.4.0` the two shared these bytes;
/// `K_i` is now domain-separated ([`master_secret`]).
///
/// `o_i` is a fixed [`ORBIT_POINT_LEN`] bytes and `sh` is last, so the
/// concatenation is injective without length framing.
pub fn commitment(o_i: &[u8; ORBIT_POINT_LEN], sh: &[u8]) -> [u8; ORBIT_POINT_LEN] {
    h(&[o_i, sh])
}

/// `K_i := TH(TAG_MASTER_SECRET, o_i ‖ Sh_i)` — the stage master secret.
///
/// **Not** [`commitment`]: `u_i` and `K_i` are domain-separated, so neither is a
/// function of the other and both require `(o_i, Sh_i)`. Through `0.4.0` this was
/// an alias, which let a leaked `K_i` advance the orbit directly.
///
/// Cheap `H` deliberately — a memory-hard step here would prolong the window in
/// which `o_i` and the stage points are live; resistance lives in the stage's
/// `≥ 96`-bit entropy instead (DESIGN.md *Master Secret*).
///
/// Internal: the value handed to a holder is [`export_key`] of this, never this.
pub fn master_secret(o_i: &[u8; ORBIT_POINT_LEN], sh: &[u8]) -> [u8; ORBIT_POINT_LEN] {
    tagged_h(TAG_MASTER_SECRET, &[o_i, sh])
}

/// `K_i^L := TH(TAG_EXPORT_LABEL, K_i ‖ L)` — the exported key for label `L`.
///
/// Applied for **every** label including the empty one, so `K_i` itself never
/// leaves the engine. `label` must already be canonicalised
/// ([`crate::text::canonicalize_stage_text`]); the engine owns that rule so the
/// same label typed years apart yields the same key.
///
/// `K_i` is a fixed [`ORBIT_POINT_LEN`] bytes and the label is last, so the
/// concatenation is injective without length framing. **Appending any further
/// field would need explicit framing** — two adjacent variable-length fields are
/// not unambiguously decodable.
pub fn export_key(k_i: &[u8; ORBIT_POINT_LEN], label: &[u8]) -> [u8; ORBIT_POINT_LEN] {
    tagged_h(TAG_EXPORT_LABEL, &[k_i, label])
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
    advance_with(c, steps, |x| orbit_argon2_single(x, profile))
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
    // Both cheap hashes happen in this instant, before the wipe: K_i is retained
    // for export, u_i is consumed by the long H* below. They are domain-separated,
    // so neither yields the other.
    let k_i = master_secret(&o_i, &sh);
    let u_i = Zeroizing::new(commitment(&o_i, &sh));
    drop(o_i); // zeroized on drop — raw orbit point gone before the long H*
    drop(sh); // zeroized on drop — raw Shamir polynomial gone before the long H*
    let o_next = advance_with(&u_i, steps, hstar);
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
    orbit_step_with(o_i, sh, steps, |x| orbit_argon2_single(x, profile))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroize::Zeroize;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn h_matches_known_sha512_vector() {
        // Non-circular sanity: SHA-512("abc"), FIPS 180-4 appendix.
        assert_eq!(
            hex(&h(&[b"abc"])),
            "ddaf35a193617abacc417349ae204131\
             12e6fa4e89a97ea20a9eeee64b55d39a\
             2192992a274fc1a836ba3c23a3feebbd\
             454d4423643ce80e2a9ac94fa54ca49f"
                .replace(char::is_whitespace, "")
        );
        assert_eq!(h(&[b"abc"]).len(), ORBIT_POINT_LEN, "512-bit state");
    }

    #[test]
    fn tagged_h_separates_domains() {
        let m = b"same message";
        assert_ne!(
            tagged_h(TAG_MASTER_SECRET, &[m]),
            tagged_h(TAG_EXPORT_LABEL, &[m]),
            "different tags must give different digests"
        );
        assert_ne!(
            tagged_h(TAG_MASTER_SECRET, &[m]),
            h(&[m]),
            "a tagged hash is never the bare hash"
        );
        // The tag prefix is exactly one SHA-512 block, whatever the tag length.
        let t = h(&[TAG_MASTER_SECRET]);
        assert_eq!(t.len() * 2, 128, "2 x 64 bytes = one SHA-512 block");
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
    fn k_i_is_domain_separated_from_u_i() {
        let o = [1u8; ORBIT_POINT_LEN];
        let sh = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]; // two GF(2^32) coeffs

        // The 0.4.0 defect, now a regression test: these were the same bytes, so
        // a leaked K_i advanced the orbit directly.
        assert_ne!(master_secret(&o, &sh), commitment(&o, &sh),
                   "K_i must not be u_i");
        assert_eq!(commitment(&o, &sh), h(&[&o, &sh]), "u_i keeps the bare formula");
        assert_eq!(master_secret(&o, &sh), tagged_h(TAG_MASTER_SECRET, &[&o, &sh]));

        // Different Sh (a different recalled stage) → different K_i and u_i.
        let sh2 = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x89];
        assert_ne!(master_secret(&o, &sh), master_secret(&o, &sh2));
        assert_ne!(commitment(&o, &sh), commitment(&o, &sh2));
    }

    #[test]
    fn export_key_is_applied_for_every_label_including_empty() {
        let o = [1u8; ORBIT_POINT_LEN];
        let sh = [0xAAu8; 8];
        let k = master_secret(&o, &sh);

        // K_i itself is never what a holder receives.
        assert_ne!(export_key(&k, b""), k, "the empty label is still derived");
        assert_ne!(export_key(&k, b"MAIN-STASH"), k);
        // Distinct labels give distinct keys; the same label is stable.
        assert_ne!(export_key(&k, b"MAIN-STASH"), export_key(&k, b"COLD-STASH"));
        assert_eq!(export_key(&k, b"MAIN-STASH"), export_key(&k, b"MAIN-STASH"));
        assert_ne!(export_key(&k, b""), export_key(&k, b"MAIN-STASH"));
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
    fn orbit_step_returns_k_i_and_advances_from_u_i() {
        let hstar = |x: &[u8]| h(&[x, b"HSTAR"]);
        let o = [9u8; ORBIT_POINT_LEN];
        let sh_bytes = vec![0xDEu8, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];

        let expected_k = master_secret(&o, &sh_bytes);
        let expected_u = commitment(&o, &sh_bytes);
        let expected_next = advance_with(&expected_u, 3, hstar);

        let (k_i, o_next) = orbit_step_with(
            Zeroizing::new(o),
            Zeroizing::new(sh_bytes.clone()),
            3,
            hstar,
        );
        assert_eq!(k_i, expected_k, "K_i is the tagged master secret");
        assert_eq!(o_next, expected_next, "o_{{i+1}} = H*(u_i), iterated");
        // The advance must NOT be reachable from the returned key.
        assert_ne!(o_next, advance_with(&k_i, 3, hstar),
                   "a leaked K_i must not advance the orbit");
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
